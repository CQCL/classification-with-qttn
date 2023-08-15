from array import array
from pathlib import Path
from typing import List
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Ket, Bra
from discopy.quantum.circuit import Box, Measure, Discard
import tensornetwork as tn
from jax import vmap, value_and_grad, jit, vmap
from sklearn.utils import gen_batches
from tqdm import tqdm
import time 
import pickle
import numpy as np
import pickle
import optax
from functools import partial
import yaml

from datetime import datetime

from ansatz import apply_box, make_density_matrix, make_state_vector
from ansatz import IQPAnsatz, Ansatz9, Ansatz14

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)

with open('CTN_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)

n_qubits = conf['n_qubits'] # number of qubits per word
n_layers = conf['n_layers']
batch_size = conf['batch_size']
lr = conf['lr']

if conf['post_sel']:
    box_vec = make_state_vector
else:
    box_vec = make_density_matrix
discard = not conf['post_sel']

parse_type = conf['parse_type']

# ------------------------------- READ IN DATA ----------------------------- #
load_path = f'../Data/CTN/{conf["data_name"]}/'

w2i = pickle.load(file=open(f'{load_path}w2i', 'rb'))
train_data = pickle.load(file=open(f'{load_path}train_data', 'rb'))
val_data = pickle.load(file=open(f'{load_path}val_data', 'rb'))
test_data = pickle.load(file=open(f'{load_path}test_data', 'rb'))

n_train = np.sum([len(t) for t in train_data["labels"]])
n_val = np.sum([len(t) for t in val_data["labels"]])
n_test = np.sum([len(t) for t in test_data["labels"]])

print("Number of train examples: ", n_train) 
print("Number of val examples: ", n_val) 
print("Number of test examples: ", n_test)
# ------------------------------- READ IN DATA ----------------------------- #

if conf['ansatz'] == 'IQP':
    ansatz = IQPAnsatz(conf['n_layers'], discard=discard)
elif conf['ansatz'] == 'A9':
    ansatz = Ansatz9(conf['n_layers'], discard=discard)
elif conf['ansatz'] == 'A14':
    ansatz = Ansatz14(conf['n_layers'], discard=discard)
else:
    raise ValueError

if not conf['use_jit']: 
    jit = lambda x: x

eval_args = {
    'mixed': discard, # use density matrices if discarding 
    'contractor': tn.contractors.auto # use tensor networks if speed
}

def flatten_list(li):
    return [y for x in li for y in x]

def word_vec_init(word_params):
    circ = ansatz(dom=0, cod=n_qubits, params=word_params)
    return circ.eval(**eval_args).array

def uCTN(W_params, U_params, I_params, class_params, ns):
    word_vecs = vmap(word_vec_init)(W_params)

    words = [box_vec(vec, n_qubits) for vec in word_vecs]
    circ = Id.tensor(*words)

    for n in ns:

        # apply unitary ops
        for i in range(n_qubits, n-n_qubits)[::2*n_qubits]:
            U_box = ansatz(2 * n_qubits, 2 * n_qubits, U_params)
            circ = apply_box(circ, U_box, i)

        # apply isometry
        for i in range(n // 2 // n_qubits):
            I_box = ansatz(2 * n_qubits, n_qubits, I_params)
            circ = apply_box(circ, I_box, i * n_qubits)

    # apply final classification ansatz
    circ >>= ansatz(n_qubits, n_qubits, class_params)

    # measure the middle qubit
    eff = Bra(0) if conf['post_sel'] else Discard()
    boxes = [Id(1) if i == n_qubits // 2 else eff for i in range(n_qubits)]
    circ >>= Id.tensor(*boxes)
    
    wire_state = box_vec(circ.eval(**eval_args).array, 1)
    wire_state >>= Measure()

    pred = wire_state.eval().array
    pred = jnp.array(pred, jnp.float64)

    return pred

def hCTN(W_params, U_params, I_params, class_params, ns):
    word_vecs = vmap(word_vec_init)(W_params)
    words = [box_vec(vec, n_qubits) for vec in word_vecs]
    circ = Id.tensor(*words)

    for idx, n in enumerate(ns):  

        # apply unitary ops
        for i in range(n_qubits, n-n_qubits)[::2*n_qubits]:
            U_box = ansatz(2 * n_qubits, 2 * n_qubits, U_params[idx])
            circ = apply_box(circ, U_box, i)

        # apply isometry
        for i in range(n // 2 // n_qubits): 
            I_box = ansatz(2 * n_qubits, n_qubits, I_params[idx])
            circ = apply_box(circ, I_box, i * n_qubits)

    # apply final classification ansatz
    circ >>= ansatz(n_qubits, n_qubits, class_params)

    # measure the middle qubit
    eff = Bra(0) if conf['post_sel'] else Discard()
    boxes = [Id(1) if i == n_qubits // 2 else eff for i in range(n_qubits)]
    circ >>= Id.tensor(*boxes)
    
    wire_state = box_vec(circ.eval(**eval_args).array, 1)
    wire_state >>= Measure()

    pred = wire_state.eval().array
    pred = jnp.array(pred, jnp.float64)

    return pred

CTN = uCTN if parse_type == 'unibox' else hCTN
vmap_contract = vmap(CTN, (0, None, None, None, None))

def get_preds(params, batch_words, ns):
    b_params = params['words'][batch_words]
    b_Us = params['Us']
    b_Is = params['Is']
    b_class = params['class']
    preds = vmap_contract(b_params, b_Us, b_Is, b_class, ns)

    if conf['post_sel']:
        preds = preds / jnp.sum(preds, axis=1)[:,None]

    if not conf['use_jit']:
        assert all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

    return preds

def get_loss(params, batch_words, labels, ns):
    preds = get_preds(params, batch_words, ns)
    out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, jnp.array(labels))])

    return -jnp.sum(out)

val_n_grad = partial(jit, static_argnums=(3,))(value_and_grad(get_loss))

def train_step(params, opt_state, batch_words, batch_class, ns):
    cost, grads = val_n_grad(params, batch_words, batch_class, ns)

    if conf['use_grad_clip']: 
        for k in grads:
            grads[k] = jnp.clip(grads[k], -conf['grad_clip'], conf['grad_clip'])

    if conf['use_optax_reg']:
        updates, opt_state = optimizer.update(grads, opt_state, params)
    else:            
        updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return cost, params, opt_state

def get_accs(params, batch_words, labels, ns):
    preds = get_preds(params, batch_words, ns)
    acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2

    return acc

def random(size):
    return jnp.array(np.random.uniform(0, conf['init_val'], size=size))

def get_batches(words, labels, batch_size):
    slices = list(gen_batches(len(labels), batch_size))
    batched_w = [words[s] for s in slices]
    batched_l = [labels[s] for s in slices]

    return zip(batched_w, batched_l)

n_words = max(w2i.values())
print("Number of unique tokens: ", n_words+1)

word_emb_size = ansatz.n_params(n_qubits)
rule_emb_size = ansatz.n_params(2 * n_qubits)

if conf['use_optax_reg'] is True:
    optimizer = optax.adamw(conf['lr'])
else:
    optimizer = optax.adam(conf['lr'])

n_rules = 1 << len(train_data["labels"]) + 1

word_params = random((n_words+1, word_emb_size))
class_params = random(word_emb_size)

if parse_type == 'height':
    U_params = random((n_rules, rule_emb_size))
    I_params = random((n_rules, rule_emb_size))
elif parse_type == 'unibox':
    U_params = random(rule_emb_size)
    I_params = random(rule_emb_size)
else:
    raise ValueError("Parse type not recognised")

params = {'words': word_params, 'Us': U_params, 'Is': I_params, 'class': class_params}
opt_state = optimizer.init(params)

val_accs = []
test_accs = []
losses = []
all_params = []

n_max = len(train_data['words'][-1][0])
N = n_max * n_qubits
ns = tuple([int(N / (jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))]) 

def evaluate(data, n):
    acc = 0
    for i, (words, labels) in enumerate(zip(data['words'], data['labels'])):
        if len(words):
            n_max = int(np.power(2, i+1))
            N = n_max*n_qubits
            ns = tuple([int(N/(jnp.power(2, j))) for j in range(i+1)])
            for batch_words, batch_labels in get_batches(words, labels, conf['batch_size']):
                batch_words = np.array(batch_words, np.int64)
                batch_acc = get_accs(params, batch_words, batch_labels, ns)
                acc += batch_acc / n
    return acc 

val_acc = evaluate(val_data, n_val)
print("Initial acc  {:0.2f}  ".format(val_acc))
val_accs.append(val_acc)

for epoch in range(conf['n_epochs']):

    start_time = time.time()

    # calc cost and update params
    loss = 0
    for i, (words, labels) in enumerate(zip(train_data['words'], train_data['labels'])):
        if len(words):
            n_max = int(np.power(2, i+1))
            N = n_max*n_qubits
            ns = tuple([int(N/(jnp.power(2, j))) for j in range(i+1)])
            for batch_words, batch_labels in get_batches(words, labels, batch_size):
                batch_words = np.array(batch_words, np.int64)
                cost, params, opt_state = train_step(params, opt_state, batch_words, batch_labels, ns)
                loss += cost / n_train

    losses.append(loss)
    all_params.append(params)

    print("Loss  {:0.2f}".format(loss))
    val_acc = evaluate(val_data, n_val)
    print("Val Acc  {:0.2f}".format(val_acc))
    val_accs.append(val_acc)

    test_acc = evaluate(test_data, n_test)
    print("Test Acc: ", test_acc)
    test_accs.append(test_acc)

    epoch_time = time.time() - start_time
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

    save_dict = {
        'params_dict': all_params,
        'opt_state': opt_state,
        'test_accs': test_accs,
        'val_accs': val_accs,
        'losses': losses,
        'w2i': w2i
    }

    # ------------------------------ SAVE DATA -----------------–------------ #
    timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f'../Results/CTN/{conf["data_name"]}/{conf["parse_type"]}/{timestr}/'
    for key, value in conf.items():
        full_save_path = f'{save_path}{key}'
        Path(full_save_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(obj=value, file=open(f'{full_save_path}/{key}', 'wb'))
