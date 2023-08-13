from pathlib import Path
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Bra
from discopy.quantum.circuit import Measure, Discard
import tensornetwork as tn
from jax import vmap, value_and_grad, jit, vmap
from sklearn.utils import gen_batches
from tqdm import tqdm
import time 
import pickle
import numpy as np
import pickle
import optax
from datatime import datetime

import yaml

from ansatz import apply_box, make_density_matrix, make_state_vector
from ansatz import IQPAnsatz, Ansatz9, Ansatz14

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)

with open('SCTN_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)

n_qubits = conf['n_qubits'] # number of qubits per word

if conf['post_sel']:
    box_vec = make_state_vector
else:
    box_vec = make_density_matrix

# ------------------------------------- SETTINGS -------------------------------- #
data_name = 'genome' 
post_sel = False
window_size = 4

thr = 50
pos_shift = False
model = 'CTN'
use_jit = False
use_grad_clip = True
use_optax_reg = True
include_final_classification_box = True
grad_clip = 100.0
n_epochs = 10
batch_size = 64
init_val = 0.001
ansatz = 'A14' 
# ------------------------------------- SETTINGS -------------------------------- #

parse_type = 'unibox'
n_qubits = 1
n_layers = 1
lr = 0.01


# ------------------------------- READ IN DATA ---------------------------- #
print("Reading in data ... ")
print("Load data")

save_path = f'Data/CTN_SLIDE_{window_size}/{data_name}/'

w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
train_data = pickle.load(file=open(f'{save_path}{"train_data"}', 'rb'))
val_data = pickle.load(file=open(f'{save_path}{"val_data"}', 'rb'))
test_data = pickle.load(file=open(f'{save_path}{"test_data"}', 'rb'))

print("Number of train examples: ", len(train_data['labels']))
print("Number of val examples: ", len(val_data['labels']))
print("Number of test examples: ", len(test_data['labels']))
# ------------------------------- READ IN DATA ----------------------------- #

                
discard = not conf['post_sel']

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

print("Model: ", model)
print("window size: ", window_size)
print("Parse: ", parse_type)
print("Number of epochs: ", n_epochs)
print("Batch size: ", batch_size)
print("Ansatz: ", ansatz)
print("Number of word qubits: ", n_qubits)
print("Number of layers: ", n_layers)
print("Using post selection: ", post_sel)
print("Include classification box: ", include_final_classification_box)
print("Using gradient clipping: ", use_grad_clip)

def flatten_list(li):
    return [y for x in li for y in x]

def word_vec_init(word_params):
    circ = ansatz(dom=0, cod=n_qubits, params=word_params)
    return circ.eval(**eval_args).array

single_batch_vec_init = jit(vmap(word_vec_init)) # vmap over initial states in a tree

def uCTN(W_params, U_params, I_params, class_params, ns):
    word_vecs = vmap(word_vec_init)(W_params)

    words = [box_vec(vec, "w") for vec in word_vecs]
    circ = Id.tensor(*words)

    for n in ns:

        # apply unitary ops
        for i in range(n_qubits, n-n_qubits)[::2*n_qubits]:
            U_box = ansatz(2 * n_qubits, 2 * n_qubits, U_params)
            circ = apply_box(circ, U_box, i)

        # apply isometry
        for i in range(n-n_qubits)[::2*n_qubits]:
            I_box = ansatz(2 * n_qubits, n_qubits, I_params)
            circ = apply_box(circ, I_box, i)

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
        for i in range(n-n_qubits)[::2*n_qubits]:
            I_box = ansatz(2 * n_qubits, n_qubits, I_params[idx])
            circ = apply_box(circ, I_box, i)

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

def get_loss(params, batch_words, batch_inds, labels, ns):
    preds = get_preds(params, batch_words, ns)
    preds = jnp.array([jnp.average(preds[batch_inds[i]:batch_inds[i+1]], axis=0) for i in range(len(batch_inds)-1)])

    out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, jnp.array(labels))])

    return  -jnp.sum(out)


val_n_grad = value_and_grad(get_loss)

def train_step(params, opt_state, batch_words, batch_labels, batch_inds, ns):
    cost, grads = val_n_grad(params, batch_words, batch_inds, batch_labels, ns)

    if use_grad_clip: 
        for k in grads:
            grads[k] = jnp.clip(grads[k], -grad_clip, grad_clip)

    if conf['use_optax_reg']:
        updates, opt_state = optimizer.update(grads, opt_state, params)
    else:            
        updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return cost, params, opt_state

def get_preds(params, batch_words, ns):
    b_params = params['words'][batch_words]
    b_Us = params['Us']
    b_Is = params['Is']
    b_class = params['class']
    preds = vmap_contract(b_params, b_Us, b_Is, b_class, ns)

    if conf['post_sel']:
        preds = preds / jnp.sum(preds, axis=1)[:,None] # renormalise output

    if not conf['use_jit']:
        assert all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

    return preds
    
def get_accs(params, batch_words, ns, counts, labels):
    preds = get_preds(params, batch_words, ns)
    preds = jnp.array([jnp.average(preds[counts[i]:counts[i+1]], axis=0) for i in range(len(counts)-1)])
    acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2

    return acc

def random(size):
    return jnp.array(np.random.uniform(0, conf['init_val'], size=size))

# TODO do we need these conversions?
def get_batches(words, counts, labels, batch_size):
    for s in gen_batches(len(words), batch_size):
        word = np.array(words[s], np.int64)
        count = np.array(counts[s], np.int64)
        label = np.array(labels[s], np.int64)
        yield word, count, label

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

no_rules = int(np.log2(window_size)+1)

params = {'words': word_params,'Us': U_params, 'Is': I_params, 'class': class_params}
opt_state = optimizer.init(params)

N = window_size * n_qubits
ns = tuple([int(N/(1 << i)) for i in range(int(jnp.log2(window_size)))])

val_accs = []
test_accs = []
losses = []
all_params = []


sum_accs = []

def evaluate(data, n):
    acc = 0
    batches = get_batches(data['words'], data['counts'], data['labels'], batch_size)
    for batch_words, batch_counts, batch_labels in batches: 
        batch_counts = np.concatenate(([0], batch_counts))
        cum_inds = tuple(np.array([np.cumsum(batch_counts[:i+1])[-1] for i in range(len(batch_counts))], np.int64))
        try:
            batch_words = jnp.array(flatten_list(batch_words), jnp.int64)
        except:
            batch_words = jnp.array(batch_words, jnp.int64)
        acc += get_accs(params, batch_words, ns, cum_inds, batch_labels) / n
    return acc

val_acc = evaluate(val_data, len(val_data['words']))
print("Initial acc  {:0.2f}  ".format(val_acc))
val_accs.append(val_acc)

for epoch in range(conf['n_epochs']):
    start_time = time.time()

    loss = 0
    batches = get_batches(train_data['words'], train_data['counts'], train_data['labels'], batch_size)
    for batch_words, batch_counts, batch_labels in tqdm(batches): # this was 16 untul 30th for sm reason!!!
        batch_counts.insert(0,0)
        cum_inds = tuple(np.array([np.cumsum(batch_counts[:i+1])[-1] for i in range(len(batch_counts))], np.int64)) 
        try:
            batch_words = jnp.array(flatten_list(batch_words), jnp.int64)
        except:
            batch_words = jnp.array(batch_words, jnp.int64)
        cost, params, opt_state = train_step(params, opt_state, batch_words, batch_labels, cum_inds, ns)
        loss += cost / len(train_data['labels'])

    losses.append(loss)
    all_params.append(params)

    val_acc = evaluate(val_data, len(val_data['words']))
    val_accs.append(val_acc)
    epoch_time = time.time() - start_time
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Loss  {:0.2f}".format(loss))
    print("Val Acc  {:0.2f}".format(val_acc))

    test_acc = evaluate(test_data, len(test_data['words']))
    test_accs.append(test_acc)
    print("Test acc  {:0.2f}".format(test_acc))

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
    save_path = f'../Results/{timestr}/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for key, value in save_dict.items():
        full_save_path = f'{save_path}{key}'
        pickle.dump(obj=value, file=open(f'{full_save_path}/{key}', 'wb'))
    # ------------------------------- SAVE DATA ---------------–------------ #
