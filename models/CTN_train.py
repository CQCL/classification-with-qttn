from pathlib import Path
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
model = 'CTN'
thr = 16
number_of_structures = 100
post_sel = True
use_jit = True
use_grad_clip = True
use_optax_reg = True
include_final_classification_box = True
grad_clip = 100.0
no_epochs = 25
batch_size = 64
init_val = 0.01
ansatz = 'A14' 
n_qubits = 1 # number of qubits per word
# ------------------------------------- SETTINGS -------------------------------- #

parse_type = 'unibox' # 'height' or 'unibox'

load_path = f'Data/{model}/{data_name}/'


w2i = pickle.load(file=open(f'{load_path}w2i', 'rb'))
train_data = pickle.load(file=open(f'{load_path}train_data', 'rb'))
val_data = pickle.load(file=open(f'{load_path}val_data', 'rb'))
test_data = pickle.load(file=open(f'{load_path}test_data', 'rb'))

n_train = sum([len(labels["labels"]) for labels in train_data])
print("Number of train examples: ", n_train) 
n_val = sum([len(labels["labels"]) for labels in val_data])
print("Number of val examples: ", n_val) 
n_test = sum([len(labels["labels"]) for labels in test_data])
print("Number of test examples: ", n_test)

# ------------------------------- READ IN DATA ----------------------------- #

n_qubits = 1
n_layers = 1
lr = 0.01

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


def word_vec_init(word_params):
    circ = ansatz(dom=0, cod=n_qubits, params=word_params)
    return circ.eval(**eval_args).array

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

n_max = thr if thr else 64
N = n_max * n_qubits
ns = tuple([int(N / (jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))]) 

def evaluate(data, n):
    acc = 0
    for d in tqdm(data):
        if len(d["labels"]) == 0:
            continue
        batches = get_batches(d["words"], d["labels"], conf['batch_size'])
        for batch_words, batch_labels in batches:
            batch_acc = get_accs(params, batch_words, batch_labels, ns)
            acc += batch_acc / n
    return acc

if data_name == 'genome':
    sum_accs = []
    for batch_words, batch_labels in get_batches(val_data['words'], val_data['labels'], batch_size):
        batch_words = np.array(batch_words, np.int64)
        acc = get_accs(params, batch_words, batch_labels, ns)
        sum_accs.append(acc)

else: # grouped into 2^n width trees for batching with different length sequences
    sum_accs = []
    for i, (words, labels) in enumerate(zip(val_data['words'], val_data['labels'])):
        if len(words):
            n_max = int(np.power(2, i+1))
            ns = tuple([int(N/(jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))])
            for batch_words, batch_labels in get_batches(words, labels, batch_size):
                batch_words = np.array(batch_words, np.int64)
                acc = get_accs(params, batch_words, batch_labels, ns)
                sum_accs.append(acc)

val_acc = np.sum(sum_accs) / np.sum([len(labels) for labels in val_data["labels"]])
val_acc = evaluate(val_data, n_val)
print("Initial acc  {:0.2f}  ".format(val_acc))
val_accs.append(val_acc)

for epoch in range(conf['n_epochs']):

    start_time = time.time()

    if data_name == 'genome':
        # calc cost and update params
        sum_loss = []
        for batch_words, batch_labels in tqdm(get_batches(train_data['words'], train_data['labels'], batch_size)):
            batch_words = np.array(batch_words, np.int64)
            loss, params, opt_state = train_step(params, opt_state, batch_words, batch_labels, ns)
            sum_loss.append(loss)
    
    else:
        # calc cost and update params
        sum_loss = []
        for i, (words, labels) in enumerate(zip(train_data['words'], train_data['labels'])):
            if len(words):
                n_max = int(np.power(2, i+1))
                N = n_max*n_qubits
                ns = tuple([int(N/(jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))])
                for batch_words, batch_labels in get_batches(words, labels, batch_size):
                    batch_words = np.array(batch_words, np.int64)
                    loss, params, opt_state = train_step(params, opt_state, batch_words, batch_labels, ns)
                    sum_loss.append(loss)

    if data_name == 'genome':
        sum_accs = []
        for batch_words, batch_labels in get_batches(val_data['words'], val_data['labels'], batch_size):
            batch_words = np.array(batch_words, np.int64)
            acc = get_accs(params, batch_words, batch_labels, ns)
            sum_accs.append(acc)

    else:
        sum_accs = []
        for i, (words, labels) in enumerate(zip(val_data['words'], val_data['labels'])):
            if len(words):
                n_max = int(np.power(2, i+1))
                N = n_max*n_qubits
                ns = tuple([int(N/(jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))])
                for batch_words, batch_labels in get_batches(words, labels, batch_size):
                    batch_words = np.array(batch_words, np.int64)
                    acc = get_accs(params, batch_words, batch_labels, ns)
                    sum_accs.append(acc)
        
    val_acc = np.sum(sum_accs) / np.sum([len(labels) for labels in val_data["labels"]])
    loss = np.sum(sum_loss)/ np.sum([len(labels) for labels in train_data["labels"]])
    epoch_time = time.time() - start_time
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Loss  {:0.2f}  ".format(loss))
    print("Acc  {:0.2f}  ".format(val_acc))
    val_accs.append(val_acc)
    losses.append(loss)
    all_params.append(params)

    if data_name == 'genome':
        sum_accs = []
        ns = tuple([int(N/(jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))])
        for batch_words, batch_labels in get_batches(test_data['words'], test_data['labels'], batch_size):
            batch_words = np.array(batch_words, np.int64)
            acc = get_accs(params, batch_words, batch_labels, ns)
            sum_accs.append(acc)

    else:
        sum_accs = []
        for i, (words, labels) in enumerate(zip(test_data['words'], test_data['labels'])):
            if len(words):
                n_max = int(np.power(2, i+1))
                N = n_max*n_qubits
                ns = tuple([int(N/(jnp.power(2, i))) for i in range(int(jnp.log2(n_max)))])
                for batch_words, batch_labels in get_batches(words, labels, batch_size):
                    batch_words = np.array(batch_words, np.int64)
                    acc = get_accs(params, batch_words, batch_labels, ns)
                    sum_accs.append(acc)

    test_acc = np.sum(sum_accs) / np.sum([len([l for l in labels]) for labels in test_data["labels"]])
    test_accs.append(test_acc)
    print("Test set accuracy: ", test_acc)


# ------------------------------- SAVE DATA ---------------–------------ #

save_dict = {
    'params_dict': all_params,
    'opt_state': opt_state,
    'test_accs': test_accs,
    'val_accs': val_accs,
    'losses': losses,
    'w2i': w2i
}

# ------------------------------ SAVE DATA -----------------–------------ #

save_path = f'../Results'
Path(save_path).mkdir(parents=True, exist_ok=True)
for key, value in conf.items():
    full_save_path = f'{save_path}{key}'
    pickle.dump(obj=value, file=open(f'{full_save_path}/{key}', 'wb'))
