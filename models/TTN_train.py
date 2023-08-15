from pathlib import Path
from re import T
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Ket, Bra
from discopy.quantum.circuit import Box, Measure, Discard
import tensornetwork as tn
import jax
from jax import vmap, value_and_grad, jit, vmap
from sklearn.utils import gen_batches
from tqdm import tqdm
import time 
import pickle
import numpy as np
import pickle
import optax
import yaml

from ansatz import apply_box, make_density_matrix, make_state_vector
from ansatz import IQPAnsatz, Ansatz9, Ansatz14

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)
seed = 0
key = jax.random.PRNGKey(seed)

with open('TTN_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)


# ------------------------------- READ IN DATA ---------------------------- #
print("Reading in data ... ")

load_path = f'../Data/{conf["model"]}/{conf["data_name"]}/{conf["parse_type"]}/'

w2i = pickle.load(file=open(f'{load_path}{"w2i"}', 'rb'))
r2i = pickle.load(file=open(f'{load_path}{"r2i"}', 'rb'))
train_data = pickle.load(file=open(f'{load_path}{"train_data"}', 'rb'))
val_data = pickle.load(file=open(f'{load_path}{"val_data"}', 'rb'))
test_data = pickle.load(file=open(f'{load_path}{"test_data"}', 'rb'))

print("Number of train examples: ", len(train_data["labels"]))
print("Number of val examples: ", len(val_data["labels"]))
print("Number of test examples: ", len(test_data["labels"]))
# ------------------------------- READ IN DATA ----------------------------- #

n_qubits = conf['n_qubits'] # number of qubits per word
init_val = conf['init_val']
batch_size = conf['batch_size']
            
if not conf['use_jit']: 
    jit = lambda x: x

discard = not conf['post_sel']

eval_args = {
    'mixed': discard, # use density matrices if discarding 
    'contractor': tn.contractors.auto # use tensor networks if speed
}

if conf['ansatz'] == 'IQP':
    ansatz = IQPAnsatz(conf['n_layers'], discard=discard)
elif conf['ansatz'] == 'A9':
    ansatz = Ansatz9(conf['n_layers'], discard=discard)
elif conf['ansatz'] == 'A14':
    ansatz = Ansatz14(conf['n_layers'], discard=discard)
else:
    raise ValueError

if conf['post_sel']:
    box_vec = make_state_vector
else:
    box_vec = make_density_matrix

def word_vec_init(word_params):
    """ initialise word states according to params """
    circ = ansatz(dom=0, cod=n_qubits, params=word_params)
    return circ.eval(**eval_args).array

def PQC(rule_params, vec1, vec2):
    """ apply rule with two density matrices as input """
    circ = box_vec(vec1, n_qubits) @ box_vec(vec2, n_qubits)
    circ >>= ansatz(dom=2 * n_qubits, cod=n_qubits, params=rule_params)
    out = circ.eval(**eval_args).array
    return out

def measure(out_vec, class_params):
    """ apply classification box and measure classification qubit """
    circ = box_vec(out_vec, n_qubits)

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

vmap_word_vec_init = jit(vmap(word_vec_init)) # vmap over initial states in a tree

def merge(vecs, rule_offset): # update correct tree state according to rule offsets
    rule = rule_offset[:-2]
    idx1, idx2 = jnp.array(rule_offset[-2:], dtype=jnp.int32)

    input_vec1 = vecs[idx1]
    input_vec2 = vecs[idx2]
    new_vec = PQC(jnp.array(rule), input_vec1, input_vec2)
    out = vecs.at[idx1].set(new_vec)

    return out, out

def contract(word_params, rules, offsets, class_params): # scan contractions down the tree
    input_vecs = vmap_word_vec_init(word_params)
    rules_offs = jnp.concatenate((rules, offsets), axis=1)
    input_vecs, _ = jax.lax.scan(merge, init=input_vecs, xs=rules_offs)
    preds = measure(input_vecs[0], class_params)
    return preds

vmap_contract = jit(vmap(contract, (0,0,0,None))) # vmap over all trees in batch

def get_loss(params, batch_words, batch_rules, batch_offsets, labels):
    b_params = params['words'][batch_words]
    b_rules = params['rules'][batch_rules]
    preds = vmap_contract(b_params, b_rules, batch_offsets, params['class'])

    if conf['post_sel']:
        preds = preds / jnp.sum(preds, axis=1)[:,None]

    if not conf['use_jit']:
        assert all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

    out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, labels)])

    return -jnp.sum(out)

val_n_grad = jit(value_and_grad(get_loss))

def train_step(params, opt_state, batch_words, batch_rules, batch_offsets, batch_labels):
    cost, grads = val_n_grad(params, batch_words, batch_rules, batch_offsets, batch_labels)

    if conf['use_grad_clip']: 
        for k in grads:
            grads[k] = jnp.clip(grads[k], -conf['grad_clip'], conf['grad_clip'])

    if conf['use_optax_reg']:
        updates, opt_state = optimizer.update(grads, opt_state, params)
    else:            
        updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return cost, params, opt_state

def get_accs(params, batch_words, batch_rules, batch_offsets, labels):
    
    preds = vmap_contract(params['words'][batch_words], params['rules'][batch_rules], batch_offsets, params['class'])
    
    if conf['post_sel']:
        preds = preds / jnp.sum(preds, axis=1)[:,None] # renormalise output

    if not conf['use_jit']:
        assert all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

    acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2 

    return acc

def random(size):
    return jnp.array(np.random.uniform(0, conf['init_val'], size=size))

def get_batches(words, rules, offsets, labels, batch_size):
    for s in gen_batches(len(labels), batch_size):
        word = [jnp.array(s) for s in words[s]]
        rule = [jnp.array(r) for r in rules[s]]
        offset = [jnp.array(o) for o in offsets[s]]
        label = jnp.array(labels[s])

        yield word, rule, offset, label

def pad_right(v, n, pad_value):
    """ pad vector v on the right with n values of pad_value """
    # return jnp.pad(v, (0, n), constant_values=pad_value)
    if n == 0:
        return v
    if len(v) == 0:
        return np.array([pad_value] * n)
    return np.concatenate((v, [pad_value] * n))

def pad_trees(batch_words, batch_rules, batch_offsets, max_words, words_pad_idx, rules_pad_idx):
    pad_words = []
    pad_rules = []
    pad_offsets = []

    # pad offsets with unused indices: void[-2] = merge(void[-2], void[-1])
    void_idx = [max_words - 2, max_words - 1]
    for words, rules, offsets in zip(batch_words, batch_rules, batch_offsets):
        pad_len = max_words-len(words)
        pad_words.append(pad_right(words, pad_len, words_pad_idx))
        pad_rules.append(pad_right(rules, pad_len, rules_pad_idx))
        pad_offsets.append(pad_right(offsets, pad_len, void_idx))
    return jnp.array(pad_words), jnp.array(pad_rules), jnp.array(pad_offsets)

n_words = max(w2i.values())
n_rules = max(r2i.values())

print("Rule(s): ", list(r2i.keys()))
print("Number of unique tokens: ", n_words+1)
print("Number of unique rules: ", n_rules+1)

word_emb_size = ansatz.n_params(n_qubits)
rule_emb_size = ansatz.n_params(2 * n_qubits)

# pad to max tree width and batch size
words_pad_idx = n_words+1
rules_pad_idx = n_rules+1

max_words = max([len(words) for data in [train_data, val_data, test_data]
                            for words in data['words']])

print("Max tree width: ", max_words)

if conf['use_optax_reg'] is True:
    optimizer = optax.adamw(conf['lr'])
else:
    optimizer = optax.adam(conf['lr'])

# initialise optax opxtimiser
word_params = random((n_words + 1, word_emb_size))
rule_params = random((n_rules + 1, rule_emb_size))
class_params = random(word_emb_size)
params = {'words': word_params, 'rules': rule_params, 'class': class_params}
opt_state = optimizer.init(params)

val_accs = []
test_accs = []
losses = []
all_params = []

def evaluate(data, n):
    acc = 0
    batches = get_batches(data["words"], data["rules"], data["offsets"], data["labels"], conf['batch_size'])
    for b_words, b_rules, b_offsets, b_labels in tqdm(batches):
        pad_words, pad_rules, pad_offsets = pad_trees(b_words, b_rules, b_offsets, max_words, words_pad_idx, rules_pad_idx)
        batch_acc = get_accs(params, pad_words, pad_rules, pad_offsets, b_labels)
        acc += batch_acc
    return acc / n

# initial acc
val_acc = evaluate(val_data, len(val_data["labels"]))
print("Initial acc  {:0.2f}  ".format(val_acc))
val_accs.append(val_acc)

for epoch in range(conf['n_epochs']):
    start_time = time.time()

    loss = 0
    batches = get_batches(train_data["words"], train_data["rules"], train_data["offsets"], train_data["labels"], conf['batch_size'])
    for b_words, b_rules, b_offsets, b_labels in tqdm(batches):
        pad_words, pad_rules, pad_offsets = pad_trees(b_words, b_rules, b_offsets, max_words, words_pad_idx, rules_pad_idx)
        cost, params, opt_state = train_step(params, opt_state, pad_words, pad_rules, pad_offsets, b_labels)
        loss += cost / len(train_data['labels'])

    losses.append(loss)
    all_params.append(params)

    print("Loss  {:0.2f}".format(loss))
    val_acc = evaluate(val_data, len(val_data['labels']))
    print("Val Acc  {:0.2f}".format(val_acc))
    val_accs.append(val_acc)

    test_acc = evaluate(test_data, len(test_data['labels']))
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
    save_path = f'../Results/{conf["model"]}/{conf["data_name"]}/{conf["parse_type"]}/{timestr}/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for key, value in save_dict.items():
        pickle.dump(obj=value, file=open(f'{save_path}{key}', 'wb'))
    # ------------------------------- SAVE DATA ---------------–------------ #
