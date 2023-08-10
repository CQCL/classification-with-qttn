from pathlib import Path
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Bra
from discopy.quantum.circuit import Measure, Discard
import tensornetwork as tn
from jax import vmap, value_and_grad
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

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)

# ------------------------------------- SETTINGS -------------------------------- #
with open('SCTN_config.yaml', 'r') as f:
    conf = yaml.safe_load(f)
# ------------------------------------- SETTINGS -------------------------------- #

n_qubits = conf['n_qubits'] # number of qubits per word

if conf['post_sel']:
    box_vec = make_state_vector
else:
    box_vec = make_density_matrix

# ------------------------------- READ IN DATA ---------------------------- #
load_path = f'../Data/{conf["data_name"]}/{conf["parse_type"]}/'

w2i = pickle.load(file=open(f'{load_path}w2i', 'rb'))
r2i = pickle.load(file=open(f'{load_path}r2i', 'rb'))
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

def SCTN(W_params, U_params, I_params, Us, Is, class_params):
    word_vecs = vmap(word_vec_init)(W_params)
    words = [box_vec(vec, n_qubits) for vec in word_vecs]
    circ = Id.tensor(*words)

    for idx, (U_inds, I_inds) in enumerate(zip(Us, Is)):  

        # apply unitary ops
        for u in U_inds:
            U_box = ansatz(2 * n_qubits, 2 * n_qubits, U_params[idx])
            circ = apply_box(circ, U_box, u * n_qubits)

        # apply discard op
        I_box = ansatz(2 * n_qubits, n_qubits, I_params[idx])
        circ = apply_box(circ, I_box, I_inds * n_qubits)

    # apply final classification ansatz
    circ >>= ansatz(n_qubits, n_qubits, class_params)

    # measure the middle qubit
    eff = Bra(0) if conf['post_sel'] else Discard()
    boxes = [Id(1) if i == n_qubits // 2 + 1 else eff for i in range(n_qubits)]
    circ >>= Id.tensor(*boxes)
    
    wire_state = box_vec(circ.eval(**eval_args).array, 1)
    wire_state >>= Measure()

    pred = wire_state.eval().array
    pred = jnp.array(pred, jnp.float64)

    return pred

vmap_mera = vmap(SCTN, (0, 0, 0, None, None, None))

def get_loss(params, batch_words, batch_rules, batch_Us, batch_Is, labels):
    preds = vmap_mera(params['words'][batch_words], params['Us'][batch_rules], params['Is'][batch_rules],  batch_Us, batch_Is, params['class'])

    if conf['post_sel']:
        preds = preds / jnp.sum(preds, axis=1)[:,None] # renormalise output

    if not conf['use_jit']:
        assert all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)
    
    out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, jnp.array(labels))])

    return -jnp.sum(out)

val_n_grad = value_and_grad(get_loss)

def train_step(params, opt_state, batch_words, batch_rules, batch_Us, batch_Is, batch_labels):
    
    cost, grads = val_n_grad(params, batch_words, batch_rules, batch_Us, batch_Is, batch_labels)

    if conf['use_grad_clip']: 
        for k in grads:
            grads[k] = jnp.clip(grads[k], -conf['grad_clip'], conf['grad_clip'])

    if conf['use_optax_reg']:
        updates, opt_state = optimizer.update(grads, opt_state, params)
    else:            
        updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return cost, params, opt_state

def get_accs(params, batch_words, batch_rules, batch_Us, batch_Is, labels):
    preds = vmap_mera(params['words'][batch_words], params['Us'][batch_rules], params['Is'][batch_rules],  batch_Us, batch_Is, params['class'])

    if conf['post_sel']:
        preds = preds / jnp.sum(preds, axis=1)[:,None]

    if not conf['use_jit']:
        assert all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

    acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2

    return acc

def random(size):
    return jnp.array(np.random.uniform(0, conf['init_val'], size=size))

def get_batches(words, rules, labels, batch_size):
    for s in gen_batches(len(labels), batch_size):
        word = np.array(words[s], np.int64)
        rule = np.array(rules[s], np.int64)
        yield word, rule, labels[s]

n_words = max(w2i.values())
n_rules = max(r2i.values())

print("Rule(s): ", n_rules+1)
print("Number of unique tokens: ", n_words+1)

word_emb_size = ansatz.n_params(n_qubits)
rule_emb_size = ansatz.n_params(2 * n_qubits)

if conf['use_optax_reg'] is True:
    optimizer = optax.adamw(conf['lr'])
else:
    optimizer = optax.adam(conf['lr'])

# initialise optax optimiser
word_params = random((n_words + 1, word_emb_size))
U_params = random((n_rules + 1, rule_emb_size))
I_params = random((n_rules + 1, rule_emb_size))
class_params = random(word_emb_size)
params = {'words': word_params, 'Us': U_params, 'Is': I_params, 'class': class_params}
opt_state = optimizer.init(params)

val_accs = []
test_accs = []
losses = []
all_params = []

def evaluate(data, n):
    acc = 0
    for d in tqdm(data):
        if len(d["labels"]) == 0:
            continue
        Us = d["I_offsets"][0]  # same for all in structural batch
        Is = d["U_offsets"][0]
        batches = get_batches(d["words"], d["rules"], d["labels"], conf['batch_size'])
        for batch_words, batch_rules, batch_labels in batches:
            batch_acc = get_accs(params, batch_words, batch_rules, Us, Is, batch_labels)
            acc += batch_acc / n
    return acc

val_acc = evaluate(val_data, n_val)
print("Initial acc  {:0.2f}  ".format(val_acc))
val_accs.append(val_acc)

for epoch in range(conf['n_epochs']):
    start_time = time.time()
    # calc cost and update params
    loss = 0
    for data in tqdm(train_data):
        if len(data["labels"]) == 0:
            continue
        Us = data["I_offsets"][0] # same for all in structural batch 
        Is = data["U_offsets"][0]
        batches = get_batches(data["words"], data["rules"], data["labels"], conf['batch_size'])
        for batch_words, batch_rules, batch_labels in batches:
            cost, params, opt_state = train_step(params, opt_state, batch_words, batch_rules, Us, Is, batch_labels)
            loss += cost / n_train

    losses.append(loss)
    all_params.append(params)

    print("Loss  {:0.2f}  ".format(loss))
    val_acc = evaluate(val_data, n_val)
    print("Val Acc  {:0.2f}  ".format(val_acc))
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
    save_path = f'Results'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for key, value in conf.items():
        full_save_path = f'{save_path}{key}'
        pickle.dump(obj=value, file=open(f'{full_save_path}/{key}', 'wb'))
