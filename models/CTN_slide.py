from lzma import FILTER_DELTA
from pathlib import Path
from random import shuffle
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Ket, H, Bra
from discopy.quantum.circuit import Box, Measure, Discard
import tensornetwork as tn
from jax import vmap, value_and_grad, jit, vmap
from sklearn.utils import gen_batches
from tqdm import tqdm
import time 
import pickle
from matplotlib.pyplot import axes
import numpy as np
import pickle
import optax
from functools import partial

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)

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
no_epochs = 10
batch_size = 64
init_val = 0.001
ansatz = 'A14' 
# ------------------------------------- SETTINGS -------------------------------- #

for parse_type in ['unibox', 'height']:

    # ------------------------------- READ IN DATA ---------------------------- #
    print("Reading in data ... ")
    print("Load data")

    save_path = f'Data/CTN_SLIDE_{window_size}/{data_name}/'

    w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
    train_data = pickle.load(file=open(f'{save_path}{"train_data"}', 'rb'))
    val_data = pickle.load(file=open(f'{save_path}{"val_data"}', 'rb'))
    test_data = pickle.load(file=open(f'{save_path}{"test_data"}', 'rb'))

    print("Number of train examples: ", len(train_data["labels"]))
    print("Number of val examples: ", len(val_data["labels"]))
    print("Number of test examples: ", len(test_data["labels"]))
    # ------------------------------- READ IN DATA ----------------------------- #
    
    for n_word_qubits in [1]: 

        for n_layers in [1,2]:

            for lr in [0.01,0.001,0.0001]:

                n_rule_qubits = 2 * n_word_qubits  # number of qubits in each sub-PQC
                if not use_jit: 
                    jit = lambda x: x
                eval_args = {}
                mixed = not post_sel
                eval_args['mixed'] = mixed  # use density matrices if discarding 
                eval_args['contractor'] = tn.contractors.auto  # use tensor networks if speed

                print("Model: ", model)
                print("window size: ", window_size)
                print("Parse: ", parse_type)
                print("Number of epochs: ", no_epochs)
                print("Batch size: ", batch_size)
                print("Ansatz: ", ansatz)
                print("Number of word qubits: ", n_word_qubits)
                print("Number of layers: ", n_layers)
                print("Using post selection: ", post_sel)
                print("Include classification box: ", include_final_classification_box)
                print("Using gradient clipping: ", use_grad_clip)

                def flatten_list(li):
                    return [y for x in li for y in x]

                def IQP(n_qubits, n_layers, params):

                    circ = Id(n_qubits)
                    if n_qubits == 1:
                        assert len(params) == 3
                        circ = circ.Rx(params[0], 0)
                        circ = circ.Rz(params[1], 0)
                        circ = circ.Rx(params[2], 0)
                    else:
                        
                        assert len(params) == (n_qubits-1) * n_layers
                        lay_params = (n_qubits-1) 
                        
                        for n in range(n_layers):
                            hadamards = Id(0).tensor(*(n_qubits * [H]))
                            circ = circ >> hadamards
                            for i in range(n_qubits-1):
                                tgt = i
                                src = i+1
                                circ = circ.CRz(params[i+(n*lay_params)], src, tgt)
                    return circ

                def ansatz14(n_qubits, n_layers, params):
                    
                    circ = Id(n_qubits)

                    if n_qubits == 1:
                        assert len(params) == 3
                        circ = circ.Rx(params[0], 0)
                        circ = circ.Rz(params[1], 0)
                        circ = circ.Rx(params[2], 0)
                    else:
                        
                        assert len(params) == 4 * n_qubits * n_layers
                        lay_params = 4 * n_qubits
                        
                        for n in range(n_layers):
                            # single qubit rotation wall 1
                            for i in range(n_qubits):
                                param = params[i + (n * lay_params)]
                                circ = circ.Ry(param, i)

                            # entangling ladder 1
                            for i in range(n_qubits):
                                src = (n_qubits - 1 + i) % n_qubits
                                tgt = (n_qubits - 1 + i + 1) % n_qubits
                                param = params[i + n_qubits + (n * lay_params)]
                                circ = circ.CRx(param, src, tgt)

                            # single qubit rotation wall 2
                            for i in range(n_qubits):
                                param = params[i  + 2 * n_qubits+(n * lay_params)]
                                circ = circ.Ry(param, i)

                            # entangling ladder 2
                            for i in range(n_qubits):
                                src = (n_qubits - 1 + i) % n_qubits
                                tgt = (n_qubits - 1 + i - 1) % n_qubits
                                param = params[i + 3  * n_qubits + (n * lay_params)]
                                circ = circ.CRx(param, src, tgt)
                    return circ

                def ansatz9(n_qubits, n_layers, params):

                    circ = Id(n_qubits)

                    if n_qubits == 1:
                        assert len(params) == 3
                        circ = circ.Rx(params[0], 0)
                        circ = circ.Rz(params[1], 0)
                        circ = circ.Rx(params[2], 0)
                    else:
                        assert len(params) == n_qubits * n_layers
                        
                        lay_params = n_qubits
                        
                        for n in range(n_layers):
                            
                            hadamards = Id().tensor(*(n_qubits * [H]))
                            circ = circ >> hadamards

                            for i in range(n_qubits - 1):
                                circ = circ.CZ(i, i + 1)

                            for i in range(n_qubits):
                                param = params[i+(n * lay_params)]
                                circ = circ.Rx(param, i)

                    return circ

                def Ansatz(n_qubits, n_layers, params, type):
                    if type == 'IQP':
                        return IQP(n_qubits, n_layers, params)
                    elif type == 'A14':
                        return ansatz14(n_qubits, n_layers, params)
                    elif type == 'A9':
                        return ansatz9(n_qubits, n_layers, params)

                def word_vec_initilisation(word_params): # initialise word states according to params
                    circ = Ket(*[0] * n_word_qubits)
                    circ >>= Ansatz(n_word_qubits, n_layers, word_params, ansatz)
                    return circ.eval(**eval_args).array

                single_batch_vec_init = jit(vmap(word_vec_initilisation)) # vmap over initial states in a tree

                def make_words(data, name): # return discopy box from post discard density matrix
                    from discopy.quantum.circuit import qubit as d_qb
                    dom, cod = d_qb ** 0, d_qb ** n_word_qubits
                    box = Box(name, dom, cod, is_mixed=mixed, data=data)
                    box.array = box.data
                    return box

                def make_state(data, name):
                    from discopy.quantum.circuit import qubit as d_qb
                    dom, cod = d_qb ** (2 * n_word_qubits), d_qb ** (2 * n_word_qubits)
                    box = Box(name, dom, cod, is_mixed=True, data=data)
                    box.array = box.data
                    return box

                def uCTN(W_params, U_params, I_params, class_params, ns):

                    words = [make_words(vec, "w") for vec in single_batch_vec_init(W_params)]
                    circ = Id().tensor(*words)

                    for n in ns:

                        # apply unitary ops
                        for i in range(n_word_qubits, n-n_word_qubits)[::2*n_word_qubits]:
                            circ >>= Id(i)@Ansatz(n_rule_qubits, n_layers, U_params, ansatz)@Id(n-i-2*n_word_qubits)

                        # apply isometry
                        for i in range(n-n_word_qubits)[::2*n_word_qubits]:
                            circ >>= Id(i)@Ansatz(n_rule_qubits, n_layers, I_params, ansatz)@Id(n-i-2*n_word_qubits)

                        if post_sel:
                            circ >>= Id().tensor(*[Id(1) @ Bra(0)] * int(n/2))
                        else: 
                            circ >>= Id().tensor(*[Id(1) @ Discard()] * int(n/2))

                    # apply final classification ansatz
                    if include_final_classification_box:
                        circ >>= Ansatz(n_word_qubits, n_layers, class_params, ansatz)

                    # measure the middle qubit
                    left = n_word_qubits // 2
                    right = n_word_qubits - left - 1

                    # circ >>= Discard(left) @ Measure() @ Discard(right)
                    if not post_sel:
                        circ >>= Discard(left) @ Measure() @ Discard(right)
                    
                    pred = circ.eval(**eval_args).array

                    if post_sel:
                        pred = jnp.square(jnp.abs(pred+1e-7))
                        axes = list(range(n_word_qubits))
                        axes.pop(left)
                        pred = jnp.sum(pred, axis=tuple(axes))

                    else:
                        pred = jnp.array(pred, jnp.float64)

                    return pred

                def hCTN(W_params, U_params, I_params, class_params, ns):

                    words = [make_words(vec, "w") for vec in single_batch_vec_init(W_params)]
                    circ = Id().tensor(*words)

                    for idx, n in enumerate(ns):  

                        # apply unitary ops
                        for i in range(n_word_qubits, n-n_word_qubits)[::2*n_word_qubits]:
                            circ >>= Id(i)@Ansatz(n_rule_qubits, n_layers, U_params[idx], ansatz)@Id(n-i-2*n_word_qubits)

                        # apply isometry
                        for i in range(n-n_word_qubits)[::2*n_word_qubits]:
                            circ >>= Id(i)@Ansatz(n_rule_qubits, n_layers, I_params[idx], ansatz)@Id(n-i-2*n_word_qubits)
                        if post_sel:
                            circ >>= Id().tensor(*[Id(1) @ Bra(0)] * int(n/2))
                        else: 
                            circ >>= Id().tensor(*[Id(1) @ Discard()] * int(n/2))

                    # apply final classification ansatz
                    if include_final_classification_box:
                        circ >>= Ansatz(n_word_qubits, n_layers, class_params, ansatz)

                    # measure the middle qubit
                    left = n_word_qubits // 2
                    right = n_word_qubits - left - 1

                    # circ >>= Discard(left) @ Measure() @ Discard(right)
                    if not post_sel:
                        circ >>= Discard(left) @ Measure() @ Discard(right)
                    
                    pred = circ.eval(**eval_args).array

                    if post_sel:
                        pred = jnp.square(jnp.abs(pred+1e-7))
                        axes = list(range(n_word_qubits))
                        axes.pop(left)
                        pred = jnp.sum(pred, axis=tuple(axes))

                    else:
                        pred = jnp.array(pred, jnp.float64)

                    return pred

                if parse_type == 'unibox':
                    apply_batch_contractions = vmap(uCTN, (0, None, None, None, None))
                elif parse_type == 'height':
                    apply_batch_contractions = vmap(hCTN, (0, None, None, None, None))

                def loss(params, batch_words, batch_inds, labels, ns):

                    preds = get_preds(params, batch_words, ns)
                    
                    preds = jnp.array([jnp.average(preds[batch_inds[i]:batch_inds[i+1]], axis=0) for i in range(len(batch_inds)-1)])

                    out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, jnp.array(labels))])

                    return  -jnp.sum(out)

                if use_jit:
                    val_n_grad = partial(jit, static_argnums=(2, 4))(value_and_grad(loss))
                else:
                    val_n_grad = value_and_grad(loss)

                def train_step(params, opt_state, batch_words, batch_labels, batch_inds, ns):

                    cost, grads = val_n_grad(params, jnp.array(batch_words, jnp.int64), batch_inds, jnp.array(batch_labels), ns)

                    if use_grad_clip: 
                        grads["words"] = jnp.clip(grads["words"], -grad_clip, grad_clip)
                        grads["Us"] = jnp.clip(grads["Us"], -grad_clip, grad_clip)
                        grads["Is"] = jnp.clip(grads["Is"], -grad_clip, grad_clip)
                        grads["class"] = jnp.clip(grads["class"], -grad_clip, grad_clip)
                        
                    # update relevant params
                    if use_optax_reg is True:
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)

                    else:            
                        updates, opt_state = optimizer.update(grads, opt_state)
                        params = optax.apply_updates(params, updates)

                    return cost, params, opt_state, grads

                def get_preds(params, batch_words, ns):
                    
                    preds = apply_batch_contractions(params['words'][jnp.array(batch_words)], params['Us'], params["Is"], params['class'], ns)

                    if post_sel:
                        preds = preds / jnp.sum(preds, axis=1)[:,None]

                    assert use_jit or all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

                    return preds
                    
                def get_accs(params, batch_words, ns, counts, labels):
                
                    preds = get_preds(params, batch_words, ns)
                    preds = jnp.array([jnp.average(preds[counts[i]:counts[i+1]], axis=0) for i in range(len(counts)-1)])
                    acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2

                    return acc

                def initialise_params_h(no_words, no_rules, word_emb_size, rule_embed_size): # initialise word, rule and classification params
                    word_params = jnp.array(np.random.uniform(0, init_val, size=(no_words+1, word_emb_size)))
                    U_params = jnp.array(np.random.uniform(0, init_val, size=(no_rules, rule_embed_size)))
                    I_params = jnp.array(np.random.uniform(0, init_val, size=(no_rules, rule_embed_size)))
                    class_params = jnp.array(np.random.uniform(0, init_val, size=(word_emb_size)))
                    return word_params, U_params, I_params, class_params

                def initialise_params(no_words, word_emb_size, rule_embed_size): # initialise word, rule and classification params
                    word_params = jnp.array(np.random.uniform(0, init_val, size=(no_words+1, word_emb_size)))
                    U_params = jnp.array(np.random.uniform(0, init_val, size=rule_embed_size))
                    I_params = jnp.array(np.random.uniform(0, init_val, size=rule_embed_size))
                    class_params = jnp.array(np.random.uniform(0, init_val, size=(word_emb_size)))
                    return word_params, U_params, I_params, class_params

                def get_batches(words, counts, labels, batch_size):
                    slices = list(gen_batches(len(words), batch_size))
                    batched_w = [words[s] for s in slices]
                    batched_c = [counts[s] for s in slices]
                    batched_l = [labels[s] for s in slices]
                    return zip(batched_w,batched_c,batched_l)

                no_words = max(w2i.values())
                print("Number of unique tokens: ", no_words+1)

                if ansatz == 'A14':
                    if n_word_qubits == 1:
                        word_emb_size = 3
                    else:
                        word_emb_size = n_word_qubits * 4 * n_layers
                    rule_embed_size = n_rule_qubits * 4 * n_layers
                elif ansatz == 'A9':
                    if n_word_qubits == 1:
                        word_emb_size = 3
                    else:
                        word_emb_size = n_word_qubits * n_layers
                    rule_embed_size = n_rule_qubits * n_layers
                elif ansatz == 'IQP':
                    if n_word_qubits == 1:
                        word_emb_size = 3
                    else:
                        word_emb_size = (n_word_qubits-1) * n_layers
                    rule_embed_size = (n_rule_qubits-1) * n_layers

                if use_optax_reg is True:
                    optim = 'adamW'
                    optimizer = optax.adamw(lr)
                else:
                    optim = 'adam'
                    optimizer = optax.adam(lr)

                print("Optimizer: ", optim)
                print("lr: ", lr)

                no_rules = int(np.log2(window_size)+1)

                # initialise optax optimiser
                if parse_type == 'height':
                    word_params, U_params, I_params, class_params = initialise_params_h(no_words+1, no_rules, word_emb_size, rule_embed_size)
                elif parse_type == 'unibox':
                    word_params, U_params, I_params, class_params = initialise_params(no_words+1, word_emb_size, rule_embed_size)

                params = {'words': word_params,'Us': U_params, 'Is': I_params, 'class': class_params}

                N = window_size*n_word_qubits
                ns = tuple([int(N/(jnp.power(2, i))) for i in range(int(jnp.log2(window_size)))])
 
                val_accuracies = []
                test_accuracies = []
                losses = []
                all_params = []
                all_opt_states = []
                all_grads = []

                sum_accs = []
                for batch_words, batch_counts, batch_labels in tqdm(get_batches(val_data['words'], val_data['counts'], val_data['labels'], batch_size)): 
                    batch_counts = np.concatenate(([0], batch_counts))
                    cum_inds = tuple(np.array([np.cumsum(batch_counts[:i+1])[-1] for i in range(len(batch_counts))], np.int64)) 
                    try:
                        batch_words = jnp.array(flatten_list(batch_words), jnp.int64)
                        acc = get_accs(params, batch_words, ns, cum_inds, batch_labels)
                    except:
                        batch_words = jnp.array(batch_words, jnp.int64)
                        acc = get_accs(params, batch_words, ns, cum_inds, batch_labels)
                    sum_accs.append(acc)

                val_acc = np.sum(sum_accs) / len(val_data['labels'])
                print("Initial acc  {:0.2f}  ".format(val_acc))
                val_accuracies.append(val_acc)

                for epoch in range(no_epochs):

                    start_time = time.time()

                    sum_loss = []
                    for batch_words, batch_counts, batch_labels in tqdm(get_batches(train_data['words'], train_data['counts'], train_data['labels'], batch_size)): # this was 16 untul 30th for sm reason!!!
                        batch_counts.insert(0,0)
                        cum_inds = tuple(np.array([np.cumsum(batch_counts[:i+1])[-1] for i in range(len(batch_counts))], np.int64)) 
                        try:
                            batch_words = jnp.array(flatten_list(batch_words), jnp.int64)
                            cost, params, opt_state, grads = train_step(params, opt_state, batch_words, batch_labels, cum_inds, ns)
                        except:
                            batch_words = jnp.array(batch_words, jnp.int64)
                            cost, params, opt_state, grads = train_step(params, opt_state, batch_words, batch_labels, cum_inds, ns)
                        sum_loss.append(cost)

                    cost = np.sum(sum_loss) / len(train_data['labels'])
                    losses.append(cost)
                    all_params.append(params)

                    sum_accs = []
                    for batch_words, batch_counts, batch_labels in tqdm(get_batches(val_data['words'], val_data['counts'], val_data['labels'], batch_size)): 
                        batch_counts = np.concatenate(([0], batch_counts))
                        cum_inds = tuple(np.array([np.cumsum(batch_counts[:i+1])[-1] for i in range(len(batch_counts))], np.int64)) 
                        try:
                            batch_words = jnp.array(flatten_list(batch_words), jnp.int64)
                            acc = get_accs(params, batch_words, ns, cum_inds, batch_labels)
                        except:
                            batch_words = jnp.array(batch_words, jnp.int64)
                            acc = get_accs(params, batch_words, ns, cum_inds, batch_labels)
                        sum_accs.append(acc)

                    val_acc = np.sum(sum_accs) / len(val_data['labels'])
                    epoch_time = time.time() - start_time
                    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                    print("Loss  {:0.2f}  ".format(cost))
                    print("Acc  {:0.2f}  ".format(val_acc))
                    val_accuracies.append(val_acc)
                    
                    sum_accs = []
                    for batch_words, batch_counts, batch_labels in tqdm(get_batches(test_data['words'], test_data['counts'], test_data['labels'], batch_size)): 
                        batch_counts = np.concatenate(([0], batch_counts))
                        cum_inds = tuple(np.array([np.cumsum(batch_counts[:i+1])[-1] for i in range(len(batch_counts))], np.int64)) 
                        try:
                            batch_words = jnp.array(flatten_list(batch_words), jnp.int64)
                            acc = get_accs(params, batch_words, ns, cum_inds, batch_labels)
                        except:
                            batch_words = jnp.array(batch_words, jnp.int64)
                            acc = get_accs(params, batch_words, ns, cum_inds, batch_labels)
                        sum_accs.append(acc)

                    test_acc = np.sum(sum_accs) / len(test_data['labels'])
                    print("Test acc  {:0.2f}  ".format(test_acc))
                    test_accuracies.append(test_acc)
                
                    # ------------------------------ SAVE DATA -----------------–------------ #    
                    if post_sel:
                        save_path = f'Results/{data_name}/{model}_SLIDE_{window_size}/{parse_type}/post_sel/{n_word_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                    else:
                        save_path = f'Results/{data_name}/{model}_SLIDE_{window_size}/{parse_type}/discards/{n_word_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'

                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    pickle.dump(obj=all_params, file=open(f'{save_path}{"param_dict"}', 'wb'))
                    pickle.dump(obj=opt_state, file=open(f'{save_path}{"final_opt_state"}', 'wb'))
                    pickle.dump(obj=test_accuracies, file=open(f'{save_path}{"test_accs"}', 'wb'))
                    pickle.dump(obj=val_accuracies, file=open(f'{save_path}{"val_accs"}', 'wb'))
                    pickle.dump(obj=losses, file=open(f'{save_path}{"loss"}', 'wb'))
                    pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
                print(save_path)
                    # ------------------------------- SAVE DATA ---------------–------------ #


