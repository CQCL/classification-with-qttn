from pathlib import Path
from re import T
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Ket, H, Bra
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

from .ansatz import IQPAnsatz, Ansatz9, Ansatz14

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)
seed = 0
key = jax.random.PRNGKey(seed)

# ------------------------------------- SETTINGS -------------------------------- #
model = 'PTN'
if model in ['PTN', 'TTN']:
    syntactic = False
elif model == 'STTN':
    syntactic = True
assert model in ['PTN', 'TTN', 'STTN']

data_name = 'RT'
reduce_train = False
cut = False

if data_name == 'clickbait':
    reduce_val = 2031
else:
    reduce_val = 2352
thr = 16
if thr == 64:
    save_name = f'{data_name}_full'
else:
    save_name = data_name
save_name = data_name
number_of_structures = 100

if data_name == 'genome':
    batch_size = 16
    assert syntactic == False, "No syntax defined for genome data, please chose PTN or TTN model}."
else:
    batch_size = 64

post_sel = True
use_jit = True
use_grad_clip = True
include_final_classification_box = True
use_dropout = False
use_param_reg = False 
use_optax_reg = True
grad_clip = 100.0
no_epochs = 15
init_val = 0.01
ansatz = 'A14'
# ------------------------------------- SETTINGS -------------------------------- #

if model == 'TTN':
    parse_types = ['unibox', 'rule', 'height']
elif model in ['PTN', 'TTN']:
    parse_types = ['unibox', 'height']

for parse_type in ['unibox', 'height']: 

    print("Data: ", data_name)
    print("Model: ", model)
    print("Parse type: ", parse_type)
    print("Post sel: ", post_sel)                

    # ------------------------------- READ IN DATA ---------------------------- #
    print("Reading in data ... ")

    if data_name == 'genome':
        save_path = f'Data/{model}/{data_name}/{parse_type}/'

    else:
        if cut:
            if reduce_train:
                save_path = f'Data/{model}/{save_name}/{thr}_{number_of_structures}_REDUCED_{reduce_val}/{parse_type}/'
            else:
                save_path = f'Data/{model}/{save_name}/{thr}_{number_of_structures}/{parse_type}/'
        else:
            if reduce_train:
                save_path = f'Data/{model}/{save_name}/REDUCED_{reduce_val}/{parse_type}/'
            else:
                save_path = f'Data/{model}/{save_name}/{parse_type}/'
    
    w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
    r2i = pickle.load(file=open(f'{save_path}{"r2i"}', 'rb'))
    train_data = pickle.load(file=open(f'{save_path}{"train_data"}', 'rb'))
    val_data = pickle.load(file=open(f'{save_path}{"val_data"}', 'rb'))
    test_data = pickle.load(file=open(f'{save_path}{"test_data"}', 'rb'))

    N = int(np.ceil(len(train_data['words'])/4))
    train_data['words'] = train_data['words'][:N]
    train_data['rules'] = train_data['rules'][:N]
    train_data['offsets'] = train_data['offsets'][:N]
    train_data['labels'] = train_data['labels'][:N]

    print("Number of train examples: ", len(train_data["labels"]))
    print("Number of val examples: ", len(val_data["labels"]))
    print("Number of test examples: ", len(test_data["labels"]))
    # ------------------------------- READ IN DATA ----------------------------- #
    
    for n_qubits in [1,2]:
                
        for n_layers in [1,2]:

            for lr in [0.01, 0.001, 0.0001]:

                
                if not use_jit: 
                    jit = lambda x: x
                eval_args = {}
                mixed = not post_sel
                eval_args['mixed'] = mixed  # use density matrices if discarding 
                eval_args['contractor'] = tn.contractors.auto  # use tensor networks if speed

                print("Use syntax: ", syntactic)
                print("Number of epochs: ", no_epochs)
                print("Batch size: ", batch_size)
                print("Ansatz: ", ansatz)
                print("Number of word qubits: ", n_qubits)
                print("Number of layers: ", n_layers)
                print("Using post selection: ", post_sel)
                print("Init value: ", init_val)
                print("Using gradient clipping: ", use_grad_clip)

                def Ansatz(n_qubits, n_layers, params, type):
                    if type == 'IQP':
                        return IQP(n_qubits, n_layers, params)
                    elif type == 'A14':
                        return ansatz14(n_qubits, n_layers, params)
                    elif type == 'A9':
                        return ansatz9(n_qubits, n_layers, params)

                def make_vec(n_qubits, data): # return discopy box from post discard density matrix
                    from discopy.quantum.circuit import qubit as d_qb
                    dom, cod = d_qb ** 0, d_qb ** n_qubits
                    box = Box('some name', dom, cod, is_mixed=mixed, data=data)
                    box.array = box.data
                    return box

                def word_vec_initilisation(word_params): # initialise word states according to params
                    circ = Ket(*[0] * n_qubits)
                    circ >>= Ansatz(n_qubits, n_layers, word_params, ansatz)
                    out = circ.eval(**eval_args).array
                    return out

                def PQC(rule_params, vec1, vec2): # apply rule with two density matrices as input 
                    
                    circ = make_vec(n_qubits, vec1) @ make_vec(n_qubits, vec2)

                    circ >>= Ansatz(2 * n_qubits, n_layers, rule_params, ansatz)

                    if post_sel:
                        circ >>= Id.tensor(*[Id(1) @ Bra(0)] * n_qubits)
                    else: 
                        circ >>= Id.tensor(*[Id(1) @ Discard()] * n_qubits)

                    # contract to get output tensor
                    out = circ.eval(**eval_args).array

                    return out

                def measure(out_vec, class_params): # apply classification box and measure classification qubit
                    
                    output = make_vec(n_qubits, out_vec)

                    # apply final classification ansatz
                    if include_final_classification_box:
                        output >>= Ansatz(n_qubits, n_layers, class_params, ansatz)

                    # measure the middle qubit
                    left = n_qubits // 2
                    right = n_qubits - left - 1
                    
                    if not post_sel:
                        output >>= Discard(left) @ Measure() @ Discard(right)
                    
                    pred = output.eval(**eval_args).array

                    if post_sel:
                        pred = jnp.square(jnp.abs(pred+1e-7))
                        axes = list(range(n_qubits))
                        axes.pop(left)
                        pred = jnp.sum(pred, axis=tuple(axes))

                    else:
                        pred = jnp.array(pred, jnp.float64)

                    return pred

                single_batch_vec_init = jit(vmap(word_vec_initilisation)) # vmap over initial states in a tree

                def combine(vecs, rule_offset): # update correct tree state according to rule offsets
                    rule = rule_offset[:-2]
                    idx1, idx2 = jnp.array(rule_offset[-2:], dtype=jnp.int32)

                    input_vec1 = vecs[idx1]
                    input_vec2 = vecs[idx2]
                    new_vec = PQC(jnp.array(rule), input_vec1, input_vec2)
                    out = vecs.at[idx1].set(new_vec)

                    return out, out

                def contract(word_params, rules, offsets, class_params): # scan contractions down the tree
                    input_vecs = single_batch_vec_init(word_params)
                    rules_offs = jnp.concatenate((rules, offsets), axis=1)
                    input_vecs, _ = jax.lax.scan(combine, init=input_vecs, xs=rules_offs)
                    preds = measure(input_vecs[0], class_params)
                    return preds

                apply_batch_contractions = jit(vmap(contract, (0,0,0,None))) # vmap over all trees in batch

                def loss(params, batch_words, batch_rules, batch_offsets, labels):
                    
                    preds = apply_batch_contractions(params['words'][batch_words], params['rules'][batch_rules], batch_offsets, params['class'])

                    if post_sel: # renormalise output
                        preds = preds / jnp.sum(preds, axis=1)[:,None]

                    assert use_jit or all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)
                    
                    out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, jnp.array(labels))])

                    return  -jnp.sum(out)

                val_n_grad = jit(value_and_grad(loss))

                def train_step(params, opt_state, batch_words, batch_rules, batch_offsets, batch_labels):
                    
                    cost, grads = val_n_grad(params, jnp.array(batch_words), jnp.array(batch_rules), jnp.array(batch_offsets), jnp.array(batch_labels))
                
                    if use_grad_clip: 
                        grads["words"] = jnp.clip(grads["words"], -grad_clip, grad_clip)
                        grads["rules"] = jnp.clip(grads["rules"], -grad_clip, grad_clip)
                        grads["class"] = jnp.clip(grads["class"], -grad_clip, grad_clip)

                    # update relevant params
                    if use_optax_reg is True:
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)

                    else:            
                        updates, opt_state = optimizer.update(grads, opt_state)
                        params = optax.apply_updates(params, updates)

                    return cost, params, opt_state, grads

                def get_accs(params, batch_words, batch_rules, batch_offsets, labels):
                    
                    preds = apply_batch_contractions(params['words'][jnp.array(batch_words)], params['rules'][jnp.array(batch_rules)], jnp.array(batch_offsets), params['class'])
                    
                    if post_sel:
                        preds = preds / jnp.sum(preds, axis=1)[:,None]

                    assert use_jit or all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

                    acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2 

                    return acc

                def initialise_params(no_words, word_emb_size, no_rules, rule_embed_size): # initialise word, rule and classification params
                    word_params = jnp.array(np.random.uniform(0, init_val, size=(no_words+1, word_emb_size)))
                    rule_params = jnp.array(np.random.uniform(0, init_val, size=(no_rules+1, rule_embed_size)))
                    class_params = jnp.array(np.random.uniform(0, init_val, size=(word_emb_size)))
                    return word_params, rule_params, class_params

                def get_batches(words, rules, offsets, labels, batch_size):

                    slices = list(gen_batches(len(labels), batch_size))
                    batched_w = [words[s] for s in slices]
                    batched_r = [rules[s] for s in slices]
                    batched_o = [offsets[s] for s in slices]
                    batched_l = [labels[s] for s in slices]

                    return zip(batched_w, batched_r, batched_o, batched_l)

                def pad_trees(batch_words, batch_rules, batch_offsets, max_words, words_pad_idx, rules_pad_idx):
                    pad_words = []
                    pad_rules = []
                    pad_offsets = []
                    for words, rules, offsets in zip(batch_words, batch_rules, batch_offsets):
                        pad_len = max_words-len(words)
                        if pad_len!=0:
                            pad_words.append(np.concatenate((words, words_pad_idx*np.ones(shape=(pad_len), dtype=np.int32))))
                            if len(words)==1:
                                pad_rules.append(rules_pad_idx*np.ones(shape=(pad_len), dtype=np.int32))
                                pad_offsets.append([[int(max_words-2), int(max_words-1)]]*(pad_len)) # void pad indices
                            else:
                                pad_rules.append(np.concatenate((rules, rules_pad_idx*np.ones(shape=(pad_len), dtype=np.int32))))
                                pad_offsets.append(np.concatenate((offsets, [[int(max_words-2), int(max_words-1)]]*(pad_len)))) # void pad indices
                        else:
                            pad_words.append(words)
                            pad_rules.append(rules)
                            pad_offsets.append(offsets)
                    return pad_words, pad_rules, pad_offsets

                no_words = max(w2i.values())
                no_rules = max(r2i.values())

                print("Rule(s): ", r2i.keys())
                print("Number of unique tokens: ", no_words+1)
                print("Number of unique rules: ", no_rules+1)

                if ansatz == 'A14':
                    if n_qubits == 1:
                        word_emb_size = 3
                    else:
                        word_emb_size = n_qubits * 4 * n_layers
                    rule_embed_size = 2 * n_qubits * 4 * n_layers
                elif ansatz == 'A9':
                    if n_qubits == 1:
                        word_emb_size = 3
                    else:
                        word_emb_size = n_qubits * n_layers
                    rule_embed_size = 2 * n_qubits * n_layers
                elif ansatz == 'IQP':
                    if n_qubits == 1:
                        word_emb_size = 3
                    else:
                        word_emb_size = (n_qubits-1) * n_layers
                    rule_embed_size = (2 * n_qubits-1) * n_layers

                # pad to max tree width and batch size
                words_pad_idx = no_words # void pad params
                rules_pad_idx = no_rules

                max_words = max([len(words) for words in np.concatenate((train_data["words"], val_data["words"], test_data["words"]))])

                if mixed:
                    test_density_matrix = jnp.array(range(16)).reshape(2,2,2,2)
                    eval_density_matrix = make_vec(2, test_density_matrix).eval(**eval_args).array 

                    if not np.allclose(test_density_matrix, eval_density_matrix):
                        raise Exception("You need to install the GitHub version of discopy, check README.")

                if use_optax_reg is True:
                    optim = 'adamW'
                    optimizer = optax.adamw(lr)
                else:
                    optim = 'adam'
                    optimizer = optax.adam(lr)

                print("Optimizer: ", optim)
                print("lr: ", lr)

                # initialise optax opxtimiser
                word_params, rule_params, class_params = initialise_params(no_words+1, word_emb_size, no_rules+1, rule_embed_size)
                params = {'words': word_params, 'rules': rule_params, 'class': class_params}
                opt_state = optimizer.init(params)

                val_accuracies = []
                test_accuracies = []
                losses = []
                all_params = []
                all_opt_states = []
                all_grads = []

                # initial acc
                sum_accs = []
                for i, (batch_words, batch_rules, batch_offsets, batch_labels) in tqdm(enumerate(get_batches(val_data["words"], val_data["rules"], val_data["offsets"], val_data["labels"], batch_size))):
                    pad_words, pad_rules, pad_offsets = pad_trees(batch_words, batch_rules, batch_offsets, max_words, words_pad_idx, rules_pad_idx)
                    pad_words = np.array(pad_words, dtype = np.int64)
                    acc = get_accs(params, pad_words, pad_rules, pad_offsets, batch_labels)
                    sum_accs.append(acc)
                
                val_acc = sum(sum_accs) / len(val_data["labels"])
                val_accuracies.append(val_acc)
                print("Initial Acc  {:0.2f}  ".format(val_acc))

                for epoch in range(no_epochs):

                    start_time = time.time()

                    # calc cost and update params
                    sum_loss = []
                    for batch_words, batch_rules, batch_offsets, batch_labels in tqdm(get_batches(train_data["words"], train_data["rules"], train_data["offsets"], train_data["labels"], batch_size)): # all length x examples batches together for jax 
                        pad_words, pad_rules, pad_offsets = pad_trees(batch_words, batch_rules, batch_offsets, max_words, words_pad_idx, rules_pad_idx)
                        pad_words = np.array(pad_words, dtype = np.int64)
                        cost, params, opt_state, grads = train_step(params, opt_state, pad_words, pad_rules, pad_offsets, batch_labels)
                        sum_loss.append(cost)

                    sum_accs = []
                    for i, (batch_words, batch_rules, batch_offsets, batch_labels) in tqdm(enumerate(get_batches(val_data["words"], val_data["rules"], val_data["offsets"], val_data["labels"], batch_size))):
                        pad_words, pad_rules, pad_offsets = pad_trees(batch_words, batch_rules, batch_offsets, max_words, words_pad_idx, rules_pad_idx)
                        pad_words = np.array(pad_words, dtype = np.int64)
                        acc = get_accs(params, pad_words, pad_rules, pad_offsets, batch_labels)
                        sum_accs.append(acc)

                    val_acc = sum(sum_accs) / len(val_data["labels"])
                    cost = sum(sum_loss)/ len(train_data["labels"])
                    epoch_time = time.time() - start_time
                    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                    print("Loss  {:0.2f}  ".format(cost))
                    print("Acc  {:0.2f}  ".format(val_acc))
                    val_accuracies.append(val_acc)
                    losses.append(cost)
                    all_params.append(params)

                    sum_accs = []
                    for i, (batch_words, batch_rules, batch_offsets, batch_labels) in tqdm(enumerate(get_batches(test_data["words"], test_data["rules"], test_data["offsets"], test_data["labels"], batch_size))):
                        pad_words, pad_rules, pad_offsets = pad_trees(batch_words, batch_rules, batch_offsets, max_words, words_pad_idx, rules_pad_idx)
                        pad_words = np.array(pad_words, dtype = np.int64)
                        acc = get_accs(params, pad_words, pad_rules, pad_offsets, batch_labels)
                        sum_accs.append(acc)
                    
                    test_acc = sum(sum_accs) / len(test_data["labels"])
                    test_accuracies.append(test_acc)
                    print("Test set accuracy: ", test_acc)

                    # ------------------------------ SAVE DATA -----------------–------------ #
                    if reduce_train:
                        if post_sel:
                            if cut:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/post_sel/{thr}_{number_of_structures}_REDUCED_{reduce_val}/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                            else:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/post_sel/REDUCED_{reduce_val}/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                        else:
                            if cut:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/discards/{thr}_{number_of_structures}_REDUCED_{reduce_val}/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                            else:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/discards/REDUCED_{reduce_val}/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                    else:
                        if post_sel:
                            if cut:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/post_sel/{thr}_{number_of_structures}/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                            else:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/post_sel/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                        else:
                            if cut:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/discards/{thr}_{number_of_structures}/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                            else:
                                save_path = f'Results/{save_name}/{model}/{parse_type}/discards/{n_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                                        
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    pickle.dump(obj=all_params, file=open(f'{save_path}{"param_dict"}', 'wb'))
                    pickle.dump(obj=opt_state, file=open(f'{save_path}{"final_opt_state"}', 'wb'))
                    pickle.dump(obj=test_accuracies, file=open(f'{save_path}{"test_accs"}', 'wb'))
                    pickle.dump(obj=val_accuracies, file=open(f'{save_path}{"val_accs"}', 'wb'))
                    pickle.dump(obj=losses, file=open(f'{save_path}{"loss"}', 'wb'))
                    pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
                  # ------------------------------- SAVE DATA ---------------–------------ #
