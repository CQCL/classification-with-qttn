from pathlib import Path
import numpy as np
import jax.numpy as jnp
from discopy.quantum import Id, Ket, H, Bra, CZ
from discopy.quantum.circuit import Box, Measure, Discard
import tensornetwork as tn
from jax import vmap, value_and_grad, vmap
from sklearn.utils import gen_batches
from tqdm import tqdm
import time 
import pickle
import numpy as np
import pickle
import optax

import warnings
warnings.filterwarnings('ignore')

from discopy import Tensor
Tensor.set_backend("jax")
tn.set_default_backend("jax")

np.random.seed(0)

# ------------------------------------- SETTINGS -------------------------------- #
model = 'SCTN'
data_name = 'clickbait'
reduce_train = True
reduce_val = 2031
thr = 16
number_of_structures = 100
use_jit = True
post_sel = True
use_grad_clip = True
use_optax_reg = True
include_final_classification_box = True
grad_clip = 100.0
no_epochs = 25
batch_size = 64
init_val = 0.01
ansatz = 'A14'
n_word_qubits = 1 # number of qubits per word
# ------------------------------------- SETTINGS -------------------------------- #

for parse_type in ['unibox', 'height', 'rule']:

    # ------------------------------- READ IN DATA ---------------------------- #
    if reduce_train:
        save_path = f'Data/{model}/{data_name}/{thr}_{number_of_structures}_REDUCED_{reduce_val}/{parse_type}/'
    else:
        save_path = f'Data/{model}/{data_name}/{thr}_{number_of_structures}/{parse_type}/'
    
    w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
    r2i = pickle.load(file=open(f'{save_path}{"r2i"}', 'rb'))
    ordered_train_data = pickle.load(file=open(f'{save_path}{"train_data"}', 'rb'))
    ordered_val_data = pickle.load(file=open(f'{save_path}{"val_data"}', 'rb'))
    ordered_test_data = pickle.load(file=open(f'{save_path}{"test_data"}', 'rb'))

    n = np.sum([len([l for l in labels["labels"]]) for labels in ordered_train_data])
    print("Number of train examples: ", n) 
    n = np.sum([len([l for l in labels["labels"]]) for labels in ordered_val_data])
    print("Number of val examples: ", n) 
    n = np.sum([len([l for l in labels["labels"]]) for labels in ordered_test_data])
    print("Number of test examples: ", n) 
    # ------------------------------- READ IN DATA ----------------------------- #

    for n_layers in [1, 2]:

        for lr in [0.01, 0.001, 0.0001]:

            n_rule_qubits = 2 * n_word_qubits  # number of qubits in each sub-PQC
            if not use_jit: 
                jit = lambda x: x
            eval_args = {}
            mixed = not post_sel
            eval_args['mixed'] = mixed  # use density matrices if discarding 
            eval_args['contractor'] = tn.contractors.auto  # use tensor networks if speed

            print("Number of epochs: ", no_epochs)
            print("Batch size: ", batch_size)
            print("Ansatz: ", ansatz)
            print("Parse type: ", parse_type)
            print("Number of word qubits: ", n_word_qubits)
            print("Number of layers: ", n_layers)
            print("Include classification box: ", include_final_classification_box)
            print("Using post selection: ", post_sel)
            print("Using gradient clipping: ", use_grad_clip)

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

            single_batch_vec_init = vmap(word_vec_initilisation)

            def make_words(data, name): # return discopy box from post discard density matrix
                from discopy.quantum.circuit import qubit as d_qb
                dom, cod = d_qb ** 0, d_qb ** n_word_qubits
                box = Box(name, dom, cod, is_mixed=mixed, data=data)
                box.array = box.data
                return box

            def make_state(data, name):
                from discopy.quantum.circuit import qubit as d_qb
                dom, cod = d_qb ** (2 * n_word_qubits), d_qb ** (2 * n_word_qubits)
                box = Box(name, dom, cod, is_mixed=mixed, data=data)
                box.array = box.data
                return box

            def SCTN(W_params, U_params, I_params, Us, Is, class_params):

                words = [make_words(vec, "w") for vec in single_batch_vec_init(W_params)]
                circ = Id().tensor(*words)

                N = len(W_params)*n_word_qubits

                count = 0 
                
                for idx, (U_inds, I_inds) in enumerate(zip(Us, Is)):  

                    # apply unitary ops
                    for u in U_inds:
                        u = u*n_word_qubits
                        circ >>= Id(u)@Ansatz(n_rule_qubits, n_layers, U_params[idx], ansatz)@Id((N-count)-u-2*n_word_qubits)

                    # apply discard op
                    I_inds = I_inds*n_word_qubits
                    circ >>= Id(I_inds)@Ansatz(n_rule_qubits, n_layers, I_params[idx], ansatz)@Id((N-count)-I_inds-2*n_word_qubits)
    
                    if post_sel:
                        circ >>= Id(I_inds)@ Id().tensor(*[Id(1) @ Bra(0)] * n_word_qubits) @ Id((N-count)-(I_inds+n_word_qubits)-n_word_qubits)
                    else: 
                        circ >>= Id(I_inds)@ Id().tensor(*[Id(1) @ Discard()] * n_word_qubits) @ Id((N-count)-(I_inds+n_word_qubits)-n_word_qubits)
                    
                    count+=n_word_qubits

                # apply final classification ansatz
                if include_final_classification_box:
                    circ >>= Ansatz(n_word_qubits, n_layers, class_params, ansatz)

                # measure the middle qubit
                left = n_word_qubits // 2
                right = n_word_qubits - left - 1

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

            vmap_mera = vmap(SCTN, (0, 0, 0, None, None, None))

            def loss(params, batch_words, batch_rules, batch_Us, batch_Is, labels):
                
                preds = vmap_mera(params['words'][batch_words], params['Us'][batch_rules], params['Is'][batch_rules],  batch_Us, batch_Is, params['class'])

                if post_sel: # renormalise output
                    preds = preds / jnp.sum(preds, axis=1)[:,None]

                assert use_jit or all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)
                
                out = jnp.array([jnp.dot(label, jnp.log(pred+1e-7)) for pred, label in zip(preds, jnp.array(labels))])

                return  -jnp.sum(out)

            val_n_grad = value_and_grad(loss)

            def train_step(params, opt_state, batch_words, batch_rules, batch_Us, batch_Is, batch_labels):

                cost, grads = val_n_grad(params, jnp.array(batch_words), jnp.array(batch_rules), batch_Us, batch_Is, batch_labels)

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

            def get_accs(params, batch_words, batch_rules, batch_Us, batch_Is, labels):

                preds = vmap_mera(params['words'][jnp.array(batch_words)], params['Us'][jnp.array(batch_rules)], params['Is'][jnp.array(batch_rules)],  batch_Us, batch_Is, params['class'])

                if post_sel:
                    preds = preds / jnp.sum(preds, axis=1)[:,None]

                assert use_jit or all(jnp.allclose(jnp.sum(pred), jnp.ones(1), atol=1e-3)  for pred in preds)

                acc = np.sum([np.round(pred) == np.array(label, float) for pred, label in zip(preds, labels)]) / 2

                return acc

            def initialise_params(no_words, no_rules, word_emb_size, rule_embed_size): # initialise word, rule and classification params
                word_params = jnp.array(np.random.uniform(0, init_val, size=(no_words, word_emb_size)))
                U_params = jnp.array(np.random.uniform(0, init_val, size=(no_rules, rule_embed_size)))
                I_params = jnp.array(np.random.uniform(0, init_val, size=(no_rules, rule_embed_size)))
                class_params = jnp.array(np.random.uniform(0, init_val, size=(word_emb_size)))
                return word_params, U_params, I_params, class_params

            def get_batches(words, rules, labels, batch_size):

                slices = list(gen_batches(len(labels), batch_size))
                batched_w = [words[s] for s in slices]
                batched_r = [rules[s] for s in slices]
                batched_l = [labels[s] for s in slices]

                return zip(batched_w, batched_r, batched_l)

            no_words = max(w2i.values())
            no_rules = max(r2i.values())

            print("Rule(s): ", no_rules+1)
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

            # initialise optax optimiser
            word_params, U_params, I_params, class_params = initialise_params(no_words+1, no_rules+1, word_emb_size, rule_embed_size)
            params = {'words': word_params, 'Us': U_params, 'Is': I_params, 'class': class_params}
            opt_state = optimizer.init(params)

            val_accuracies = []
            test_accuracies = []
            losses = []
            all_params = []
            all_opt_states = []
            all_grads = []

            sum_accs = []
            for data in tqdm(ordered_val_data):
                if len(data["labels"]):
                    Us = data["U_offsets"][0]
                    Is = data["I_offsets"][0]
                    for i, (batch_words, batch_rules, batch_labels) in enumerate(get_batches(data["words"], data["rules"], data["labels"], batch_size)): # all length x examples batches together for jax 
                        batch_words = np.array(batch_words, np.int64)
                        batch_rules = np.array(batch_rules, np.int64)
                        acc = get_accs(params, batch_words, batch_rules, Us, Is, batch_labels)
                        sum_accs.append(acc)

            val_acc = np.sum(sum_accs) / np.sum([len(l['labels']) for l in ordered_val_data])
            print("Initial acc  {:0.2f}  ".format(val_acc))
            val_accuracies.append(val_acc)

            for epoch in range(no_epochs):

                start_time = time.time()

                # calc cost and update params
                sum_loss = []
                for data in tqdm(ordered_train_data):
                    if len(data["labels"]):
                        Us = data["U_offsets"][0]
                        Is = data["I_offsets"][0]
                        for i, (batch_words, batch_rules, batch_labels) in enumerate(get_batches(data["words"], data["rules"], data["labels"], batch_size)): # all length x examples batches together for jax 
                            batch_words = np.array(batch_words, np.int64)
                            batch_rules = np.array(batch_rules, np.int64)
                            cost, params, opt_state, grads = train_step(params, opt_state, batch_words, batch_rules, Us, Is, batch_labels)
                            sum_loss.append(cost)

                sum_accs = []
                for data in tqdm(ordered_val_data):
                    if len(data["labels"]):
                        Us = data["U_offsets"][0]
                        Is = data["I_offsets"][0]
                        for i, (batch_words, batch_rules, batch_labels) in enumerate(get_batches(data["words"], data["rules"], data["labels"], batch_size)): # all length x examples batches together for jax 
                            batch_words = np.array(batch_words, np.int64)
                            batch_rules = np.array(batch_rules, np.int64)
                            acc = get_accs(params, batch_words, batch_rules, Us, Is, batch_labels)
                            sum_accs.append(acc)

                val_acc = np.sum(sum_accs) / np.sum([len(l['labels']) for l in ordered_val_data])
                cost = np.sum(sum_loss)/ np.sum([len(l['labels']) for l in ordered_train_data])
                epoch_time = time.time() - start_time
                print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                print("Loss  {:0.2f}  ".format(cost))
                print("Acc  {:0.2f}  ".format(val_acc))
                val_accuracies.append(val_acc)
                losses.append(cost)
                all_params.append(params)

                sum_accs = []
                for data in tqdm(ordered_test_data):
                    if len(data["labels"]):
                        Us = data["U_offsets"][0]
                        Is = data["I_offsets"][0]
                        for i, (batch_words, batch_rules, batch_labels) in enumerate(get_batches(data["words"], data["rules"], data["labels"], batch_size)): # all length x examples batches together for jax 
                            batch_words = np.array(batch_words, np.int64)
                            batch_rules = np.array(batch_rules, np.int64)
                            acc = get_accs(params, batch_words, batch_rules, Us, Is, batch_labels)
                            sum_accs.append(acc)

                test_acc = np.sum(sum_accs) / np.sum([len(l['labels']) for l in ordered_test_data]) 
                print("Test set accuracy: ", test_acc)
                test_accuracies.append(test_acc)

                # ------------------------------ SAVE DATA -----------------–------------ #
                if reduce_train:
                    if post_sel:
                        save_path = f'Results/{data_name}/{model}/{parse_type}/post_sel/{thr}_{number_of_structures}_REDUCED_{reduce_val}/{n_word_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                    else:
                        save_path = f'Results/{data_name}/{model}/{parse_type}/discards/{thr}_{number_of_structures}_REDUCED_{reduce_val}/{n_word_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                else:
                    if post_sel:
                        save_path = f'Results/{data_name}/{model}/{parse_type}/post_sel/{thr}_{number_of_structures}/{n_word_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                    else:
                        save_path = f'Results/{data_name}/{model}/{parse_type}/discards/{thr}_{number_of_structures}/{n_word_qubits}qb_{n_layers}lay_{ansatz}/{optim}_lr_{lr}_batch_{batch_size}/'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                pickle.dump(obj=all_params, file=open(f'{save_path}{"param_dict"}', 'wb'))
                pickle.dump(obj=opt_state, file=open(f'{save_path}{"final_opt_state"}', 'wb'))
                pickle.dump(obj=test_accuracies, file=open(f'{save_path}{"test_accs"}', 'wb'))
                pickle.dump(obj=val_accuracies, file=open(f'{save_path}{"val_accs"}', 'wb'))
                pickle.dump(obj=losses, file=open(f'{save_path}{"loss"}', 'wb'))
                pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
                # ------------------------------- SAVE DATA ---------------–------------ #

