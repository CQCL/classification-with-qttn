import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import math    
from sklearn.model_selection import train_test_split
from lambeq import stairs_reader
from discopy.rigid import Box, Id

data_name = 'genome' # length 50 dna seqs
w2i = dict({'A':0, 'G':1, 'C': 2, 'T':3})
thr = 32
pad_idx = 4

with open('Data/genome_seqs.txt') as f:
    seqs = [line.strip() for line in f]

with open('Data/genome_labels.txt') as f:
    labels = [line.strip() for line in f]

if thr < 64:
    seqs = [s[:thr] for s in seqs]

train_labels = []
for label in labels:
    if label == '0':
        train_labels.append([1,0])
    elif label == '1':
        train_labels.append([0,1])

def rename_box(layer, name):
    left, box, right = layer
    return Id(left) @ Box(name, box.dom, box.cod) @ Id(right)

def hPTN(sentence):
    diagram = stairs_reader.sentence2diagram(sentence, tokenised=True)
    cuts = diagram.foliation().boxes
    new_diagram = cuts[0]
    for i, cut in enumerate(cuts[1:]):
        new_diagram = new_diagram.then(
            *[rename_box(layer, f'layer_{i}') for layer in cut.layers])
    return new_diagram

def uPTN(sentence):
    diagram = stairs_reader.sentence2diagram(sentence, tokenised=True)
    cuts = diagram.foliation().boxes
    new_diagram = cuts[0]
    for i, cut in enumerate(cuts[1:]):
        new_diagram = new_diagram.then(
            *[rename_box(layer, 'UNIBOX') for layer in cut.layers])
    return new_diagram 

def get_diagrams_PTN(texts, labels):
    uPTN = []
    hPTN = []
    parsed_labels = []
    for label, review in zip(labels, texts):
        review = list(review)
        try:
            h = hPTN(review)
            u = uPTN(review)
            uPTN.append(u)
            hPTN.append(h)
            parsed_labels.append(label)
        except:
            # print("Could not parse: ", review)x
            continue
    return uPTN, hPTN, parsed_labels

def tree_process(input_trees, w2i, r2i):
    batch_words = []
    batch_rules = []
    batch_offsets = []
    for tree in input_trees:
        tree_words = [w2i[box.name] for box in tree.foliation()[0].boxes]
        n_words = len(tree_words)
        tree_rules = []
        idxs = list(range(n_words))
        pos = []
        for i, layer in enumerate(tree.foliation()[1:]):
            for box, offset in zip(layer.boxes, layer.offsets):
                if len(box.dom) == 1: # type-raising
                    continue
                else:
                    if parse_type == 'unibox':
                        tree_rules.append(0)
                    else:
                        tree_rules.append(r2i[box.name])
                    pos.append((idxs[offset], idxs[offset+1]))
                    del idxs[offset+1]
        batch_rules.append(tree_rules)
        batch_words.append(tree_words)
        batch_offsets.append(pos)
    return batch_words, batch_rules, batch_offsets

def pad_CTN(txt, labels, max_N, w2i, pad_idx):
    assert math.log(max_N, 2).is_integer(), "Please provide max_N = 2^n"
    lens = [np.power(2,n) for n in range(1, int(math.log(max_N, 2)+1))]
    lens.insert(0,0)
    ordered_pads = []
    ordered_labels = []
    for i in range(1, len(lens)):
        temp = []
        temp_labels = []
        for sent, label in zip(txt, labels):
            if lens[i-1] < len(sent) <= lens[i]:
                words = [w2i[w] for w in sent]
                n_pad = lens[i] - len(words)
                left = n_pad // 2
                right = n_pad - left
                left_pad = np.ones(left, dtype=int)*pad_idx
                right_pad = np.ones(right, dtype=int)*pad_idx
                temp.append(np.concatenate((left_pad, words, right_pad)))
                temp_labels.append(label)
        ordered_pads.append(temp)
        ordered_labels.append(temp_labels)
    return ordered_pads, ordered_labels

train_sents, val_sents, train_labels, val_labels = train_test_split(seqs, train_labels, test_size=0.1, random_state=0, shuffle=True)
train_sents, test_sents, train_labels, test_labels = train_test_split(train_sents, train_labels, test_size=0.1111, random_state=0, shuffle=True)

train_words, train_labels = pad_CTN(train_sents, train_labels, thr, w2i, pad_idx)
val_words, val_labels = pad_CTN(val_sents, val_labels, thr, w2i, pad_idx)
test_words, test_labels = pad_CTN(test_sents, test_labels, thr, w2i, pad_idx)
train_dict_CTN = {"words": train_words[-1], "labels": train_labels[-1]}
val_dict_CTN = {"words": val_words[-1], "labels": val_labels[-1]}
test_dict_CTN = {"words": test_words[-1], "labels": test_labels[-1]}

print("CTN Train examples: ", len(train_dict_CTN["words"]))
print("CTN Validation examples: ", len(val_dict_CTN["words"]))
print("CTN Test examples: ",len(test_dict_CTN["words"]))

if thr < 64:
    save_path = f'Data/CTN/{data_name}_cut_{thr}/'    
else:
    save_path = f'Data/CTN/{data_name}/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=train_dict_CTN, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_dict_CTN, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_dict_CTN, file=open(f'{save_path}{"test_data"}', 'wb'))
if thr < 64:
    save_path = f'Data/TTN/{data_name}_cut_LAST_{thr}/'    
else:
    save_path = f'Data/TTN/{data_name}/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=train_dict_CTN, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_dict_CTN, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_dict_CTN, file=open(f'{save_path}{"test_data"}', 'wb'))
# # -------------------------------------- PTN process----------------------------------- #

# get r2i
max_words = thr
all_rules = ['layer_'+str(i) for i in range(max_words-1)]
indx = [i for i in range(max_words-1)]
Hr2i = dict(zip(all_rules, indx))
Ur2i = dict({'UNIBOX':0}) 

train_sents = [list(s) for s in train_sents]
val_sents = [list(s) for s in val_sents]
test_sents = [list(s) for s in test_sents]

uPTN_train_dict = {"words": [[w2i[s] for s in sent] for sent in train_sents], "rules":[np.zeros(len(s), dtype=int) for s in train_sents] , "offsets": [[(0,i+1) for i in range(len(s))] for s in train_sents], "labels": train_labels[-1]}
uPTN_val_dict = {"words": [[w2i[s] for s in sent] for sent in val_sents], "rules":[np.zeros(len(s), dtype=int) for s in val_sents] , "offsets": [[(0,i+1) for i in range(len(s))] for s in val_sents], "labels": val_labels[-1]}
uPTN_test_dict = {"words": [[w2i[s] for s in sent] for sent in test_sents], "rules":[np.zeros(len(s), dtype=int) for s in test_sents] , "offsets": [[(0,i+1) for i in range(len(s))] for s in test_sents], "labels": test_labels[-1]}
 
print("hPTN Train examples: ", len(uPTN_train_dict["words"]))
print("hPTN Validation examples: ", len(uPTN_val_dict["words"]))
print("hPTN Test examples: ",len(uPTN_test_dict["words"]))

if thr < 64:
    save_path = f'Data/PTN/{data_name}_cut_{thr}/unibox'    
else:
    save_path = f'Data/PTN/{data_name}/unibox'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=Hr2i, file=open(f'{save_path}{"r2i"}', 'wb'))
pickle.dump(obj=uPTN_train_dict, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=uPTN_val_dict, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=uPTN_test_dict, file=open(f'{save_path}{"test_data"}', 'wb'))

hPTN_train_dict = {"words": [[w2i[s] for s in sent] for sent in train_sents], "rules":[list(range(len(s))) for s in train_sents] , "offsets": [[(0,i+1) for i in range(len(s))] for s in train_sents], "labels": train_labels[-1]}
hPTN_val_dict = {"words": [[w2i[s] for s in sent] for sent in val_sents], "rules":[list(range(len(s))) for s in val_sents] , "offsets": [[(0,i+1) for i in range(len(s))] for s in val_sents], "labels": val_labels[-1]}
hPTN_test_dict = {"words": [[w2i[s] for s in sent] for sent in test_sents], "rules":[list(range(len(s))) for s in test_sents] , "offsets": [[(0,i+1) for i in range(len(s))] for s in test_sents], "labels": test_labels[-1]}
 
print("hPTN Train examples: ", len(hPTN_train_dict["words"]))
print("hPTN Validation examples: ", len(hPTN_val_dict["words"]))
print("hPTN Test examples: ",len(hPTN_test_dict["words"]))

if thr < 64:
    save_path = f'Data/PTN/{data_name}_cut_{thr}/height'    
else:
    save_path = f'Data/PTN/{data_name}/height'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=Hr2i, file=open(f'{save_path}{"r2i"}', 'wb'))
pickle.dump(obj=hPTN_train_dict, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=hPTN_val_dict, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=hPTN_test_dict, file=open(f'{save_path}{"test_data"}', 'wb'))
# -------------------------------------- PTN process----------------------------------- #



