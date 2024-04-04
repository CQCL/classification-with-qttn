import numpy as np
import pickle
from pathlib import Path
import math    
from sklearn.model_selection import train_test_split
from lambeq import stairs_reader
from discopy.rigid import Box, Id

data_name = 'protein-binding' # length 50 dna seqs
w2i = dict({'A':0, 'G':1, 'C': 2, 'T':3})
thr = 64 # must be 2^N
pad_idx = 4

with open('Data/protein-binding_seqs.txt') as f:
    seqs = [line.strip() for line in f]

with open('Data/protein-binding_labels.txt') as f:
    labels = [line.strip() for line in f]

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

def save_processed_data(save_path, train_dict, val_dict, test_dict, w2i, r2i = None):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
    pickle.dump(obj=train_dict, file=open(f'{save_path}{"train_data"}', 'wb'))
    pickle.dump(obj=val_dict, file=open(f'{save_path}{"val_data"}', 'wb'))
    pickle.dump(obj=test_dict, file=open(f'{save_path}{"test_data"}', 'wb'))
    if r2i is not None: 
        pickle.dump(obj=r2i, file=open(f'{save_path}{"r2i"}', 'wb'))

seqs = [s[:thr] for s in seqs]
data_labels = [[1,0] if label == '0' else [0,1] for label in labels]
train_sents, val_sents, train_labels, val_labels = train_test_split(seqs, data_labels, test_size=0.1, random_state=0, shuffle=True)
train_sents, test_sents, train_labels, test_labels = train_test_split(train_sents, train_labels, test_size=0.1111, random_state=0, shuffle=True)

train_words, train_labels = pad_CTN(train_sents, train_labels, thr, w2i, pad_idx)
val_words, val_labels = pad_CTN(val_sents, val_labels, thr, w2i, pad_idx)
test_words, test_labels = pad_CTN(test_sents, test_labels, thr, w2i, pad_idx)
train_dict_CTN = {"words": train_words, "labels": train_labels}
val_dict_CTN = {"words": val_words, "labels": val_labels}
test_dict_CTN = {"words": test_words, "labels": test_labels}

save_path = f'Data/CTN/{data_name}_cut_{thr}/'    
save_processed_data(save_path, train_dict_CTN, val_dict_CTN, test_dict_CTN, w2i)

# TTN - also save prepadded (to 2^n) examples 
# this data must then be processed in TTN_preprocessing to decompose contraction into fast jax instructions ! 
save_path = f'Data/TTN/prepad/{data_name}_cut_{thr}/'  
save_processed_data(save_path, train_dict_CTN, val_dict_CTN, test_dict_CTN, w2i)

# PTN - no need to pad sents for PTN 
train_dict_PTN = {"words": train_sents, "labels": train_labels[-1]} 
val_dict_PTN = {"words": val_sents, "labels": val_labels[-1]}
test_dict_PTN = {"words": test_sents, "labels": test_labels[-1]}
save_path = f'Data/PTN/{data_name}_cut_{thr}/'  
print(save_path)
save_processed_data(save_path, train_dict_PTN, val_dict_PTN, test_dict_PTN, w2i)


