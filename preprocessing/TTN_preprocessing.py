import numpy as np
import math
import pickle 
from pathlib import Path

data_name = 'protein-binding'
model = 'TTN'
thr = 32

def flatten_list(li):
    return [y for x in li for y in x]

if data_name == 'protein-binding':
    print("Reading in data ... ")
    if thr == 64:
        save_path = f'Data/{model}/{data_name}/'
    else:
        save_path = f'Data/{model}/{data_name}_cut_{thr}/'
    w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
    train_data = pickle.load(file=open(f'{save_path}{"train_data"}', 'rb'))
    val_data = pickle.load(file=open(f'{save_path}{"val_data"}', 'rb'))
    test_data = pickle.load(file=open(f'{save_path}{"test_data"}', 'rb'))

elif data_name in ['clickbait', 'rotten-tomatoes']:
    data_name = f'{data_name}_full'
    parse_type = 'unibox'
    save_path = f'Data/{model}/{data_name}/{parse_type}/'
    w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
    train_data = pickle.load(file=open(f'{save_path}{"train_data"}', 'rb'))
    val_data = pickle.load(file=open(f'{save_path}{"val_data"}', 'rb'))
    test_data = pickle.load(file=open(f'{save_path}{"test_data"}', 'rb'))
    train_data["words"] = flatten_list(train_data["words"])
    val_data["words"] = flatten_list(val_data["words"])
    test_data["words"] = flatten_list(test_data["words"])
    train_data["labels"] = flatten_list(train_data["labels"])
    val_data["labels"] = flatten_list(val_data["labels"])
    test_data["labels"] = flatten_list(test_data["labels"])

print("Train examples: ", len(train_data["words"]))
print("Validation examples: ", len(val_data["words"]))
print("Test examples: ",len(test_data["words"]))

train_offsets = []
train_rules_u = []
train_rules_h = []
    
for words in train_data["words"]:
    N = len(words)
    offsets = []
    rules_u = np.zeros(N-1, dtype=np.int64)
    rules_h = []
    for n in range(int(np.log2(N))):
        temp = list(range(N))[::np.power(2, n+1)]
        temp = [(t, t+np.power(2, n)) for t in temp]
        for t in temp:
            offsets.append(t)
        temp_rules = np.ones(len(temp), dtype=np.int64)*n
        for rule in temp_rules:
            rules_h.append(rule)
    train_offsets.append(offsets)
    train_rules_u.append(rules_u)
    train_rules_h.append(rules_h)

val_offsets = []
val_rules_u = []
val_rules_h = []
for words in val_data["words"]:
    N = len(words)
    offsets = []
    rules_u = np.zeros(N-1, dtype=np.int64)
    rules_h = []
    for n in range(int(np.log2(N))):
        temp = list(range(N))[::np.power(2, n+1)]
        temp = [(t, t+np.power(2, n)) for t in temp]
        for t in temp:
            offsets.append(t)
        temp_rules = np.ones(len(temp), dtype=np.int64)*n
        for rule in temp_rules:
            rules_h.append(rule)
    val_offsets.append(offsets)
    val_rules_u.append(rules_u)
    val_rules_h.append(rules_h)

test_offsets = []
test_rules_u = []
test_rules_h = []
for words in test_data["words"]:
    N = len(words)
    offsets = []
    rules_u = np.zeros(N-1, dtype=np.int64)
    rules_h = []
    for n in range(int(np.log2(N))):
        temp = list(range(N))[::np.power(2, n+1)]
        temp = [(t, t+np.power(2, n)) for t in temp]
        for t in temp:
            offsets.append(t)
        temp_rules = np.ones(len(temp), dtype=np.int64)*n
        for rule in temp_rules:
            rules_h.append(rule)
    test_offsets.append(offsets)
    test_rules_u.append(rules_u)
    test_rules_h.append(rules_h)

train_dict_u = dict({'words': train_data['words'], 'rules': train_rules_u, 'offsets': train_offsets, 'labels': train_data['labels']})
val_dict_u = dict({'words': val_data['words'], 'rules': val_rules_u, 'offsets': val_offsets, 'labels': val_data['labels']})
test_dict_u = dict({'words': test_data['words'], 'rules': test_rules_u, 'offsets': test_offsets, 'labels': test_data['labels']})

train_dict_h = dict({'words': train_data['words'], 'rules': train_rules_h, 'offsets': train_offsets, 'labels': train_data['labels']})
val_dict_h = dict({'words': val_data['words'], 'rules': val_rules_h, 'offsets': val_offsets, 'labels': val_data['labels']})
test_dict_h = dict({'words': test_data['words'], 'rules': test_rules_h, 'offsets': test_offsets, 'labels': test_data['labels']})

# get r2i
max_words = int(np.log2(thr))
all_rules = ['layer_'+str(i) for i in range(max_words-1)]
indx = [i for i in range(max_words-1)]
Hr2i = dict(zip(all_rules, indx))
Ur2i = dict({'UNIBOX':0})

if data_name == 'protein-binding':
    if thr == 64:
        save_path = f'Data/{model}/{data_name}/unibox/'
    else:
        save_path = f'Data/{model}/{data_name}_cut_{thr}/unibox/'
else:
    save_path = f'Data/{model}/{data_name}/unibox/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=Ur2i, file=open(f'{save_path}{"r2i"}', 'wb'))
pickle.dump(obj=train_dict_u, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_dict_u, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_dict_u, file=open(f'{save_path}{"test_data"}', 'wb'))

if data_name == 'protein-binding':
    if thr == 64:
        save_path = f'Data/{model}/{data_name}/height/'
    else:
        save_path = f'Data/{model}/{data_name}_cut_{thr}/height/'
else:
    save_path = f'Data/{model}/{data_name}/height/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=Hr2i, file=open(f'{save_path}{"r2i"}', 'wb'))
pickle.dump(obj=train_dict_h, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_dict_h, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_dict_h, file=open(f'{save_path}{"test_data"}', 'wb'))


# PTN
model = 'PTN'    
offsets  = list(range(thr-1))
offsets = [(0, off+1) for off in offsets]
rules_u = np.zeros(thr-1, np.int64)
rules_h = list(range(thr-1))

train_dict_u = dict({'words': train_data['words'], 'rules': [rules_u]*len(train_data['words']), 'offsets': [offsets]*len(train_data['words']), 'labels': train_data['labels']})
val_dict_u = dict({'words': val_data['words'], 'rules': [rules_u]*len(val_data['words']), 'offsets': [offsets]*len(val_data['words']), 'labels': val_data['labels']})
test_dict_u = dict({'words': test_data['words'], 'rules': [rules_u]*len(test_data['words']), 'offsets': [offsets]*len(test_data['words']), 'labels': test_data['labels']})

train_dict_h = dict({'words': train_data['words'], 'rules': [rules_h]*len(train_data['words']), 'offsets': [offsets]*len(train_data['words']), 'labels': train_data['labels']})
val_dict_h = dict({'words': val_data['words'], 'rules': [rules_h]*len(val_data['words']), 'offsets': [offsets]*len(val_data['words']), 'labels': val_data['labels']})
test_dict_h = dict({'words': test_data['words'], 'rules': [rules_h]*len(test_data['words']), 'offsets': [offsets]*len(test_data['words']), 'labels': test_data['labels']})
# get r2i
max_words = int(np.log2(thr))
all_rules = ['layer_'+str(i) for i in range(max_words-1)]
indx = [i for i in range(max_words-1)]
Hr2i = dict(zip(all_rules, indx))
Ur2i = dict({'UNIBOX':0})

if data_name == 'protein-binding':
    if thr == 64:
        save_path = f'Data/{model}/{data_name}/unibox/'
    else:
        save_path = f'Data/{model}/{data_name}_cut_{thr}/unibox/'
else:
    save_path = f'Data/{model}/{data_name}/unibox/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=Ur2i, file=open(f'{save_path}{"r2i"}', 'wb'))
pickle.dump(obj=train_dict_u, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_dict_u, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_dict_u, file=open(f'{save_path}{"test_data"}', 'wb'))

if data_name == 'protein-binding':
    if thr == 64:
        save_path = f'Data/{model}/{data_name}/height/'
    else:
        save_path = f'Data/{model}/{data_name}_cut_{thr}/height/'
else:
    save_path = f'Data/{model}/{data_name}/height/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=Hr2i, file=open(f'{save_path}{"r2i"}', 'wb'))
pickle.dump(obj=train_dict_h, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_dict_h, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_dict_h, file=open(f'{save_path}{"test_data"}', 'wb'))

