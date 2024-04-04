import numpy as np
import math
from pathlib import Path
import pickle
from lambeq import stairs_reader
from discopy.rigid import Box, Id
import itertools
from sklearn.model_selection import train_test_split

data_name = 'clickbait'
include_SCTN = False # only recommended for clickbait data with cut-off number of structures and thr defined 
number_of_structures = 50 # cut-off most common syntactic structures to keep
thr = 64 # cut-off sentence length
reduce_train = True
if data_name == 'clickbait': # suggested values (used in cited experiments)
    reduce_val = 2031
else:
    reduce_val = 2352

def pad_left(v, n, pad_value):
    """ pad vector v on the right with n values of pad_value """
    if n == 0:
        return v
    if len(v) == 0:
        return np.array([pad_value] * n)
    return np.concatenate(([pad_value] * n, v))

def pad_right(v, n, pad_value):
    """ pad vector v on the right with n values of pad_value """
    if n == 0:
        return v
    if len(v) == 0:
        return np.array([pad_value] * n)
    return np.concatenate((v, [pad_value] * n))

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
    
def SCTN_process(input_trees, w2i, r2i):
    batch_words = []
    batch_rules = []
    batch_units = []
    batch_isoms = []

    for tree in input_trees:
        tree_words = [w2i[box.name] for box in tree.foliation()[0].boxes]
        idxs = len(tree_words)
        tree_rules = []
        posUs = []
        posIs = []
        for layer in tree.foliation()[1:]:

            for box, offset in zip(layer.boxes, layer.offsets):
                if len(box.dom) == 1: # type-raising
                    continue
                else:
                    if parse_type == 'unibox':
                        tree_rules.append(0)
                    else:
                        tree_rules.append(r2i[box.name])
                    posIs.append((offset)) # apply isometries at syntactic contractions
                    temp = []
                    if offset-1 >= 0:
                        temp.append(offset-1)
                    if offset+1 < (idxs-1):
                        temp.append(offset+1)
                    posUs.append(temp)
                    idxs -= 1
        batch_rules.append(tree_rules)
        batch_words.append(tree_words)
        batch_units.append(posUs)
        batch_isoms.append(posIs)
    return batch_words, batch_rules, batch_units, batch_isoms

def ordered_SCTN_process(tree_set, label_set, w2i, r2i):
    ordered_processed_trees = []
    for trees, labels in zip(tree_set, label_set):
        words, rules, U_offsets, I_offsets = SCTN_process(trees, w2i, r2i)
        dict = ({"words": words, "rules": rules, "U_offsets": U_offsets, "I_offsets": I_offsets, "labels": labels})
        ordered_processed_trees.append(dict)
    return ordered_processed_trees

def rename_box(layer, name):
    left, box, right = layer
    return Id(left) @ Box(name, box.dom, box.cod) @ Id(right)

def height_tree(diagram):
    cuts = diagram.foliation().boxes
    new_diagram = cuts[0]
    for i, cut in enumerate(cuts[1:]):
        new_diagram = new_diagram.then(
            *[rename_box(layer, f'layer_{i}') for layer in cut.layers])
    return new_diagram

def height_PTN(sentence):
    diagram = stairs_reader.sentence2diagram(sentence, tokenised=True)
    cuts = diagram.foliation().boxes
    new_diagram = cuts[0]
    for i, cut in enumerate(cuts[1:]):
        new_diagram = new_diagram.then(
            *[rename_box(layer, f'layer_{i}') for layer in cut.layers])
    return new_diagram

def unibox_PTN(sentence):
    diagram = stairs_reader.sentence2diagram(sentence, tokenised=True)
    cuts = diagram.foliation().boxes
    new_diagram = cuts[0]
    for i, cut in enumerate(cuts[1:]):
        new_diagram = new_diagram.then(
            *[rename_box(layer, 'UNIBOX') for layer in cut.layers])
    return new_diagram    

def get_diagrams_PTN(texts, labels):
    parsed_trees_str = []
    parsed_labels = []
    for label, review in zip(labels, texts):  
        try:
            if parse_type == 'height':
                d_str = height_PTN(review)
            elif parse_type == 'unibox':
                d_str = unibox_PTN(review)
            parsed_trees_str.append(d_str)
            parsed_labels.append(label)
        except:
            continue
    return parsed_trees_str, parsed_labels

def group_trees(trees):  # group according to shared structure
    shared_structures_ids = []
    for t1, tree1 in enumerate(trees):
        off1 = tree1.offsets
        if t1 not in list(itertools.chain(*shared_structures_ids)):
            shared = []
            shared.append(t1)
            for t2, tree2 in enumerate(trees):
                off2 = tree2.offsets
                if t2 not in shared:
                    if off1 == off2:
                        shared.append(t2)
            shared_structures_ids.append(shared)
    shared_structures_ids = sorted(shared_structures_ids, key=len, reverse=True)
    return shared_structures_ids

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

def flatten_list(li):
    return [y for x in li for y in x]

def trim(sents, trees, labels, thr):
    trim_sents = []
    trim_trees = []
    trim_labels = []
    for sent, tree, label in zip(sents, trees, labels):
        if len(sent) <= thr:
            trim_sents.append(sent)
            trim_trees.append(tree)
            trim_labels.append(label)
    return trim_sents, trim_trees, trim_labels

def rebalance(ind_sets, label_sets, count_thr): # rebalance pos \ neg after cutting for SCTN 
    pos_count = 0
    neg_count = 0
    balanced_inds = []
    balanced_labels = []
    for inds, labels in zip(ind_sets, label_sets):
        temp_inds = []
        temp_labels = []
        for i, label in enumerate(labels):

            if np.allclose(label, [1,0]):          
                if pos_count < count_thr:
                    temp_inds.append(inds[i])
                    temp_labels.append(label)
                    pos_count+=1

            elif np.allclose(label, [0,1]):
                if neg_count < count_thr:
                    temp_inds.append(inds[i])
                    temp_labels.append(label)
                    neg_count+=1

        if len(temp_inds):
            balanced_inds.append(temp_inds)
            balanced_labels.append(temp_labels)
    return balanced_inds, balanced_labels

def reorder_split_inds(ordered_inds, split_inds, split_labels):
    reordered_inds = []
    reordered_labels = []
    for set in ordered_inds:
        temp_inds = []
        temp_labels = []
        for ind, label in zip(split_inds, split_labels):
            if ind in set:
                temp_inds.append(ind)
                temp_labels.append(label)
        reordered_inds.append(temp_inds)
        reordered_labels.append(temp_labels)
    return reordered_inds, reordered_labels

def get_data_balance(data_labels, print_data):
    n_pos = np.sum([len([l for l in labels if np.allclose(l,[1,0])]) for labels in data_labels])
    n_neg = np.sum([len([l for l in labels if np.allclose(l,[0,1])]) for labels in data_labels])
    if print_data is not None:
        print(f'{print_data} pos', n_pos)
        print(f'{print_data} neg', n_neg)
    return n_pos, n_neg

def extract_idx_data(data, idxs):
    return [data[i] for i in idxs]

def extract_data_points(sents, trees, labels, idxs):
    return extract_idx_data(sents, idxs), extract_idx_data(trees, idxs), extract_idx_data(labels, idxs)

def sort_by_idx_after_SCTN_grouping(sents, trees, labels, idxs):
    idxs_sents = [sents[t] for t in flatten_list(idxs)]
    idxs_trees = [trees[t] for t in flatten_list(idxs)]
    idxs_labels = [sents[t] for t in flatten_list(labels)]
    return idxs_sents, idxs_trees, idxs_labels

def save_data(model, save_path, train_dict, val_dict, test_dict, w2i, r2i):
    if include_SCTN:
        if reduce_train:
            save_path = f'Data/{model}/{save_name}/{thr}_{number_of_structures}_REDUCED_{reduce_val}/'
        else:
            save_path = f'Data/{model}/{save_name}/{thr}_{number_of_structures}/'
    else:
        if reduce_train:
            save_path = f'Data/{model}/{save_name}/REDUCED_{reduce_val}/'
        else:
            save_path = f'Data/{model}/{save_name}/'
    if model in ['SCTN', 'STN']:
        save_path += parse_type + '/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
    pickle.dump(obj=r2i, file=open(f'{save_path}{"r2i"}', 'wb'))
    pickle.dump(obj=train_dict, file=open(f'{save_path}{"train_data"}', 'wb'))
    pickle.dump(obj=val_dict, file=open(f'{save_path}{"val_data"}', 'wb'))
    pickle.dump(obj=test_dict, file=open(f'{save_path}{"test_data"}', 'wb'))
    print(f'Saving {model} data at: {save_path}')

# --------------------- READ IN parsed data for processing--------------- #
print("Loading parsed data.")
save_path = f'/Users/caz/Documents/bt_cleanup/Data/{data_name}_trees/'
w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
data = pickle.load(file=open(f'{save_path}{"parsed_data"}', 'rb'))
sents = [data["sents"][i] for i in range(50)]
trees = [data["trees"][i] for i in range(50)]
labels = [data["labels"][i] for i in range(50)]
sents, trees, labels = data["sents"], data["trees"], data["labels"]
# --------------------- READ IN parsed data for processing--------------- #

print("Data: ", data_name)
print("Threshold length: ", thr)
if include_SCTN: # must group similar structures and rebalancing data for batching and fair testing 
    
    print("Keeping ", number_of_structures, " syntactic forms.")
    
    print("Grouping structures ...")
    ordered_inds = group_trees(trees)
    ordered_inds = ordered_inds[:number_of_structures]
    ordered_labels = [[labels[i] for i in inds] for inds in ordered_inds]

    print("Rebalancing... ")
    n_pos, n_neg = get_data_balance(ordered_labels, None)
    count_thr = min(n_pos, n_neg)
    bal_inds, bal_labels  = rebalance(ordered_inds, ordered_labels, count_thr)

    print("Number of tress remaining: ", len(flatten_list(bal_inds)))
    print("Train/dev/test ... ")
    train_inds, val_inds, train_labels, val_labels = train_test_split(flatten_list(bal_inds), flatten_list(bal_labels), test_size=0.1, random_state=0, shuffle=True)
    train_inds, test_inds, train_labels, test_labels = train_test_split(train_inds, train_labels, test_size=0.1111, random_state=0, shuffle=True)

    train_inds, train_labels = reorder_split_inds(ordered_inds, train_inds, train_labels)
    val_inds, val_labels = reorder_split_inds(ordered_inds, val_inds, val_labels)
    test_inds, test_labels = reorder_split_inds(ordered_inds, test_inds, test_labels)

    # print test
    n_pos, n_neg = get_data_balance(train_labels, 'Train')
    n_pos, n_neg = get_data_balance(val_labels, 'Validation')
    n_pos, n_neg = get_data_balance(test_labels, 'Test')

    if reduce_train:
        print("Rebalancing TRAIN... ")
        train_inds, train_labels  = rebalance(train_inds, train_labels, int(reduce_val/2))
        n_pos, n_neg = get_data_balance(train_labels, 'train')

    # ----------------------------------------- SCTN saving ------------------------------------ #
    ordered_train_data = [[trees[t] for t in inds] for inds in train_inds]
    ordered_val_data = [[trees[t] for t in inds] for inds in val_inds]
    ordered_test_data = [[trees[t] for t in inds] for inds in test_inds]

for parse_type in ['unibox']: # , 'rule', 'height']:

    # --------------------- Grouping shared structure ----------------------- #
    if thr is not None:
        print("Trimming data ... ")
        sents, trees, labels = trim(sents, trees, labels, thr)
    
    if thr == 64:
        save_name = f'{data_name}_full'
    else:
        save_name = data_name

    # get r2i
    if parse_type == 'height':
        lenghts = [len(s) for s in sents]
        max_words = max(lenghts)
        all_rules = ['layer_'+str(i) for i in range(max_words-1)]
        indx = [i for i in range(max_words-1)]
        r2i = dict(zip(all_rules, indx))
    elif parse_type == 'unibox':
        r2i = dict({'UNIBOX':0})
    elif parse_type == 'rule':
        r2i = dict({'L': 0, 'U': 1, 'BA' : 2, 'FA' : 3, 'BC' : 4, 'FC' : 5, 'BX' : 6, 'GBC' : 7, 'GFC' : 8, 'GBX' : 9, 'LP' : 10, 'RP' : 11, 'BTR' : 12, 
        'FTR' : 13, 'CONJ' : 14, 'ADJ_CONJ' : 15, 'RPL': 16, 'RPR':17, 'NONE': 18})

    if include_SCTN:

        if parse_type == 'height':
            ordered_train_data = [[height_tree(tree) for tree in trees] for trees in ordered_train_data]
            ordered_val_data = [[height_tree(tree) for tree in trees] for trees in ordered_val_data]
            ordered_test_data = [[height_tree(tree) for tree in trees] for trees in ordered_test_data]

        train_dict_SCTN = ordered_SCTN_process(ordered_train_data, train_labels, w2i, r2i)
        val_dict_SCTN = ordered_SCTN_process(ordered_val_data, val_labels, w2i, r2i)
        test_dict_SCTN = ordered_SCTN_process(ordered_test_data, test_labels, w2i, r2i)

        print("SCTN Number of train examples: ", np.sum([len(data["labels"]) for data in train_dict_SCTN]))
        print("SCTN Number of val examples: ", np.sum([len(data["labels"]) for data in val_dict_SCTN]))
        print("SCTN Number of test examples: ", np.sum([len(data["labels"]) for data in test_dict_SCTN]))
        save_data('SCTN', save_path, train_dict_SCTN, val_dict_SCTN, test_dict_SCTN, w2i, r2i)
        # ----------------------------------------- SCTN saving ------------------------------------ #

        # ----------------------------------------- PATH/TTN saving ------------------------------------ #
        train_sents, train_trees, train_labels = sort_by_idx_after_SCTN_grouping(sents, trees, labels, train_inds)
        val_sents, val_trees, val_labels = sort_by_idx_after_SCTN_grouping(sents, trees, labels, val_inds)
        test_sents, test_trees, test_labels = sort_by_idx_after_SCTN_grouping(sents, trees, labels, test_inds)
        
        if parse_type == 'height': # dont reprocess !
            train_trees = flatten_list(ordered_train_data)
            val_trees = flatten_list(ordered_val_data)
            test_trees = flatten_list(ordered_test_data) 

        # ---------------------------- sanity check ------------------------ # 
        # t1 = 53
        # t2 = 5
        # t3 = 3
        # flatten_list(ordered_train_data)[t1].draw()
        # print(train_sents[t1])
        # train_trees[t1].draw()
        # flatten_list(ordered_val_data)[t2].draw()
        # print(val_sents[t2])
        # val_trees[t2].draw()
        # flatten_list(ordered_test_data)[t3].draw()
        # print(test_sents[t3])
        # test_trees[t3].draw()
        # ---------------------------- sanity check ------------------------ # 

    else: # don't have to deal with grouping structures and rebalancing data 
        
        print("Train/dev/test ... ")
        train_inds, val_inds, train_labels, val_labels = train_test_split(list(range(len(sents))), labels, test_size=0.1, random_state=0, shuffle=True)
        train_inds, test_inds, train_labels, test_labels = train_test_split(train_inds, train_labels, test_size=0.1111, random_state=0, shuffle=True)

        if reduce_train:
            train_inds = train_inds[:reduce_val]
            train_labels = train_labels[:reduce_val]
            print("TRAIN pos reduced: ", len(train_inds))
            print("TRAIN neg reduced: ", len(train_labels)) 

        train_sents, train_trees, train_labels = extract_data_points(sents, trees, labels, train_inds)
        val_sents, val_trees, val_labels = extract_data_points(sents, trees, labels, val_inds)
        test_sents, test_trees, test_labels = extract_data_points(sents, trees, labels, test_inds)

        if parse_type == 'height':
            train_trees = [height_tree(tree) for tree in train_trees]
            val_trees = [height_tree(tree) for tree in val_trees]
            test_trees = [height_tree(tree) for tree in test_trees]
        
    train_words, train_rules, train_offsets = tree_process(train_trees, w2i, r2i)
    val_words, val_rules, val_offsets = tree_process(val_trees, w2i, r2i)
    test_words, test_rules, test_offsets = tree_process(test_trees, w2i, r2i)

    train_dict = {"words": train_words, "rules": train_rules, "offsets": train_offsets, "labels": train_labels}
    val_dict = {"words": val_words, "rules": val_rules, "offsets": val_offsets, "labels": val_labels}
    test_dict = {"words": test_words, "rules": test_rules, "offsets": test_offsets, "labels": test_labels}
    print("STN Train examples: ", len(train_dict['words']))
    print("STN Validation examples: ", len(val_dict['words']))
    print("STN Test examples: ", len(test_dict['words']))
    save_data('STN', save_path, train_dict, val_dict, test_dict, w2i, r2i)

    if parse_type == 'unibox' or parse_type == 'height':
        train_trees_stairs, train_labels = get_diagrams_PTN(train_sents, train_labels)
        val_trees_stairs, val_labels = get_diagrams_PTN(val_sents, val_labels)
        test_trees_stairs, test_labels = get_diagrams_PTN(test_sents, test_labels)
        train_words_stairs, train_rules_stairs, train_offsets_stairs = tree_process(train_trees_stairs, w2i, r2i)
        val_words_stairs, val_rules_stairs, val_offsets_stairs = tree_process(val_trees_stairs, w2i, r2i)
        test_words_stairs, test_rules_stairs, test_offsets_stairs = tree_process(test_trees_stairs, w2i, r2i)
        train_dict_stairs = {"words": train_words_stairs, "rules": train_rules_stairs, "offsets": train_offsets_stairs, "labels": train_labels}
        val_dict_stairs = {"words": val_words_stairs, "rules": val_rules_stairs, "offsets": val_offsets_stairs,  "labels": val_labels}
        test_dict_stairs = {"words": test_words_stairs, "rules": test_rules_stairs, "offsets": test_offsets_stairs, "labels": test_labels}
        print("PATH Train examples: ", len(train_dict_stairs['words']))
        print("PATH Validation examples: ", len(val_dict_stairs['words']))
        print("PATH Test examples: ", len(test_dict_stairs['words']))
        save_data('PTN', save_path, train_dict_stairs, val_dict_stairs, test_dict_stairs, w2i, r2i)
    # ----------------------------------------- PTN / TTTN saving ------------------------------------ #

# ----------------------------------------- MERA / bTREE saving ------------------------------------ #
pad_idx = max(w2i.values())+1
train_words, train_labels = pad_CTN(train_sents, train_labels, thr, w2i, pad_idx)
val_words, val_labels = pad_CTN(val_sents, val_labels, thr, w2i, pad_idx)
test_words, test_labels = pad_CTN(test_sents, test_labels, thr, w2i, pad_idx)
train_dict_CTN = {"words": train_words, "labels": train_labels}
val_dict_CTN = {"words": val_words, "labels": val_labels}
test_dict_CTN = {"words": test_words, "labels": test_labels}
print("MERA Train examples: ", np.sum([len(data) for data in train_dict_CTN["words"]]))
print(train_dict_CTN["labels"])
print("MERA Validation examples: ", np.sum([len(data) for data in val_dict_CTN["words"]]))
print("MERA Test examples: ", np.sum([len(data) for data in test_dict_CTN["words"]]))
save_data('CTN', save_path, train_dict_CTN, val_dict_CTN, test_dict_CTN, w2i, r2i)
# TTN - also save prepadded (to 2^n) examples 
# this data must then be processed in TTN_preprocessing to decompose contraction into fast jax instructions ! 
save_data('TTN/prepad', save_path, train_dict_CTN, val_dict_CTN, test_dict_CTN, w2i, r2i)
# ----------------------------------------- MERA / bTREE saving ------------------------------------ #
