from curses import window
import numpy as np
from pathlib import Path
import pickle
from nltk.stem import WordNetLemmatizer
from lambeq import TreeReader
from lambeq import TreeReaderMode
import spacy
from discopy.rigid import Box, Id
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd


def tokenize(lines):
    print("Tokenizing ...")
    tok_sents = []
    for line in lines:
        line = line.replace(". . .", "...")
        line = list(tokenizer(str(line)))
        line = [str(w) for w in line]
        tok_sents.append(line)
    return tok_sents

def compute_w2i(sentences, min_freq, lemmatise=False, max_vocab=None):
        
        # Initialize a dictionary of word frequency
        word_freq = {}
        for sent in sentences: 
            for word in sent:
                word_freq[str(word)] = word_freq.get(word, 0) + 1 # update freq

        # word2index dictionary ordered by occurrence
        w2f = {word: freq for (word, freq) in word_freq.items() if freq>=min_freq}
        unk_words = [word for (word, freq) in word_freq.items() if freq<min_freq]

        # or by max_vocab !
        if max_vocab is not None:
            word_freq = {word: freq for word, freq in sorted(word_freq.items(), key=lambda item: item[1])[::-1]}
            word_freq = {word: freq for idx, (word, freq) in enumerate(word_freq.items()) if idx<max_vocab}

        w2i = {w: idx for (idx, w) in enumerate(w2f.keys())}

        # create single index for unk 
        unk_index = len(w2i)
        for word in unk_words:
            w2i[word]=unk_index # unk index

        # assign same index for all lemmatizations 
        if lemmatise is True:
            lemmatizer = WordNetLemmatizer()
            Lw2i = dict()
            for word, index in w2i.items():
                lemma = lemmatizer.lemmatize(word.lower()) 
                if lemma in Lw2i.keys():
                    Lw2i[word] = Lw2i[lemma]
                else:
                    Lw2i[lemma] = index
                    Lw2i[word] = index
            return Lw2i
        else:
            return w2i

data_name = 'gen'
print("Data: ", data_name)

tokenizer = spacy.load("en_core_web_sm")
tokenizer.disable_pipes(
    ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
)
tokenizer.enable_pipe("senter")

if data_name == 'RT':
    
    with open('Data/pos.txt') as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open('Data/neg.txt') as f:
        neg_lines = [line.strip() for line in f if line.strip()]
    
    pos_sents = tokenize(pos_lines)
    neg_sents = tokenize(neg_lines)

    sents = np.concatenate((pos_sents, neg_sents))
    labels = np.concatenate(([[1,0]]*len(pos_sents),[[0,1]]*len(neg_sents)))

    min_freq = 1
    w2i = compute_w2i(sents, min_freq)

elif data_name == 'IMDb': # 50,000 reviews
    
    df = pd.read_csv('IMDB_data.csv') 
    labels = df['sentiment']
    sents = df['review']

    pos_lines = [sent for (sent,label) in zip(sents, labels) if label == 'positive']
    neg_lines = [sent for (sent,label) in zip(sents, labels) if label == 'negative' ]

    pos_sents = tokenize(pos_lines)
    neg_sents = tokenize(neg_lines)

    sents = np.concatenate((pos_sents, neg_sents))
    labels = np.concatenate(([[1,0]]*len(pos_sents),[[0,1]]*len(neg_sents)))
    
    min_freq = 1
    w2i = compute_w2i(sents, min_freq)

elif data_name == 'genome':

    with open('Data/genome_seqs.txt') as f:
        all_lines = [line.strip() for line in f]

    with open('Data/genome_labels.txt') as f:
        all_labels = [line.strip() for line in f]

    pos_sents = []
    neg_sents = []
    for line, label in zip(all_lines, all_labels):
        if label == '0':
            pos_sents.append([l for l in line])
        elif label == '1':
            neg_sents.append([l for l in line])

    pos_sents = [p[:10] for p in pos_sents]
    neg_sents = [p[:10] for p in neg_sents]

    sents = np.concatenate((pos_sents, neg_sents))
    labels = np.concatenate(([[1,0]]*len(pos_sents),[[0,1]]*len(neg_sents)))
    min_freq = 1
    w2i = compute_w2i(sents, min_freq)

elif data_name == 'gen':
    window = 10
    save_path = f'Results/TTN_bio_gen_weak_sim_mem_window_{window}/CTN_slide_test/'
    w2i = pickle.load(file=open(f'{save_path}{"w2i"}', 'rb'))
    test_sents = pickle.load(file=open(f'{save_path}{"gen_samples"}', 'rb'))
    test_labels = pickle.load(file=open(f'{save_path}{"labels"}', 'rb'))
    print(test_sents[0])
    

# transfer to w2i 
window_size = 4
pad_idx = max(w2i.values())+1

# split sent, labels into train, val, test
train_sents, val_sents, train_labels, val_labels = train_test_split(sents, labels, test_size=0.1, random_state=0, shuffle=True)
train_sents, test_sents, train_labels, test_labels = train_test_split(train_sents, train_labels, test_size=0.1111, random_state=0, shuffle=True)

train_split_sents = []
train_counts = []
for sent in train_sents:
    count = 0
    temp = []
    if len(sent) <= window_size:
        temp.append(np.concatenate(([w2i[s] for s in sent], pad_idx*np.ones(window_size-len(sent)))))
        count+=1
    else:
        for i in range(len(sent)-window_size):
            temp.append(np.array([w2i[s] for s in sent[i:i+window_size]]))
            count+=1
    train_split_sents.append(temp)
    train_counts.append(count)

val_split_sents = []
val_counts = []
for sent in val_sents:
    count = 0
    temp = []
    if len(sent) <= window_size:
        temp.append(np.concatenate(([w2i[s] for s in sent], pad_idx*np.ones(window_size-len(sent)))))
        count+=1
    else:
        for i in range(len(sent)-window_size):
            temp.append(np.array([w2i[s] for s in sent[i:i+window_size]]))
            count+=1
    val_split_sents.append(temp)
    val_counts.append(count)

test_split_sents = []
test_counts = []
for sent in test_sents:
    count = 0
    temp = []
    if len(sent) <= window_size:
        temp.append(np.concatenate(([w2i[s] for s in sent], pad_idx*np.ones(window_size-len(sent)))))
        count+=1
    else:
        for i in range(len(sent)-window_size):
            temp.append(np.array([w2i[s] for s in sent[i:i+window_size]]))
            count+=1
    test_split_sents.append(temp)
    test_counts.append(count)

train_data = {"words": train_split_sents, "counts": train_counts, "labels": train_labels}
val_data = {"words": val_split_sents, "counts": val_counts, "labels": val_labels}
test_data = {"words": test_split_sents, "counts": test_counts, "labels": test_labels}

print("Saving parsed data.")
save_path = f'Data/CTN_SLIDE_{window_size}/{data_name}/'
print(save_path)
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=train_data, file=open(f'{save_path}{"train_data"}', 'wb'))
pickle.dump(obj=val_data, file=open(f'{save_path}{"val_data"}', 'wb'))
pickle.dump(obj=test_data, file=open(f'{save_path}{"test_data"}', 'wb'))
