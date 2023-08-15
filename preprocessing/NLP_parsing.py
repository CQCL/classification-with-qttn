import numpy as np
from pathlib import Path
import pickle
from nltk.stem import WordNetLemmatizer
from lambeq import TreeReader
from lambeq import TreeReaderMode
import spacy
from discopy.rigid import Box, Id
from tqdm import tqdm

data = 'rotten-tomatoes'
print("Data: ", data)
assert data in ['clickbait', 'rotten-tomatoes']

parser = TreeReader(mode = TreeReaderMode.RULE_ONLY)

# ----------------------- read in, tokenize ------------------------ #
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

def rename_box(layer, name):
    left, box, right = layer
    return Id(left) @ Box(name, box.dom, box.cod) @ Id(right)

def height_tree(sentence):
    diagram = parser.sentence2diagram(sentence, tokenised=True)
    cuts = diagram.foliation().boxes
    new_diagram = cuts[0]
    for i, cut in enumerate(cuts[1:]):
        new_diagram = new_diagram.then(
            *[rename_box(layer, f'layer_{i}') for layer in cut.layers])
    return new_diagram

def get_diagrams(texts, labels):
    print("Parsing sentences ... ")
    parsed_sents = []
    parsed_trees = []
    parsed_labels = []
    for label, review in tqdm(zip(labels, texts)): 
        try:
            d = parser.sentence2diagram(review, tokenised=True)
            parsed_sents.append(review)
            parsed_trees.append(d)
            parsed_labels.append(label)
        except:
            continue
    return parsed_sents, parsed_trees, parsed_labels

if data == 'clickbait':
    with open('Data/clickbait.txt') as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open('Data/non_clickbait.txt') as f:
        neg_lines = [line.strip() for line in f if line.strip()]

elif data == 'rotten-tomatoes':
    with open('Data/pos.txt') as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open('Data/neg.txt') as f:
        neg_lines = [line.strip() for line in f if line.strip()]

tokenizer = spacy.load("en_core_web_sm")
tokenizer.disable_pipes(
    ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
)
tokenizer.enable_pipe("senter")

pos_sents = tokenize(pos_lines)
neg_sents = tokenize(neg_lines)
# ----------------------- read in, tokenize ------------------------ #

sents = np.concatenate((pos_sents, neg_sents))
labels = np.concatenate(([[1,0]]*len(pos_sents),[[0,1]]*len(neg_sents)))
sents, trees, labels = get_diagrams(sents, labels)

parsed_data = dict({"sents": sents, "trees": trees, "labels": labels})

print("Forming w2i, r2i")
min_freq = 1
w2i = compute_w2i(sents, min_freq)

print("Saving parsed data.")
save_path = f'Data/{data}_trees/'
Path(save_path).mkdir(parents=True, exist_ok=True)
pickle.dump(obj=w2i, file=open(f'{save_path}{"w2i"}', 'wb'))
pickle.dump(obj=parsed_data, file=open(f'{save_path}{"parsed_data"}', 'wb'))


