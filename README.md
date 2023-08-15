# Sequence Processing with Quantum Tensor Networks

Resources for "Sequence Processing with Quantum Tensor Networks".

## Requirements

Install the requirements with `pip install -r requirements.txt`.

## Datasets

| Name              | Paper                                                                                                                                          |                                                                                                                                  |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `Protein-binding`          | [A primer on deep learning in genomics]( https://www.nature.com/articles/s41588-018-0295-5 )                                                   | [link](https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb) |
| `clickbait`       | [Stop Clickbait: Detecting and preventing clickbaits in online news media]( https://ieeexplore.ieee.org/document/7752207 )                     |                                                                                                                                  |
| `rotten_tomatoes` | [Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.](https://aclanthology.org/P05-1015/) | [link](https://huggingface.co/datasets/rotten_tomatoes)                                                                          |
| `imdb`            | [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/)                                                             | [link](https://ai.stanford.edu/~amaas/data/sentiment/)                                                                           |

## Processing

* `BIO_preprocess.py` preprocesses the Protein-binding dataset. This takes the genetic strings and translates them to index representation. For TTN and CTN it also pads to appropriate powers of 2.
* `TTN_preprocess.py` takes in sequences and acquires the appropriate offsets for fast contraction in `TTN_train` .
* `CTN_slide_preprocess.py` takes in sequences and saves them as lists of subsequences of the desired window size.
* `NLP_parsing.py` parses language data into a syntactic tree using Lambeq's CCG parser.
* `NLP_preproces.py` translates trees into sequential instructions for syntactic models (STN, SCTN). For the other tree models (CTN, TTN) it also pads sequences in groups of the nearest power of two. There is also the option to cut the data keeping the X most common syntactic structures while maintaining dataset balance. This is necessary for SCTN.

## Models

* `TTN_train.py` is all scalable models [`PTN`, `STN`, `TTN`]. Example datasets included are: `Protein-binding`,  `rotten_tomatoes`, `clickbait`.
* `CTN_train.py` is the `CTN` model. Example datasets included are `Protein-binding` and reduced `clickbait`.
* `SCTN_train.py` is the `SCTN` model. Example datasets included are reduced `clickbait`.
* `CTN_slide.py` is the sliding window option or the `CTNs` model. Example datasets included are `Protein-binding`, `rotten_tomatoes`, `IMDb`.
