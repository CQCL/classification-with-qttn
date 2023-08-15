# Sequence Processing with Quantum Tensor Networks

Resources for "Sequence Processing with Quantum Tensor Networks".

## Requirements

Install the requirements with `pip install -r requirements.txt`.

The tensor network contraction is performed using a combination of `discopy`
and `jax`.

First, follow [these instructions](https://github.com/google/jax#installation) to install JAX with the relevant accelerator support.

Documentation of discopy can be found [here](https://docs.discopy.org/en/legacy/).

## Datasets

| Name              | Paper                                                                                                                                          |                                                                                                                                  |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `protein-binding` | [A primer on deep learning in genomics]( https://www.nature.com/articles/s41588-018-0295-5 )                                                   | [link](https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb) |
| `clickbait`       | [Stop Clickbait: Detecting and preventing clickbaits in online news media]( https://ieeexplore.ieee.org/document/7752207 )                     | [link](https://github.com/bhargaviparanjape/clickbait)                                                                           |
| `rotten-tomatoes` | [Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.](https://aclanthology.org/P05-1015/) | [link](https://huggingface.co/datasets/rotten-tomatoes)                                                                          |
| `imdb`            | [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/)                                                             | [link](https://ai.stanford.edu/~amaas/data/sentiment/)                                                                           |

The `imdb` dataset is not included in the repository. It can be downloaded from the link above, then preprocessed with the preprocessing scripts.

## Processing

* `BIO_preprocess.py` preprocesses the `protein-binding` dataset. This takes the genetic strings and translates them to index representation. For TTN and CTN it also pads to appropriate powers of 2.
* `TTN_preprocess.py` takes in sequences and acquires the appropriate offsets for fast contraction in `TTN_train.py`.
* `CTN_slide_preprocess.py` takes in sequences and saves them as lists of subsequences of the desired window size.
* `NLP_parsing.py` parses language data into a syntactic tree using Lambeq's CCG parser.
* `NLP_preprocess.py` translates trees into sequential instructions for syntactic models (STN, SCTN). For the other tree models (CTN, TTN) it also pads sequences in groups of the nearest power of two. There is also the option to cut the data keeping the X most common syntactic structures while maintaining dataset balance. This is necessary for SCTN.

## Models

* `TTN_train.py` is all scalable models [`PTN`, `STN`, `TTN`]. Example datasets included are: `protein-binding`, `rotten-tomatoes`, `clickbait`.
* `CTN_train.py` is the `CTN` model. Example datasets included are `protein-binding` and reduced `clickbait`.
* `SCTN_train.py` is the `SCTN` model. Example datasets included are reduced `clickbait`.
* `CTN_slide.py` is the sliding window option or the `CTNs` model. Example datasets included are `protein-binding`, `rotten-tomatoes`.

JIT compilation for `CTN_train.py` and `SCTN_train.py` takes time in the beginning, but speeds up the training process. The first epoch will be slow, but subsequent epochs will be faster. JIT compiltation can be disabled in the config file.
