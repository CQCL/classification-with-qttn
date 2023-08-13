# Sequence Processing with Quantum Tensor Networks

Resources for "Sequence Processing with Quantum Tensor Networks".

## Requirements

Install the requirements with `pip install -r requirements.txt`.

## Datasets

| Name              | Paper                                                                                                                                          |                                                                                                                                  |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `genome`          | [A primer on deep learning in genomics]( https://www.nature.com/articles/s41588-018-0295-5 )                                                   | [link](https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb) |
| `clickbait`       | [Stop Clickbait: Detecting and preventing clickbaits in online news media]( https://ieeexplore.ieee.org/document/7752207 )                     |                                                                                                                                  |
| `rotten_tomatoes` | [Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.](https://aclanthology.org/P05-1015/) | [link](https://huggingface.co/datasets/rotten_tomatoes)                                                                          |
| `imdb`            | [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/)                                                             | [link](https://ai.stanford.edu/~amaas/data/sentiment/)                                                                           |

**Note**: the test we quotes is with a train of 40,000, val/test 5,000. I realise in this OG ppr they used 25,000 train, 25,000 test ! Could re-run one hyperparam value.


## Processing

* `parse_data.py`, `process_lang_data.py` translates the parse trees into the different models relevant offsets, rules.
* `trans_ttn_to_box_tree.py` gets the offsets and rules for utilsing box tree efficiency for the TTN model. 
* `CTN_slide_preprocess.py` groups the data into the sentences of windows.

## Models

* `TTN_train.py` is all scalable models [`PTN`, `STN`, `TTN`]
* `CTN_train.py` is as expected
* `CTN_slide.py` is the sliding window version, then we got SCTN_train also.