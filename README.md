# classification-with-qttn

## Models
TTN_train is all scalable models [PTN, STTN, TTN], CTN_train is as expected, CTN_slide is the sliding window version, then we got SCTN_train also.

## Processing
parse_data, process_lang_data translates into the different models relevant offsets, rules. trans_ttn_to_box_tree gets the offsets and rules for utilsing box tree efficiency for the TTN model. CTN_slide_preprocess groups the data into the sentences of windows.

## Datasets
Genome (lol lets call this protein_binding) - https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb
[James Zou, Mikael Huss, Abubakar Abid, Pejman Mohammadi, Ali Torkamani, and Amalio Telenti.
A primer on deep learning in genomics. Nature genetics, page 1, 2018]
Clickbait - 
[bhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly. Stop clickbait:
Detecting and preventing clickbaits in online news media. In Advances in Social Networks Analysis
and Mining (ASONAM), 2016 IEEE/ACM International Conference on, pages 9–16. IEEE, 2016]
RT - https://huggingface.co/datasets/rotten_tomatoes
[Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with
respect to rating scales. In Proceedings of the 43rd Annual Meeting of the Association for Computational
Linguistics (ACL’05), pages 115–124, Ann Arbor, Michigan, June 2005. Association for Computational
Linguistics]
imdb - https://ai.stanford.edu/~amaas/data/sentiment/
[MDb Baselines. https://paperswithcode.com/sota/sentiment-analysis-on-imdb. Accessed: 19-
01-23]
[Learning Word Vectors for Sentiment Analysis
Andrew L. Maas]
Note: the test we quotes is with a train of 40,000, val/test 5,000. I realise in this OG ppr they used 25,000 train, 25,000 test ! Could re-run one hyperparam value.
