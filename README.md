# IRMSFF
## Requirements
python 3.8

pytorch 1.11.0

nltk 3.9.1

pandas 2.0.3

scikit-learn 1.3.2

transformers 4.46.3

nlg-eval 2.4.1
## Get start
First, obtain the TLC and HDC datasets from the locations mentioned in the paper, store them in the `data` folder, and download the UniXcoder encoder checkpoint-related model parameters from Hugging Face to an appropriate location. Run `split.py` to obtain the camel-case segmented code data for Jaccard similarity evaluation.
## Retrieve phase
First, run `databaseCreate.py` to create the vector database, then run `retrieve_another.py` to perform retrieval, and evaluate the quality of similar codes by reusing the corresponding comments.
## First stage
Modify the paths for the code dataset and the UniXcoder model to your own paths, then run `run.py` to train the model.
## Second stage
Modify the paths for the code dataset, the UniXcoder model, and the model saved from the first stage to your own paths, then run `run.py` to train the model.
