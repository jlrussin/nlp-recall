import argparse
import pickle
import csv
import numpy as np
import gensim.downloader as api
from sklearn.linear_model import LinearRegression

from preprocess import preprocess
from metrics import common_words, word_movers_distance

parser = argparse.ArgumentParser()
parser.add_argument('--transcripts_dir', 
                    default='../data/transcripts/recall_1',
                    help='Path to directory containing transcripts')
parser.add_argument('--summary_path', 
                    default='../data/ground_truth_summary.txt',
                    help='Path to file to compare transcripts against')
parser.add_argument('--use_wmd', action='store_true',
                    help='Use WMD in regression')
parser.add_argument('--embed_model', default='word2vec', 
                    choices=['word2vec', 'glove'],
                    help='Type of word embeddings to use for WMD')
parser.add_argument('--load_model_fn', default='regression.P', 
                    help='Filename for loading regression model')
parser.add_argument('--save_predictions_fn', default='predictions.csv', 
                    help='Filename for saving predicted ratings')

def main(args):
    # Preprocess transcripts
    print("Loading transcript(s) from {}".format(args.transcripts_dir))
    transcripts = preprocess(args.transcripts_dir)

    # Preprocess summary
    print("Loading ground truth summary from {}".format(args.summary_path))
    summary_dict = preprocess(args.summary_path)
    summary = list(summary_dict.values())[0] # only 1 element in dict

    # Subject IDs correspond to filenames in transcripts_dir
    subject_ids = [s_id for s_id in transcripts.keys()]

    # Load embeddings
    if args.use_wmd:
        print("Downloading/loading word embeddings: {}".format(args.embed_model))
        if args.embed_model == 'word2vec':
            embedding = api.load('word2vec-google-news-300')
        elif args.embed_model == 'glove':
            embedding = api.load('glove-twitter-25')

    # Get common word types
    print("Computing common word types")
    n_common = []
    for s_id in subject_ids:
        # Documents to compare
        doc1 = transcripts[s_id]
        doc2 = summary 

        # Number of word types in common
        n = common_words(doc1, doc2)
        n_common.append(n)
    n_common = np.array(n_common)
    print("Finished computing common word types")
    
    # Get Word Mover's Distance
    if args.use_wmd:
        print("Computing Word Mover's Distance")
        wmd = []    
        for s_id in subject_ids:
            # Documents to compare
            doc1 = transcripts[s_id]
            doc2 = summary 

            # WMD
            d = word_movers_distance(doc1, doc2, embedding)
            wmd.append(d)

            print("Finished subject ID: {}".format(s_id))
        wmd = np.array(wmd)

    # Load regression model
    print("Loading regression model from {}".format(args.load_model_fn))
    with open(args.load_model_fn, 'rb') as f:
        reg = pickle.load(f)

    # Compute predictions from regression model
    print("Computing predicted ratings from regression model")
    if args.use_wmd:
        X = np.array([n_common, wmd]).T
    else:
        X = n_common.reshape(-1,1)
    y_hat = reg.predict(X)

    # Save predictions
    print("Saving predicted ratings to {}".format(args.save_predictions_fn))
    with open(args.save_predictions_fn, 'w') as f:
        fieldnames = ['Subject IDs', 'Predicted Ratings']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for s_id, yhat_i in zip(subject_ids, y_hat):
            writer.writerow({'Subject IDs': s_id, 'Predicted Ratings': yhat_i})

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

