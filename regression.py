import argparse
import pickle
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
parser.add_argument('--human_ratings_path', 
                    default='../data/human_ratings/soft1.csv',
                    help='Path to file to compare transcripts against')
parser.add_argument('--use_wmd', action='store_true',
                    help='Use WMD in regression')
parser.add_argument('--embed_model', default='word2vec', 
                    choices=['word2vec', 'glove'],
                    help='Type of word embeddings to use for WMD')
parser.add_argument('--save_model_fn', default=None, 
                    help='Filename for saving regression model')

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

    # Import human ratings
    print("Importing human ratings from {}".format(args.human_ratings_path))
    ratings_raw = np.genfromtxt(args.human_ratings_path, delimiter=',')
    ratings_ids = [str(int(s_id)) for s_id in ratings_raw[:,0]]
    ratings_data = [r for r in ratings_raw[:,1]]
    ratings_dict = {s_id:r for s_id, r in zip(ratings_ids, ratings_data)}
    ratings = [ratings_dict[s_id] for s_id in subject_ids]
    ratings = np.array(ratings)

    # Regression with common word types only
    X_c = n_common.reshape(-1,1)
    reg_c = LinearRegression().fit(X_c,ratings)
    r_c = reg_c.score(X_c,ratings)
    print("R^2 for common word types model: {}".format(r_c))

    if args.use_wmd:
        X = np.array([n_common, wmd]).T
        # Regression with WMD only
        X_d = X[:,1].reshape(-1,1)
        reg_d = LinearRegression().fit(X_d,ratings)
        r_d = reg_d.score(X_d,ratings)
        print("R^2 for Word Mover's Distance model: {}".format(r_d))
        # Regression with common word types and WMD
        reg_both = LinearRegression().fit(X,ratings)
        r_both = reg_both.score(X,ratings)
        print("R^2 for model with common word types and WMD: {}".format(r_both))

    # Save model
    if args.save_model_fn is not None:
        print("Saving regression model to {}".format(args.save_model_fn))
        with open(args.save_model_fn, 'wb') as f:
            if args.use_wmd:
                pickle.dump(reg_both, f)
            else:
                pickle.dump(reg_c, f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

