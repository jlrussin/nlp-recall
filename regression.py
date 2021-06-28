import argparse
import pickle
import numpy as np
import csv
import gensim.downloader as api
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

from preprocess import *
from metrics import *

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
parser.add_argument('--k_folds', type=int, default=6, 
                    help='Number of folds for k-fold cross-validation')
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

    # Get total number of word tokens and word types
    print("Computing total number of word tokens and types")
    n_tokens = []
    n_types = []
    for s_id in subject_ids:
        n_tok = total_word_tokens(transcripts[s_id])
        n_tokens.append(n_tok)
        n_typ = total_word_types(transcripts[s_id])
        n_types.append(n_typ)
    n_tokens = np.array(n_tokens)
    n_types= np.array(n_types)
    print("Done.")

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
    print("Done.")
    
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
    ratings_dict = {}
    with open(args.human_ratings_path, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            ratings_dict[row[0]] = float(row[1])
    ratings = [ratings_dict[s_id] for s_id in subject_ids]
    ratings = np.array(ratings)
    print("Done.")

    # Regression on all data - no cross-validation
    print("Performing regression on all data")

    # Total number of word tokens
    X_to = n_tokens.reshape(-1,1)
    reg_to = LinearRegression().fit(X_to,ratings)
    r_to = reg_to.score(X_to,ratings)
    print("R^2 for total word tokens model: {}".format(r_to))

    # Total number of word types
    X_ty = n_types.reshape(-1,1)
    reg_ty = LinearRegression().fit(X_ty,ratings)
    r_ty = reg_ty.score(X_ty,ratings)
    print("R^2 for total word types model: {}".format(r_ty))

    # Number of word types in common
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
        reg = LinearRegression().fit(X,ratings)
        r = reg.score(X,ratings)
        print("R^2 for model with common word types and WMD: {}".format(r))

    # Cross-validation
    print("Performing cross-validation with {} folds".format(args.k_folds))

    # Total number of word tokens
    cv_reg_to = LinearRegression()
    cv_results_to = cross_validate(cv_reg_to, X_to, ratings, cv=args.k_folds)
    cv_r_to = np.mean(cv_results_to['test_score'])
    print("Cross-validated R^2 for total word tokens model: {}".format(cv_r_to))

    # Total number of word tokens
    cv_reg_ty = LinearRegression()
    cv_results_ty = cross_validate(cv_reg_ty, X_ty, ratings, cv=args.k_folds)
    cv_r_ty = np.mean(cv_results_ty['test_score'])
    print("Cross-validated R^2 for total word types model: {}".format(cv_r_ty))

    cv_reg_c = LinearRegression()
    cv_results_c = cross_validate(cv_reg_c, X_c, ratings, cv=args.k_folds)
    cv_r_c = np.mean(cv_results_c['test_score'])
    print("Cross-validated R^2 for common word types model: {}".format(cv_r_c))

    if args.use_wmd:
        # Cross-validated regression with WMD only
        cv_reg_d = LinearRegression()
        cv_results_d = cross_validate(cv_reg_d, X_d, ratings, cv=args.k_folds)
        cv_r_d = np.mean(cv_results_d['test_score'])
        print("Cross-validated R^2 for WMD model: {}".format(cv_r_d))
        # Regression with common word types and WMD
        cv_reg = LinearRegression()
        cv_results = cross_validate(cv_reg, X, ratings, cv=args.k_folds)
        cv_r = np.mean(cv_results['test_score'])
        print("Cross-validated R^2 for C+D model: {}".format(cv_r))

    # Save model
    if args.save_model_fn is not None:
        print("Saving regression model to {}".format(args.save_model_fn))
        with open(args.save_model_fn, 'wb') as f:
            if args.use_wmd:
                pickle.dump(reg, f)
            else:
                pickle.dump(reg_c, f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

