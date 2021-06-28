def total_word_tokens(doc):
    return len(doc)

def total_word_types(doc):
    return len(set(doc))

def common_words(doc1, doc2):
    """ 
    Compute the number of words (types - not tokens) in common
    doc1: list of word tokens (should be after preprocessing)
    doc2: list of word tokens (should be after preprocessing)
    returns an int indicating number of common word types
    """
    types1 = set(doc1)
    types2 = set(doc2)
    common_types = types1.intersection(types2)
    n_common = len(common_types)
    return n_common


def word_movers_distance(doc1, doc2, embedding):
    """
    Compute Word Mover's Distance (WMD) between two documents
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
    WMD is a measure of similarity between two documents using word embeddings
    doc1: list of word tokens (should be after preprocessing)
    doc2: list of word tokens (should be after preprocessing)
    returns a float indicating WMD(doc1, doc2)
    """
    
    # Compute WMD
    distance = embedding.wmdistance(doc1, doc2)

    return distance
