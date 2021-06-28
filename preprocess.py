import os 
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess(path):
    """
    Load transcripts from path and preprocess them. 
    If path is a directory, all files in the directory will be preprocessed.
    If path is a file, the file will be preprocessed. 
    Preprocessing steps:
      1. Load transcript(s)
      2. Tokenize
      3. Remove stop words (e.g., 'the', 'I', 'that', etc.)
      4. Remove punctuation
      5. Make words lowercase
    Returns a dictionary:
      -keys = filenames
      -values = lists of word tokens in documents after preprocessing
    """
    # Determine if path is to a file or a directory
    assert os.path.isfile(path) or os.path.isdir(path)
    if os.path.isfile(path):
        filenames = [os.path.basename(path)]
        dir_name = os.path.dirname(path)
    elif os.path.isdir(path):
        filenames = [fn for fn in os.listdir(path)]
        dir_name = path

    # Preprocess documents
    transcripts = {}
    for fn in filenames:
        # Read file
        subject_id = fn.split('.')[0] # remove .txt extension
        path_i = os.path.join(dir_name, fn)
        with open(path_i, 'r') as f:
            transcript = f.read()[1:] # remove '\ufeff'

        # Tokenize
        tokens = word_tokenize(transcript)

        # Remove stop words, punctuation, and make lowercase
        download('stopwords', quiet=True) # list of stop words from nltk.corpus
        stop_words = stopwords.words('english')
        preprocessed = []
        for w in tokens:
            if w.lower() not in stop_words and w.isalnum():
                preprocessed.append(w.lower())
        transcripts[subject_id] = preprocessed
    
    return transcripts