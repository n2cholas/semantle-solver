import gzip
import os
import pickle
import shutil

import gensim.models.keyedvectors as word2vec
import numpy as np
import wget

RAW_VECTORS_PATH = 'GoogleNews-vectors-negative300.bin'

if os.path.exists('cached.pkl'):
    with open('cached.pkl', 'rb') as f:
        matrix, idx2word, word2idx = pickle.load(f)
else:
    # Download and gunzip file
    gzip_filename = wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")  # yapf: disable  # noqa: E501
    with gzip.open(gzip_filename, 'rb') as f_in:
        # need to write to disk since gensim only supports filenames not files
        with open(RAW_VECTORS_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # extract raw word2vec matrix and mapping
    keyed_vectors = word2vec.KeyedVectors.load_word2vec_format(
        RAW_VECTORS_PATH, binary=True)
    keyed_vectors.most_similar(['ocean'], topn=1)  # generates matrix
    idx2word = {
        i: w
        for i, w in enumerate(keyed_vectors.index_to_key)
        # keep only lower case single words
        if w.isalpha() and w.lower() == w
    }
    matrix = keyed_vectors.get_normed_vectors()[list(idx2word.keys())]
    del keyed_vectors
    matrix *= 100  # do this here instead of earlier to save memory
    idx2word = list(idx2word.values())
    word2idx = {w: i for i, w in enumerate(idx2word)}

    # cache objects + clean up for future runs
    with open('cached.pkl', 'wb') as f:
        pickle.dump((matrix, idx2word, word2idx), f)
    os.remove(RAW_VECTORS_PATH)
    os.remove(gzip_filename)

cur_word = input('Initial Guess: ')
cur_score = float(input('Score (type "done" if done): ')) * 100
possibilities = np.full(matrix.shape[0], True)
while True:
    cos_similarities = matrix.dot(matrix[word2idx[cur_word]])
    possibilities &= np.round(cos_similarities) == cur_score
    cur_word = idx2word[possibilities.argmax()]
    print('Next Guess: ', cur_word)
    cur_score = input(f'{cur_word} score (type "done" if done): ')
    try:
        cur_score = float(cur_score) * 100
    except ValueError:
        break
print('Congrats!')
