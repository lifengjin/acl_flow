import torch
import random
from typing import Tuple, List
import gensim
import logging
import pickle
def batcher(sents, sorted_indices, batch_size, viterbi=False):
    if viterbi:
        viterbi_scaler = 8
    else:
        viterbi_scaler = 8
    cur_len = len(sents[sorted_indices[0]])
    # cur_batch_size = cur_len * 20
    cur_batch_size = batch_size
    cur_batch = []
    cur_indices = []
    batches = []
    batch_indices = []
    for index in sorted_indices:

        if len(sents[index]) != cur_len or len(cur_batch) == cur_batch_size:

            batches.append(torch.tensor(cur_batch).to('cuda'))
            batch_indices.append(cur_indices)
            cur_batch = []
            cur_indices = []
            cur_len = len(sents[index])

            if cur_len <= 10:
                cur_batch_size = batch_size * 120 // viterbi_scaler
            elif cur_len > 10 and cur_len <= 20:
                cur_batch_size = batch_size * 48 // viterbi_scaler
            elif cur_len > 20 and cur_len <= 30:
                cur_batch_size = batch_size * 16 // viterbi_scaler
            elif cur_len > 30 and cur_len <= 40:
                cur_batch_size = batch_size * 8 // viterbi_scaler
            elif cur_len > 40 and cur_len <= 60:
                cur_batch_size = batch_size
            elif cur_len >= 60:
                cur_batch_size = 1

        cur_batch.append(sents[index])
        cur_indices.append(index)
    else:
        if cur_batch:
            batches.append(torch.tensor(cur_batch).to('cuda'))
            batch_indices.append(cur_indices)

    logging.info( "Batcher produces {} number of sentences with {} originals".format(sum([len(x) for x in batches]),
                                                                                   len(sents)))
    return list(zip(batches, batch_indices))

def compile_embeddings(word_vecs_file, word_dict, punct_dict, embedding_type='word'):

    if word_vecs_file is None:
        return None
    else:
        try:
            if word_vecs_file.endswith('pc') or word_vecs_file.endswith('pkl'):
                with open(word_vecs_file, 'rb') as efh:
                    word_vecs = pickle.load(efh)
            else:
                word_vecs = torch.load(word_vecs_file)
            keys = word_vecs
        except:
            word_vecs = gensim.models.KeyedVectors.load(word_vecs_file)
            keys = word_vecs.vocab.keys()
    if embedding_type == 'word':
        lower_word_vecs = {}
        for key in keys:
            word = key.lower()
            if word == '-lrb-' or word == '-rrb-' or word == '-lcb-' or word == '-rcb-':
                word = word.replace('-', '')
            elif word == '#':
                word = '-pound-'
            if word in word_vecs and word != key:
                continue
            elif word not in word_vecs:
                lower_word_vecs[word] = torch.tensor(word_vecs[key])
            elif word == key:
                lower_word_vecs[word] = torch.tensor(word_vecs[key])
        useful_vecs = []
        for i in range(len(word_dict)):
            word = word_dict[i]
            if word not in lower_word_vecs and 'PUNCT' in word:
                word = punct_dict[word]
                word = word.lower()
            assert word in lower_word_vecs, "{} not in word vecs".format(word)
            useful_vecs.append(lower_word_vecs[word])
        embeddings = torch.stack(useful_vecs, dim=0).float()
        logging.info('Embedding shapes: {}'.format(str(embeddings.shape)))
    elif embedding_type == 'context':
        return word_vecs
    return embeddings
