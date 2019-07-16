import torch
from collections import Counter, defaultdict

def is_punc(key_value_pair):
    if 'PUNCT' in key_value_pair[1]:
        return True
    return False

def find_freqs(ev_seqs, word_dict):
    punc_keys_indices = filter(is_punc, word_dict.items())
    punc_indices = set([x[0] for x in punc_keys_indices])
    punc_freqs = defaultdict(int)
    word_freqs = defaultdict(int)
    for sent in ev_seqs:
        for word in sent:
            if word in punc_indices:
                punc_freqs[word] += 1
            else:
                word_freqs[word] += 1
    word_freqs = Counter(word_freqs)
    return punc_freqs, word_freqs

def matrix_initialization(grammar_matrix, p0, ev_seqs, word_dict, branching='right', separate_frequent_word=True,
                                          filter_punc=True):
    # 0 -> 1 0 or 0 -> 0 1 as the most important rule
    # 1 -> X where X is the most frequent word at the left or right fringe
    # 2 - 7 are the most frequent tokens with punc collapsed into 2
    # all probs are in log space
    num_cats = grammar_matrix.shape[0]
    nonterm_rules_num = num_cats ** 2
    RIGHT_BRANCHING_NONTERM_RULE_INDEX = (0, num_cats)
    LEFT_BRANCHING_NONTERM_RULE_INDEX = (0, 1)
    punc_freqs, word_freqs = find_freqs(ev_seqs, word_dict)
    punc_freqs_dist = {key:punc_freqs[key]/sum(punc_freqs.values()) for key in punc_freqs}
    top5_words = word_freqs.most_common(5)
    top5_words_indices = [ x[0] for x in top5_words]
    max_prob = torch.max(grammar_matrix)

    if branching == 'right':
        p0_max = torch.max(p0)
        p0[0] = (p0_max.exp()*2).log()
        grammar_matrix[RIGHT_BRANCHING_NONTERM_RULE_INDEX] = (max_prob.exp()*2).log()
        right_fringe_words = [sent[-1] for sent in ev_seqs]
        frequent_fringe_token, _ = Counter(right_fringe_words).most_common(1)[0]
    elif branching == 'left':
        p0_max = torch.max(p0)
        p0[0] = (p0_max.exp()*2).log()
        grammar_matrix[LEFT_BRANCHING_NONTERM_RULE_INDEX] = (max_prob.exp()*2).log()
        left_fringe_words = [sent[0] for sent in ev_seqs]
        frequent_fringe_token, _ = Counter(left_fringe_words).most_common(1)[0]
    else:
        raise ValueError('unknown branching.')

    if frequent_fringe_token not in punc_freqs:
        grammar_matrix[1, nonterm_rules_num+frequent_fringe_token] = (max_prob.exp()*2).log()
        for punc_index in punc_freqs:
            punc_prob = punc_freqs_dist[punc_index]
            grammar_matrix[2, nonterm_rules_num+punc_index] = (max_prob.exp()*2*punc_prob/max(punc_freqs_dist.values(
                                                                                                    ))).log()
        starting_row = 3
        for word_index in top5_words_indices:
            if word_index == frequent_fringe_token:
                continue
            else:
                grammar_matrix[starting_row, nonterm_rules_num+word_index] = (max_prob.exp()*2).log()
                starting_row += 1
    else:
        for punc_index in punc_freqs:
            punc_prob = punc_freqs_dist[punc_index]
            grammar_matrix[1, nonterm_rules_num+punc_index] = (max_prob.exp()*2*punc_prob/max(punc_freqs_dist.values(
                                                                                                    ))).log()
        starting_row = 2
        for word_index in top5_words_indices:
            if word_index == frequent_fringe_token:
                continue
            else:
                grammar_matrix[starting_row, nonterm_rules_num+word_index] = (max_prob.exp()*2).log()
                starting_row += 1
    return grammar_matrix, p0