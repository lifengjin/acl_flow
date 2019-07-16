import nltk
import numpy as np
import os
from copy import deepcopy
import logging
import math
from typing import List
from .treenode import Node, nodes_to_tree, calc_branching_score
"""
this file is for translating sequences of states to pcfg counts and back to uhhmm counts
the main function is translate_through_pcfg
"""

# pass in the file name of the ints file
# init with a strategy that is defined in pcfg init strategies script
def init_with_strategy(ints_seqs, strategy, abp_domain_size, gold_pos_dict = None):
    trees = []
    assert isinstance(ints_seqs, list)
    for line in ints_seqs:
        this_tree = strategy(line, abp_domain_size, gold_pos_dict=gold_pos_dict)
        trees.append(this_tree)
    pcfg_probs_and_counts = extract_counts_from_trees(trees, abp_domain_size)
    return pcfg_probs_and_counts

right_branching_tendency = 0.0

# input is a list of trees instead of tuples of state sequences
def extract_counts_from_trees(trees : List[nltk.tree.Tree], K : int):
    pcfg = {}
    p0_counts = {}
    l_branches = 0
    r_branches = 0
    # PCFGMedic.cpr(trees, K)
    for tree_index, tree in enumerate(trees):
        # rules = _extract_counts_single_tree(tree, nonterms)
        # print(tree_index, tree)
        l_branch, r_branch = calc_branching_score(tree)
        l_branches += l_branch
        r_branches += r_branch
        top_node = tree.label()
        if top_node not in p0_counts:
            p0_counts[top_node] = 0
        p0_counts[top_node] += 1
        pcfg_rules = tree.productions()
        for rule in pcfg_rules:
            if rule.lhs() not in pcfg:
                pcfg[rule.lhs()] = {}
            pcfg[rule.lhs()][rule.rhs()] = pcfg[rule.lhs()].get(rule.rhs(), 0) + 1
    pcfg_counts = deepcopy(pcfg)
    for lhs in pcfg:
        total = sum(pcfg[lhs].values())
        for rhs in pcfg[lhs]:
            pcfg[lhs][rhs] /= total
    global right_branching_tendency
    right_branching_tendency = r_branches / (l_branches + r_branches)
    logging.info('Right branching tendency score is {:.4f}'.format(right_branching_tendency))
    if right_branching_tendency > 0.95:
        logging.warning('VERY RIGHT BRANCHING GRAMMAR DETECTED!!')
    return pcfg, pcfg_counts, p0_counts

def _build_nonterminals(abp_domain_size):
    # we build the 0 nonterminal just for convenience. it is not used.
    return nltk.grammar.nonterminals(','.join([str(x) for x in range(0, abp_domain_size+1)]))

def _calc_delta(sampled_pcfg, J, K, D):
    delta = np.zeros((2, J, K, D))  # the delta model. s * d * i
    K2 = K ** 2
    for a_index in range(K):
        # if a_index == 0:
        #     continuen

        if a_index in sampled_pcfg:
            lexical_sum = sum([items[1] for items in
                               sampled_pcfg[a_index].items() if not isinstance(items[0], tuple)])
            delta[0, 1:, a_index, :] = lexical_sum
            delta[1, 1:, a_index, :] = lexical_sum
    for i_index in range(2, J):
        for a_index in range(K):
            a = a_index
            for depth in range(D):
                nonterm_sum_a = 0
                nonterm_sum_b = 0
                if a in sampled_pcfg:
                    for rhs in sampled_pcfg[a]:
                        if not isinstance(rhs, tuple):
                            continue
                        prob = sampled_pcfg[a][rhs]
                        a_prime = rhs[0]
                        b_prime = rhs[1]
                        # print(prob, i_index, a_prime, b_prime, depth, delta.shape)
                        nonterm_sum_a += prob * delta[0, i_index-1, a_prime, depth] * delta[1, i_index-1, b_prime, depth]
                        if depth + 1 < D:
                            nonterm_sum_b += prob * delta[0, i_index-1, a_prime, depth + 1] * delta[1, i_index-1, b_prime, depth]
                    delta[0, i_index, a_index, depth] += nonterm_sum_a
                    delta[1, i_index, a_index, depth] += nonterm_sum_b
    # for i in range(2):
    #     for j in range(J):
    #          print(i, j, delta[i, j])
    return delta[0, -1,...].T, delta[1,-1,...].T

def _calc_gamma(deltas, sampled_pcfg, d):
    delta_A, delta_B = deltas
    gamma_As, gamma_Bs = [], []
    # gamma_A_counts, gamma_B_counts = [], []
    for depth in range(d):
        gamma_As.append({})
        gamma_Bs.append({})

        for lhs in sampled_pcfg:
            for rhs in sampled_pcfg[lhs]:
                if not isinstance(rhs, tuple):
                    continue
                if lhs not in gamma_As[depth]:
                    gamma_As[depth][lhs] = {}
                    gamma_Bs[depth][lhs] = {}

                if rhs not in gamma_As[depth][lhs]:
                    gamma_As[depth][lhs][rhs] = 0
                    gamma_Bs[depth][lhs][rhs] = 0

                gamma_As[depth][lhs][rhs] = np.nan_to_num(sampled_pcfg[lhs][rhs] * delta_A[
                            depth][rhs[0]] * delta_B[depth][rhs[1]] / delta_A[depth][lhs])
                if depth + 1 < d:
                    gamma_Bs[depth][lhs][rhs] = np.nan_to_num(sampled_pcfg[lhs][rhs] * delta_A[
                            depth+1][rhs[0]] * delta_B[depth][rhs[1]]  / delta_B[depth][lhs])
    return gamma_As, gamma_Bs

def pcfg_replace_model(hid_seqs, ev_seqs, bounded_model, pcfg_model, J=25, gold_pcfg_seqs=None,
                       strategy=None, ints_seqs=None, gold_pos_dict = None,
                       ac_coeff = 1.0, sample_alpha_flag=False, resume = False,
                       dnn=None, random_trees=False, productions=None, suppress_sampling=False):
    # import pdb; pdb.set_trace()
    D = bounded_model.D
    d = D + 1  # calculate d+1 depth models for all pseudo count models, but not using them in
    # _inc_counts
    K = bounded_model.K
    working_dir = pcfg_model.log_dir
    if not resume:
        if not any(hid_seqs) and not gold_pcfg_seqs and not productions: # initialization without any
            # parses
            pcfg_counts = {}
            p0_counts = {}
            if random_trees:
                hid_seqs = generate_random_trees(ev_seqs, K)
                _, pcfg_counts, p0_counts = extract_counts_from_trees(hid_seqs, K)
            logging.info('PCFG translator NULL initialization.')
        elif not gold_pcfg_seqs and not strategy: # normal sampling
            if productions is None:
                logging.info('No production passed in. PCFG translator normal rule count '
                             'extraction.')
                _, pcfg_counts, p0_counts = extract_counts_from_trees(hid_seqs, K)
            else:
                logging.info('Productions already calculated.')
                pcfg_counts, p0_counts = productions
                total_count = 0
                for parent in pcfg_counts:
                    total_count += sum(pcfg_counts[parent].values())
                total_count += sum(p0_counts.values())
                logging.info("TOTAL NUMBER OF NODES: {}".format(total_count))
        elif gold_pcfg_seqs:  # with a gold pcfg file
            logging.info('PCFG translator init with gold initialization.')
            _, pcfg_counts, p0_counts = extract_counts_from_trees(gold_pcfg_seqs, K)
        else:
            raise Exception("bad combination of initialization options!")

    else:
        pcfg_counts = None
        p0_counts = None

    # Sample an unbounded model
    if not suppress_sampling:
        logging.info('PCFG model sampling.')
        sampled_pcfg, p0 = pcfg_model.sample(pcfg_counts, p0_counts, annealing_coeff=ac_coeff,
                                             sample_alpha_flag=sample_alpha_flag,
                                             resume=resume, dnn=dnn)
    else:
        logging.info('Directly use PCFG model, no sampling.')
        sampled_pcfg, p0 = pcfg_model.get_current_pcfg()
    # At this point, we have successfully sampled an unbounded grammar model

    if d > 0:
        logging.info("Converting to delta")
        delta_A, delta_B = _calc_delta(sampled_pcfg, J, K, d) # This performs Equation 9 from the EMNLP 2018 submission
        # print(delta_A, delta_B)
        logging.info("Converting to gamma")
        gamma_A, gamma_B = _calc_gamma((delta_A, delta_B), sampled_pcfg, d) # This performs Equation 10 from the EMNLP 2018 submission
        logging.info("Bounded model setting the models")
        bounded_model.set_gammas((gamma_A, gamma_B))
    else: # Don't do bounding
        bounded_model.set_gammas(sampled_pcfg)
    bounded_model.set_p0(p0)
    bounded_model.set_lexis(pcfg_model)
    model_fn = os.path.join(working_dir, 'models.bin')
    prev_model_fn = os.path.join(working_dir, 'prev_models.bin')
    logging.info("Dumping out the bounded models")
    if os.path.exists(prev_model_fn):
        os.rename(model_fn, prev_model_fn)
    with open(model_fn, 'wb') as fh:
        bounded_model.dump_out_models(fh)
    return  sampled_pcfg, p0

def generate_random_trees(ev_seqs, K):
    trees = []
    for sent in ev_seqs:
        sent_len = len(sent)
        expanded_nodes = []
        expanding_nodes = []
        top_node = Node(None, 0, sent_len)
        top_node.k = np.random.randint(0, K)
        expanding_nodes.append(top_node)
        while expanding_nodes:
            cur_node = expanding_nodes.pop()
            expanded_nodes.append(cur_node)
            if sent_len == 1:
                break
            split = np.random.randint(cur_node.i+1, cur_node.j)
            node_b = Node(None, cur_node.i, split)
            node_b.k = np.random.randint(0, K)
            if node_b.is_terminal():
                expanded_nodes.append(node_b)
            else:
                expanding_nodes.append(node_b)
            node_c = Node(None, split, cur_node.j)
            node_c.k = np.random.randint(0, K)
            if node_c.is_terminal():
                expanded_nodes.append(node_c)
            else:
                expanding_nodes.append(node_c)
        trees.append(nodes_to_tree(expanded_nodes, ev_seqs))
    return trees
