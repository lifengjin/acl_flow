# import powerlaw
import numpy as np
from scipy import stats
# from loky import get_reusable_executor
import re
from nltk import tree
from lc_structs import sleft_transform
import logging
import torch
NUM_PROCESSES = 35

def zipf_measure(nonterm_counts:np.ndarray):
    # dist = nonterm_counts / nonterm_counts.sum()
    # simmed_obs = np.random.multinomial(100000, dist)
    # fit_result = powerlaw.Fit(simmed_obs, discrete=True, estimate_discrete=True, xmin=1)
    # L_ratio, p_val = fit_result.distribution_compare('power_law', 'lognormal_positive')
    # if L_ratio < 0:
    #     p_val = 2 - p_val
    # return fit_result.power_law.alpha, L_ratio, p_val
    return 0

def average_entropy(distribution_matrix:np.ndarray):
    ave_ent = 0

    for x in distribution_matrix:
        this_ent = stats.entropy(x)
        ave_ent += this_ent
    return ave_ent / len(distribution_matrix)

def average_confusability(child_parent_counts:np.ndarray):
    ave_conf = 0
    for child in child_parent_counts:
        conf = stats.entropy(child)
        ave_conf += conf
    return ave_conf / len(child_parent_counts)

# recall value for gold

def recall(sampled_seqs, gold_spans, aug_number, num_processes=NUM_PROCESSES):
    assert len(gold_spans) + aug_number == len(sampled_seqs), "{}, {}, {}".format(len(gold_spans), aug_number,
                                                                                  len(sampled_seqs))
    if aug_number != 0:
        sample_gold_seqs = zip(sampled_seqs[:-aug_number], gold_spans)
    else:
        sample_gold_seqs = zip(sampled_seqs, gold_spans)
    total_gold_spans = sum([len(x) for x in gold_spans])
    # matches = map(_recall, sample_gold_seqs)
    # for index, sseq in enumerate(sampled_seqs):
    #     match = _recall((sseq, gold_spans[index]))
    # executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
    # matches = executor.map(_recall, sample_gold_seqs)
    matches = []
    for seq in sample_gold_seqs:
        matches.append(_recall(seq))
    total_matches = sum(matches)
    recall = total_matches / total_gold_spans
    non_bloated_recall = (total_matches - len(gold_spans)) / (total_gold_spans - len(gold_spans))
    logging.info("Recall evaluation. Gold spans: {}; Matched spans: {}; Recall {}; NB Recall {}.".format(total_gold_spans,
                                                                                              total_matches,
                                                                                          recall, non_bloated_recall))
    return recall

def _recall(sample_seq_gold_span):
    sample_seq, gold_span = sample_seq_gold_span
    if sample_seq is None:
        return 0
    match = 0
    for subtree in sample_seq.subtrees(lambda t : t.height() > 2):
        if ' '.join(subtree.leaves()) in gold_span:
            match += 1
    return match

# calculate the loglikelihood of a PCFG grammar on some PCFG rule counts excluding the root rules
def calc_pcfg_loglikelihood(pcfg, p0, productions, p0_counts):
    loglikehood = 0
    for lhs in productions:
        for rhs in productions[lhs]:
            if len(rhs) > 1:
                int_rhs = tuple(int(x.symbol()) for x in rhs)
            else:
                int_rhs = int(rhs[0])
            loglikehood += np.log10(pcfg[int(lhs.symbol())][int_rhs])*productions[lhs][rhs]
    for p0_cat in p0_counts:
        p0_cat_int = int(p0_cat)
        loglikehood += np.log10(p0[p0_cat_int]) * p0_counts[p0_cat]

    return loglikehood

# calc tree likelihoods with the G matrices top-level only
def calc_top_vit_loglikelihood(p0, expansion, pcfg_split, trees):
    loglikehood = 0
    with torch.no_grad():
        expansion_3d = expansion.view(-1, int(expansion.shape[1] ** 0.5), int(expansion.shape[1]**0.5))
        for tree in trees:
            tree_ll = torch.tensor([0]).to('cuda')
            top_a = int(tree.label())
            tree_ll += p0[top_a] / np.log(10)
            productions = tree.productions()
            for production in productions:
                if len(production.rhs()) == 1:
                    continue
                else:
                    parent = int(production.lhs().symbol())
                    child1, child2 = int(production.rhs()[0].symbol()), int(production.rhs()[1].symbol())
                    tree_ll += expansion_3d[parent, child1, child2] + pcfg_split[parent, 0]
            loglikehood += tree_ll.item()
    return loglikehood



    return loglikehood

def conditional_tree_likelihood(evidence, tree_joint):
    # they should be in log domain
    return tree_joint - evidence

# def accu_ave_tree_depth(trees, num_processes=NUM_PROCESSES):
#
#     executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
#     timestep_seqs = executor.map(sleft_transform.simple_left_shifted_transform, trees)
#     print(type(timestep_seqs))
#
#     executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
#     ave_depths = executor.map(get_accu_ave_tree_depth_per_tree, timestep_seqs, timeout=1.)
#     ave_depths = list(ave_depths)
#     total_ave_depth = sum(ave_depths) / len(trees)
#     return total_ave_depth

def accu_ave_tree_depth(trees, num_processes=NUM_PROCESSES):

    # executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
    # timestep_seqs = executor.map(sleft_transform.simple_left_shifted_transform, trees)
    timestep_seqs = []
    for t in trees:
        timestep_seqs.append((sleft_transform.simple_left_shifted_transform(t)))
    # print(type(timestep_seqs))

    # executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
    # ave_depths = executor.map(get_accu_ave_tree_depth_per_tree, timestep_seqs)
    ave_depths = []
    for seq in timestep_seqs:
        ave_depths.append(get_accu_ave_tree_depth_per_tree(seq))
    # print(ave_depths)
    ave_depths = list(ave_depths)
    # print(ave_depths)
    total_ave_depth = sum(ave_depths) / len(ave_depths)
    return total_ave_depth

def get_accu_ave_tree_depth_per_tree(timesteps):
    depth = 0
    acc_depth = 0.

    for index, timestep in enumerate(timesteps):
        assert depth >= 0 and acc_depth >= 0
        if index == 0:
            if not timestep.a_action.is_null():
                depth += 1

        else:
            prev_timestep = timesteps[index-1]
            if timestep.a_action.is_null() and prev_timestep.b_action.is_null():
                depth -= 1
            elif (not timestep.a_action.is_null()) and (not prev_timestep.b_action.is_null()):
                depth += 1
            else:
                pass
        acc_depth += depth
    depth_result = acc_depth / len(timesteps)
    return depth_result

def get_max_depth(trees, num_processes=NUM_PROCESSES):

    # executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
    max_depths = []
    for t in trees:
        max_depths.append(_get_max_depth(t))
    # max_depths = executor.map(_get_max_depth, trees)
    max_depths = list(max_depths)
    ave_max_depth = sum(max_depths) / len(max_depths)

    return ave_max_depth

def _get_max_depth(tree : tree.Tree, factor : str ='right') -> int:
    tree.collapse_unary()
    max_depth = 0

    tree.chomsky_normal_form(factor=factor)

    leaf_positions = tree.treepositions('leaves')

    for leaf_p in leaf_positions:
        p_str = '0'+''.join([str(x) for x in leaf_p[:-1]])
        turns = re.findall('0[1-9]', p_str)
        this_depth = len(turns)
        if this_depth > max_depth:
            max_depth = this_depth
    if max_depth == 0 and len(leaf_positions) != 1:
        print(leaf_positions)
        print(tree)
        raise Exception

    max_depth /= len(tree.leaves())

    return max_depth

def get_lbrb_const_length(trees, num_processes=NUM_PROCESSES) -> (float, float):

    # executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
    # lbrb_lengths = executor.map(_get_lbrb_const_length, trees)
    lbrb_lengths = []
    for t in trees:
        lbrb_lengths.append(_get_lbrb_const_length(t))
    lb_lengths, rb_lengths, max_lb_length, max_rb_length = list(zip(*list(lbrb_lengths)))
    ave_lb_length = sum(lb_lengths) / len(lb_lengths)
    ave_rb_length = sum(rb_lengths) / len(rb_lengths)
    ave_max_lb_length = sum(max_lb_length) / len(max_lb_length)
    ave_max_rb_length = sum(max_rb_length) / len(max_rb_length)
    ave_all_length = sum(lb_lengths+rb_lengths) / (len(lb_lengths) + len(rb_lengths))
    ave_max_all_length = sum([max(x, y) for (x, y) in zip(max_lb_length, max_rb_length)]) / len(max_lb_length)
    return ave_lb_length, ave_rb_length, ave_max_lb_length, ave_max_rb_length, ave_all_length, ave_max_all_length

def _get_lbrb_const_length(t : tree.Tree):
    lb_const_lengths = 0
    max_lb_const_length = 0
    lb_const_num = 0
    rb_const_lengths = 0
    max_rb_const_length = 0
    rb_const_num = 0
    # total_length = len(t.leaves())
    for position in t.treepositions():
        # print(t[position])
        if not (isinstance(t[position],str) or isinstance(t[position][0],str)):
            if len(t[position][0]) == 2:
                lb_const_num += 1
                this_length = len(t[position][0].leaves())
                lb_const_lengths += this_length
                if this_length > max_lb_const_length:
                    max_lb_const_length = this_length
            if len(t[position][1]) == 2:
                rb_const_num += 1
                this_length = len(t[position][1].leaves())
                rb_const_lengths += this_length
                if this_length > max_rb_const_length:
                    max_rb_const_length = this_length
    rb_const_lengths = rb_const_lengths/rb_const_num if rb_const_num != 0 else 0
    lb_const_lengths = lb_const_lengths/lb_const_num if lb_const_num != 0 else 0
    # max_lb_const_length /= total_length
    # max_rb_const_length /= total_length
    return lb_const_lengths, rb_const_lengths, max_lb_const_length, max_rb_const_length