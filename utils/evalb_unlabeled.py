import gzip, multiprocessing
import nltk
from sklearn.metrics import v_measure_score
import argparse
from fix_terminals_wsj import single_fix_terms
from itertools import chain
import numpy as np
from collections import Counter

############
# PIOC files must be fix-terminal-ed first.

############

parser = argparse.ArgumentParser()
parser.add_argument('--gold', '-g', required=True, type=str, help='gold tree fn')
parser.add_argument('--pred', '-p', required=True, type=str, help='predicted tree fn')
parser.add_argument('--verbose', '-v', action='store_true', default=False)
# parser.add_argument('--leave-punc', '-p', action='store_false', default=True)
args = parser.parse_args()

gold_fn = args.gold
pred_fn = args.pred
# if 'negra' not in gold_fn:
#     interested_phrases = ['NP', 'VP', 'ADJP', 'PP']
# else:
#     interested_phrases = ['NP', 'VP', 'AP', 'PP']
#
print(pred_fn)
def eval(gold_pred):

    gt = gold_pred[0]
    pt = gold_pred[1]

    if pt.label() == 'x':
        return 0, 0, 0, 0, [], [], 0, Counter(), Counter(), Counter(), Counter()

    g_spans = []
    p_spans = []
    gold_labels = []
    pred_labels = []
    matching_gold_labels = []
    matching_pred_labels = []
    matching_labeled_consts = Counter()
    matching_cross_labeled_consts = Counter()
    gold_labeled_counts = Counter()
    # all_gold_label_counts = Counter()
    all_pred_label_counts = Counter()
    assert len(gt.leaves()) == len(pt.leaves()), "{}\n {}".format(gt, pt)
    # print(gt)
    for subtree in gt.subtrees(lambda x: x.height() > 2):
        g_spans.append(' '.join(subtree.leaves()))
        this_gold_labels = subtree.label().split('+')
        if this_gold_labels[0] == '':
            if len(this_gold_labels) > 1:
                chosen_label = this_gold_labels[1]
            else:
                chosen_label = 'S'  # this is special for negra
        else:
            chosen_label = this_gold_labels[0]
        if '-' in chosen_label:
            chosen_label = chosen_label.split('-')[0]
        gold_labels.append(chosen_label)
        gold_labeled_counts.update([chosen_label])

    for subtree in pt.subtrees(lambda x: x.height() > 2):
        p_spans.append(' '.join(subtree.leaves()))
        pred_labels.append(subtree.label().split('+')[0])

    ggt = gt.copy(deep=True)
    ppt = pt.copy(deep=True)

    this_total_gold_spans = len(g_spans)
    this_total_predicted_spans = len(p_spans)
    this_correct_spans = 0
    len_gold = len(g_spans)
    if len_gold == 0:
        # print('Sent', index, 'Single word sent!', gt)
        return this_total_gold_spans, this_total_predicted_spans, 1, 0, [], [], 1, Counter(), Counter(), Counter(), Counter()
    len_predicted = len(p_spans)
    all_pred_label_counts.update(pred_labels)
    gg_spans = g_spans[:]
    ggold_labels = gold_labels[:]
    for span, span_label in zip(p_spans, pred_labels):
        if span in g_spans:
            this_correct_spans += 1
            matching_pred_labels.append(span_label)
            g_span_index = g_spans.index(span)
            matching_gold_labels.append(gold_labels[g_span_index])
            matching_labeled_consts.update([gold_labels[g_span_index]])
            del g_spans[g_span_index]
            del gold_labels[g_span_index]

    i = 0
    for g_leaf, p_leaf in zip(ggt.treepositions(order='leaves'), ppt.treepositions(order='leaves')):
        ggt[g_leaf] = str(i)
        ppt[p_leaf] = str(i)
        i += 1

    gg_int_spans = []
    for subtree in ggt.subtrees(lambda x: x.height() > 2):
        gg_int_spans.append(' '.join(subtree.leaves()))
    pp_int_spans = []
    for subtree in ppt.subtrees(lambda x: x.height() > 2):
        pp_int_spans.append(' '.join(subtree.leaves()))

    for span in pp_int_spans:
        if span not in gg_int_spans:
            start, end = span[0], span[-1]
            for gspan in gg_int_spans:
                if (start in gspan and gspan[0] != start and end not in gspan) or (start not in gspan and end in gspan and
                gspan[-1] != end):
                    break
            else:
                cross_span_index = pp_int_spans.index(span)
                matching_cross_labeled_consts.update([pred_labels[cross_span_index]])

    this_r = this_correct_spans / len_gold
    this_p = this_correct_spans / len_predicted
    this_f = 2 * (this_p * this_r / (this_p + this_r + 1e-6))
    if args.verbose:
        print('Sent', index, 'Rec', '{:.04f}'.format(this_r), 'Prec', '{:.04f}'.format(this_p
                                                                                       ), 'F1', '{:.04f}'.format(this_f))

        print(gt)
        print(pt)

        print(g_spans)

        print(p_spans)

        print(matching_pred_labels)

        print(matching_gold_labels)

        exit()
    this_words = len(gt.leaves())
    return this_total_gold_spans, this_total_predicted_spans, 0, this_correct_spans, matching_gold_labels, \
           matching_pred_labels, this_words, gold_labeled_counts, matching_labeled_consts, \
           matching_cross_labeled_consts, all_pred_label_counts

def delete_punc(t):
    t = nltk.ParentedTree.convert(t)
    for sub in reversed(list(t.subtrees())):
        if sub.height() == 2:
            if 'PUNCT' in sub.label() or 'PUNCT' in sub[0]:  #
                parent = sub.parent()
                while parent and len(parent) == 1:
                    sub = parent
                    parent = sub.parent()
                try:
                    del t[sub.treeposition()]
                except:
                    print(t)
                    print(t[sub.treeposition()])
                    raise

    t = nltk.Tree.convert(t)
    t.collapse_unary(collapsePOS=True, collapseRoot=True)
    return t


def calc_measures_at_n(n, gold_spans, pred_spans, correct_spans, word_counts):
    mask = word_counts <= n

    gold_spans_sum = gold_spans[mask].sum()
    pred_spans_sum = pred_spans[mask].sum()
    correct_spans_sum = correct_spans[mask].sum()
    r = correct_spans_sum / gold_spans_sum
    p = correct_spans_sum / pred_spans_sum
    f = 2*p*r / (p+r+1e-6)
    print('Length <=', n, 'Rec', '{:.04f}'.format(r), 'Prec', '{:.04f}'.format(p), 'F1', '{:.04f}'.format(f))

if pred_fn.endswith('.gz'):
    with gzip.open(pred_fn, 'rt') as pfh:
        pred_lines = pfh.readlines()
else:
    with open(pred_fn) as pfh:
        pred_lines = pfh.readlines()

new_pred_lines = []
for line in pred_lines:
    if '#!#!' in line:
        _, t = line.split('#!#!')
    else:
        t = line
    if t == "\n":
        t = "(x x)"
    new_pred_lines.append(t)
pred_lines = new_pred_lines

with open(gold_fn) as gfh:
    gold_lines = gfh.readlines()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:

        gold_trees = pool.map(nltk.Tree.fromstring, gold_lines)
        pred_trees = pool.map(nltk.Tree.fromstring, pred_lines)

        gold_trees = pool.map(delete_punc, gold_trees)
        pred_trees = pool.map(delete_punc, pred_trees)

        gold_trees, pred_trees = zip(*pool.map(single_fix_terms, zip(gold_trees, pred_trees)))

        pool.close()
        pool.join()

    # total_gold_spans = 0
    # total_predicted_spans = 0
    # correct_spans = 0
    # total_single_word_sent = 0
    #
    # matching_gold_labels = []
    # matching_pred_labels = []

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        total_gold_spans, total_predicted_spans, total_single_word_sent, correct_spans, matching_gold_labels,\
        matching_pred_labels, word_counts, gold_labeled_counts, matching_labeled_consts \
        , matching_cross_labeled_consts , pred_label_counts = zip(*pool.map(eval, zip(gold_trees, pred_trees)))
        total_gold_spans_sum = sum(total_gold_spans)
        total_predicted_spans_sum = sum(total_predicted_spans)
        total_single_word_sent = sum(total_single_word_sent)
        correct_spans_sum = sum(correct_spans)
        matching_gold_labels = list(chain.from_iterable(matching_gold_labels))
        matching_pred_labels = list(chain.from_iterable(matching_pred_labels))
        total_gold_spans = np.array(total_gold_spans)
        total_predicted_spans = np.array(total_predicted_spans)
        correct_spans = np.array(correct_spans)
        word_counts = np.array(word_counts)

    accu_gold_counts = Counter()
    matching_label_counts = Counter()
    acc_pred_counts = Counter()
    matching_cross = Counter()
    for gold_counter, matching_counter, cross_counter, p_counter in zip(gold_labeled_counts, matching_labeled_consts,
                                                         matching_cross_labeled_consts, pred_label_counts):
        accu_gold_counts.update(gold_counter)
        matching_label_counts.update(matching_counter)
        acc_pred_counts.update(p_counter)
        matching_cross.update(cross_counter)

    r = correct_spans_sum / total_gold_spans_sum
    p = correct_spans_sum / total_predicted_spans_sum
    f = 2 * (p * r / (p + r))
    print('Total single word sent', total_single_word_sent)

    print('*'*50)
    print('Total', 'Rec', '{:.04f}'.format(r), 'Prec', '{:.04f}'.format(p), 'F1', '{:.04f}'.format(f))

    calc_measures_at_n(10, total_gold_spans, total_predicted_spans, correct_spans, word_counts)
    calc_measures_at_n(20, total_gold_spans, total_predicted_spans, correct_spans, word_counts)
    calc_measures_at_n(30, total_gold_spans, total_predicted_spans, correct_spans, word_counts)
    calc_measures_at_n(40, total_gold_spans, total_predicted_spans, correct_spans, word_counts)

    ## RVM: recall+VM
    assert len(matching_pred_labels) == len(matching_gold_labels) and len(matching_gold_labels) == correct_spans.sum()
    import pickle
    pickle.dump((matching_pred_labels, matching_gold_labels, correct_spans,
                 accu_gold_counts, matching_cross, acc_pred_counts), open('matching_labels.pkl', 'wb'))
    vm = v_measure_score(matching_gold_labels, matching_pred_labels)
    print('VM: ', vm )
    print('RVM_m: ', r*vm )
    print('RVM_h: ', 2*r*vm / (r+vm+1e-6))

    for name, count in accu_gold_counts.most_common(8):
        print(name, matching_label_counts[name] / (1e-6+count))