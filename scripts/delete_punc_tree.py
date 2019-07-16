import nltk
from .metrics import NUM_PROCESSES
from functools import partial
# from loky import get_reusable_executor
CTB_PUNCS = {'（', '、', '）', '，', '。', '“', '”', '；', '－－', '：', '——', '《', '》', '’', '－', '？',
             '━━', '———', '『', '』', '！', '—', '‘', '·', '∶', '「', '」', '／', '-', '＊', '＂', '．',
             '──', '…', '----', '〈', '〉', '?', ':', '//', '.', '~', '/', ',', '*', '～', '【', '】',
             '>', '<'}

NEGRA_PUNCS = {';', '"', '?', '...', '*lrb*', '!', ':', '/', '*rrb*', ',', '.', '·', '-', "'", '--'}

WSJ_PUNCS = {'.', ',', ':',  '\'\'', '``', '--', ';',  '?', '!', '...',
              '`', "'", 'lrb', 'rrb','lcb', 'rcb','-', '$', '#', 'us$', 'a$', 'hk$', 'c$', 'm$',
             's$'}

ALL_PUNCS = CTB_PUNCS | NEGRA_PUNCS | WSJ_PUNCS

UNIVERSAL_PUNCT = 'PUNCT'


def delete_puncs(trees, word_dict, corpus_type='ud', only_final=False):
    depunced_trees = []
    if corpus_type == 'ctb':
        puncs = {str(key) for key in word_dict if word_dict[key] in CTB_PUNCS}
    elif corpus_type == 'wsj':
        puncs = {str(key) for key in word_dict if word_dict[key] in WSJ_PUNCS}
    elif corpus_type == 'negra':
        puncs = {str(key) for key in word_dict if word_dict[key] in NEGRA_PUNCS}
    elif corpus_type == 'all':
        puncs = {str(key) for key in word_dict if word_dict[key] in ALL_PUNCS}
    else:
        puncs = UNIVERSAL_PUNCT

    for t in trees:
        if t is None:
            depunced_trees.append(t)
            continue
        t = nltk.ParentedTree.convert(t)
        for sub in reversed(list(t.subtrees())):
            if sub.height() == 2:
                if (isinstance(puncs, dict) and sub[0] in puncs) or word_dict[int(sub[0])][:5] == UNIVERSAL_PUNCT or \
                        sub.label() == UNIVERSAL_PUNCT:  #

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

            if only_final:
                break
        t = nltk.Tree.convert(t)
        t.collapse_unary(collapsePOS=True, collapseRoot=True)
        depunced_trees.append(t)
    return depunced_trees

def _delete_puncs_single(t, word_dict, corpus_type='ud', only_final=False):
    depunced_trees = []
    if corpus_type == 'ctb':
        puncs = {str(key) for key in word_dict if word_dict[key] in CTB_PUNCS}
    elif corpus_type == 'wsj':
        puncs = {str(key) for key in word_dict if word_dict[key] in WSJ_PUNCS}
    elif corpus_type == 'negra':
        puncs = {str(key) for key in word_dict if word_dict[key] in NEGRA_PUNCS}
    elif corpus_type == 'all':
        puncs = {str(key) for key in word_dict if word_dict[key] in ALL_PUNCS}
    else:
        puncs = UNIVERSAL_PUNCT

    t = nltk.ParentedTree.convert(t)
    for sub in reversed(list(t.subtrees())):
        if sub.height() == 2:
            if (isinstance(puncs, dict) and sub[0] in puncs) or word_dict[int(sub[0])][:5] == UNIVERSAL_PUNCT or \
                    sub.label() == UNIVERSAL_PUNCT:  #

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

        if only_final:
            break
    t = nltk.Tree.convert(t)
    t.collapse_unary(collapsePOS=True, collapseRoot=True)
    return t

def multiprocess_delete_puncs(trees, word_dict, num_processes=NUM_PROCESSES):
    delete_punc_func = partial(_delete_puncs_single, word_dict=word_dict)
    # executor = get_reusable_executor(max_workers=num_processes, timeout=2.)
    # depunced_trees = executor.map(delete_punc_func, trees)
    depunced_trees = []
    for t in trees:
        depunced_trees.append(delete_punc_func(t))
    return list(depunced_trees)