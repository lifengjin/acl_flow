import nltk
import sys
import gzip
import argparse
CTB_PUNCS = {'（', '、', '）', '，', '。', '“', '”', '；', '－－', '：', '——', '《', '》', '’', '－', '？',
             '━━', '———', '『', '』', '！', '—', '‘', '·', '∶', '「', '」', '／', '-', '＊', '＂', '．',
             '──', '…', '----', '〈', '〉', '?', ':', '//', '.', '~', '/', ',', '*', '～', '【', '】',
             '>', '<'}

NEGRA_PUNCS = {';', '"', '?', '...', '*lrb*', '!', ':', '/', '*rrb*', ',', '.', '·', '-', "'", '--'}

WSJ_PUNCS = {'.', ',', ':',  '\'\'', '``', '--', ';',  '?', '!', '...',
              '`', "'", 'lrb', 'rrb','lcb', 'rcb','-', '$', '#', 'us$', 'a$', 'hk$', 'c$', 'm$',
             's$'}

ALL_PUNCS = CTB_PUNCS | NEGRA_PUNCS | WSJ_PUNCS

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--style', required=True, choices=['ctb', 'wsj', 'negra', 'all', 'ud'])
parser.add_argument('--first-n', type=int, default=sys.maxsize)
parser.add_argument('--only-final', default=False, action='store_true')
args = parser.parse_args()
lt_file = args.file
corpus_type = args.style # ctb, negra, wsj, all

first_n = args.first_n

only_final = args.only_final

out_file = lt_file.split('.')

out_file.insert(-1, 'nopunc')
if out_file[-1] == 'gz':
    out_file = out_file[:-1]
out_file = '.'.join(out_file)
if first_n < sys.maxsize:
    first_n_suffix = '.f'+str(first_n)
    out_file += first_n_suffix

if lt_file.endswith('gz'):
    i = gzip.open(lt_file, 'rt', encoding='utf8')
else:
    i = open(lt_file, 'r', encoding='utf8')

with open(out_file, 'w', encoding='utf8') as o:
    k = 0
    for line in i:
        if k >= first_n:
            break
        else:
            k += 1
        t = nltk.ParentedTree.fromstring(line)
        for sub in reversed(list(t.subtrees())):
            if sub.height() == 2 and ((sub[0] in CTB_PUNCS and corpus_type == 'ctb') or (sub[0] in
                NEGRA_PUNCS and corpus_type == 'negra') or (sub[0] in WSJ_PUNCS and corpus_type ==
                'wsj') or (sub[0] in ALL_PUNCS and corpus_type == 'all') or ('PUNCT' in sub.label() and corpus_type ==
                           'ud')):  #
                # abbreviated
                #  test
                parent = sub.parent()
                while parent and len(parent) == 1:
                    sub = parent
                    parent = sub.parent()
                # print(sub, "will be deleted")
                try:
                    del t[sub.treeposition()]
                except:
                    print(t)
                    print(t[sub.treeposition()])
                    raise
            if only_final:
                break
        t = nltk.Tree.convert(t)
        # t.collapse_unary(collapsePOS=True, collapseRoot=True)
        print(t.pformat(margin=10000000), file=o)
i.close()