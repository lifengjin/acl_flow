import nltk
import sys

linetrees_file = sys.argv[1]
if len(sys.argv) > 2:
    punc_dict_file = sys.argv[2]

    with open(punc_dict_file, 'r', encoding='utf-8') as f:
        punct_dict = {}
        for line in f:
            (punct, sent_index) = line.rstrip().split("\t")
            punct_dict[sent_index] = punct
else:
    print('No punct dict is given!')
    punct_dict = {}

with open(linetrees_file) as trees_fh, open(linetrees_file.replace('linetrees', 'linetoks'), 'w') as words_fh, \
open(linetrees_file.replace('linetrees', 'lower.linetoks'), 'w') as lower_words_fh:
    for line in trees_fh:
        words = nltk.tree.Tree.fromstring(line).leaves()
        words = [punct_dict[x] if x in punct_dict else x for x in words ]
        print(' '.join(words), file=words_fh)
        print(' '.join(words).lower(), file=lower_words_fh)