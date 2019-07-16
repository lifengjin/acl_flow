import sys
import nltk

fns = sys.argv[1:]

assert all([x.endswith('linetrees') for x in fns])

for index, fn in enumerate(fns):
    fn_list = fn.split('.')

    ints_fn = fn_list[:-1]
    ints_fn.append('ints')
    ints_fn = '.'.join(ints_fn)
    if index == 0:
        dic_fn = fn_list[:-1]
        dic_fn.append('dict')
        dic_fn = '.'.join(dic_fn)
        word_dict = {}

    with open(fn, encoding='utf8') as of, open(ints_fn, 'w', encoding='utf8') as intsf:
        for line in of:
            try:
                tline = line.strip()
                t = nltk.Tree.fromstring(tline)
                words = t.leaves()

            except:
                line = line.strip()
                words = line.split(' ')
            int_sent = []
            for word in words:
                if not word.startswith('PUNCT'):
                    word = word.lower()
                if word in word_dict:
                    int_sent.append(word_dict[word])
                else:
                    word_dict[word] = len(word_dict)
                    int_sent.append(word_dict[word])
            print(' '.join([str(x) for x in int_sent]), file=intsf)

with open(dic_fn,'w',encoding='utf8') as dicf:
    for word in word_dict:
        print(word, word_dict[word], file=dicf)