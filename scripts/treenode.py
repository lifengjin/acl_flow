import numpy as np
from nltk import tree
from collections import Counter, defaultdict
class Rule:
    def __init__(self, lhs, rhs1, rhs2=None):
        self.lhs = lhs
        self.rhs1 = rhs1
        self.rhs2 = rhs2
        if self.rhs2 is None:
            self.terminal = self.rhs1

    def get_rule(self):
        if self.rhs2 is not None:
            return (self.lhs, self.rhs1, self.rhs2)
        else:
            return (self.lhs, self.rhs1)

def nodes_to_tree(nodes, sent):
    bracketed_string = [''] * (len(sent) * 2 + 1 )
    bracketed_string[1::2] = [str(x.item()) for x in sent]
    brackets = [''] * (len(sent)+1)
    nodes.sort(key=lambda x : x.span_length, reverse=True)
    # print(nodes)
    for node in nodes:
        brackets[node.i] += '(' + str(node.k) + ' '
        brackets[node.j] = ')' + brackets[node.j]
        # print(brackets)
    bracketed_string[::2] = brackets
    # print(bracketed_string)
    # exit()
    try:
        this_tree = tree.Tree.fromstring(''.join(bracketed_string))
    except:
        print(''.join(bracketed_string))
        print(nodes)
        raise
    productions = this_tree.productions()
    production_counter_dict = defaultdict(Counter)
    for rule in productions:
        production_counter_dict[rule.lhs()][rule.rhs()] += 1
    p0_counter = Counter()
    p0_counter[this_tree.label()] = 1
    production_counters = (production_counter_dict, p0_counter)
    l_branch, r_branch = calc_branching_score(this_tree)
    return this_tree, production_counters, (l_branch, r_branch)

class Node:
    def __init__(self, cat, i, j, D=0, K=0, parent=None):
        self.D = D
        self.K = K
        self.cat = int(cat)
        self.i = i
        self.j = j
        self.s, self.d = -1, -1
        self.Q = (D+1) * 2 * K
        self.d = 0
        self.k = 0
        self.s = 0
        self.span_length = self.j - self.i
        self.parent = parent
        self.__unwind_cat()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} ({},{})".format((self.s, self.d, self.k), self.i, self.j)

    def str(self):
        return self.__str__()

    def __unwind_cat(self):
        if self.D == -1:
            self.k = self.cat
        elif self.K != 0 and self.D != -1:
            self.s, self.d, self.k = np.unravel_index(self.cat, (2, self.D+1, self.K))
            # return "s{} d{} {}".format(self.s, self.d, self.k)
        else:
            return self.cat

    def is_terminal(self):
        if self.j - self.i == 1:
            return True
        return False

# class Node_tree:
#     def
# used in calc right branching warning.
def calc_branching_score(t):
    r_branch = 0
    l_branch = 0
    # print(t)
    for position in t.treepositions():
        # print(t[position])
        if not (isinstance(t[position],str) or isinstance(t[position][0],str)):
            if len(t[position][0]) == 2:
                l_branch += 1
            if len(t[position][1]) == 2:
                r_branch += 1
    return l_branch, r_branch