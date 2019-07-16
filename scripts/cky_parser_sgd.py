import itertools
from .cky_utils import *
from .treenode import Node, Rule, nodes_to_tree
import logging
# import numpy as np
import scipy.sparse as sparse
from scipy.special import logsumexp
import time
import bidict
import torch
# from .cky_parser_lcg_sparse import sparse_vit_add_trick

# for dense grammar only! ie D must be -1
class batch_CKY_parser:
    def __init__(self, K=0, D=0, max_len=40, seed=None, pcfg_model=None):
        # assert D == -1
        if seed is not None:
            self.prng = torch.cuda.manual_seed(seed=seed)

        assert D != 0 and K != 0, 'Sampler initialization error: K {}, D {}'.format(K, D)
        # logging.info("sampler: getting K {} and D {}".format(K, D))
        self.K = K
        self.D = D
        self.lexis = None # Preterminal expansion part of the grammar (this will be dense)
        self.G = None     # Nonterminal expansion part of the grammar (usually be a sparse matrix representation)
        self.p0 = None
        self.pcfg_model = pcfg_model
        self.max_len = max_len
        self.num_points = max_len + 1
        self.chart = np.zeros((max_len+1, max_len+1), dtype=object) # split points = len + 1
        # self.viterbi_chart = np.zeros_like(self.chart, dtype=np.float32)
        self.Q = self.calc_Q(self.K, self.D)
        # self.identity_Q = np.ones((self.Q, self.Q), dtype=np.float32)
        # logging.info("sampler: getting Q {}".format(self.Q))
        self.this_sent_len = -1
        self.U = 0 # the random numbers used for sampling
        self.counter = 0
        self.vocab_prob_list = []
        self.finished_vocab = set()

    def set_models(self, p0, expansion, emission, embeddings=None, pcfg_split=None, embedding_type='none', k2=None):
        self.log_G = expansion
        self.log_p0 = p0
        self.log_lexis = emission
        self.embeddings = embeddings
        self.pcfg_split = pcfg_split
        self.embedding_type = embedding_type
        self.k2 = k2

    def marginal(self, sents, viterbi_flag=False, only_viterbi=False, sent_indices=None):
        self.sent_indices = sent_indices
        if not only_viterbi:
            lexical_l1 = self.compute_inside_logspace(sents)
            # nodes_list, logprob_list = self.sample_tree(sents)
            # nodes_list, logprob_list = self.sample_tree_logspace(sents)
            logprob_list = self.marginal_likelihood_logspace(sents)
        else:
            logprob_list = []
        self.viterbi_sent_indices = self.sent_indices
        self.sent_indices = None

        if viterbi_flag:
            with torch.no_grad():
                vnodes = []
                for sent_index, sent in enumerate(sents):
                    self.sent_indices = [self.viterbi_sent_indices[sent_index],]
                    backchart, max_cats = self.compute_viterbi_inside(sent)
                    this_vnodes = self.viterbi_backtrack(backchart, sent, max_cats)
                    vnodes += this_vnodes

        self.this_sent_len = -1

        vtree_list, vproduction_counter_dict_list, vlr_branches_list = [], [], []

        if viterbi_flag:
            for sent_index, sent in enumerate(sents):
                vthis_tree, vproduction_counter_dict, vlr_branches = nodes_to_tree(vnodes[sent_index], sent)
                vtree_list.append(vthis_tree)
                vproduction_counter_dict_list.append(vproduction_counter_dict)
                vlr_branches_list.append(vlr_branches)
        else:
            vtree_list, vproduction_counter_dict_list, vlr_branches_list = [None]*len(sents), [None]*len(sents), \
                                                                 [None]*len(sents)

        self.counter+=1
        return [], logprob_list, [], [], \
               vtree_list, vproduction_counter_dict_list, vlr_branches_list, lexical_l1

    # @profile
    def compute_inside_logspace(self, sents): #sparse
        try:
            self.this_sent_len = len(sents[0])
        except:
            print(sents)
            raise
        batch_size = len(sents)
        sent_len = self.this_sent_len

        num_points = sent_len + 1
        # left chart is the left right triangle of the chart, the top row is the lexical items, and the bottom cell is the
        #  top cell. The right chart is the left chart pushed against the right edge of the chart. The chart is a square.
        self.left_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to('cuda')
        self.right_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to('cuda')
        # print('lex')
        lexical_l1, _ = self.get_lexis_prob(sents, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):

            left_min = 0
            left_max = sent_len - ij_diff
            right_min = ij_diff
            right_max = sent_len
            height = ij_diff

            b = self.left_chart[0:height, left_min:left_max] # (all_ijdiffs, i-j, batch, Q) a square of the left chart
            c = torch.flip(self.right_chart[0:height, right_min:right_max], dims=[0])
            #
            dot_temp_mat = torch.logsumexp(b[...,None]+c[...,None,:], dim=0).view(sent_len-ij_diff, batch_size, -1)
            # i-j, batch, Q**2
            # dense
            filtered_kron_mat = self.log_G + dot_temp_mat[:, :, None, :] # i-j, batch, Q, Q**2
            y1 = torch.logsumexp(filtered_kron_mat, dim=-1) # i-j, batch, Q

            self.left_chart[height, left_min:left_max] = y1
            self.right_chart[height, right_min:right_max] = y1
        return lexical_l1

    def marginal_likelihood_logspace(self, sents):
        batch_size = len(sents)
        nodes_list = []

        sent_len = self.this_sent_len
        topnode_pdf = self.left_chart[sent_len-1, 0]

        # draw the top node
        p_topnode = topnode_pdf + self.log_p0
        # norm_term = np.linalg.norm(p_topnode,1)
        logprob_e = torch.logsumexp(p_topnode, dim=1)
        logprobs = logprob_e / np.log(10)

        return logprobs

    # @profile
    def sample_tree_logspace(self, sents):
        batch_size = len(sents)
        nodes_list = []

        sent_len = self.this_sent_len
        topnode_pdf = self.left_chart[sent_len-1, 0]

        # draw the top node
        p_topnode = topnode_pdf + self.log_p0
        # norm_term = np.linalg.norm(p_topnode,1)
        logprob_e = torch.logsumexp(p_topnode, dim=1)
        logprobs = (logprob_e / np.log(10)).to('cpu').tolist()

        # normed_p_topnode = torch.exp(p_topnode - logprob_e)

        top_As = torch.distributions.Categorical(logits=p_topnode).sample().squeeze()
        A_cats = top_As.tolist()
        # random_darts = torch.rand((sent_len * batch_size))
        # random_dart_index = 0

        for sent_index, sent in enumerate(sents):
            # print('process sent {}'.format(sent_index))
            if batch_size == 1:
                A_cat = A_cats
            else:
                A_cat = A_cats[sent_index]
            expanding_nodes = []
            expanded_nodes = []
            # rules = []
            assert self.this_sent_len > 0, "must call inside pass first!"


            # print(logprob, logprob_e)

            # prepare the downward sampling pass
            top_node = Node(A_cat, 0, sent_len, self.D, self.K)
            if sent_len > 1:
                expanding_nodes.append(top_node)
            else:
                expanded_nodes.append(top_node)
            # rules.append(Rule(None, A_cat))
            temp_kron_vector = self.kron_vec_q2
            temp_multiply_vector = self.dot_vec_q2
            kron_temp_vector_2d_view = self.kron_vec_view_qq
            kth_node = -1

            while expanding_nodes:
                # print(sent_len, expanding_nodes)
                working_node = expanding_nodes.pop()
                i, j = working_node.i, working_node.j
                left_i = j - i - 1
                left_j = i
                right_j = j-1
                height = left_i
                a_likelihood = self.left_chart[left_i, left_j, sent_index, working_node.cat]
                cur_G_row = self.log_G[working_node.cat]


                # print(self.left_chart[0:height, left_j, sent_index][..., None].shape, torch.flip(self.right_chart[
                #                                                                 0:height, right_j, sent_index],
                #                                                                                  dims=[0]).shape )
                all_k_dist = self.left_chart[0:height, left_j, sent_index][..., None] + torch.flip(self.right_chart[
                                                                                0:height, right_j, sent_index][...,None,:],
                                                                                                   dims=[0])

                all_k_dist = all_k_dist.squeeze().view((height, -1)) + cur_G_row
                all_k_dist = all_k_dist.view((-1, ))
                normed_k_dist = all_k_dist - a_likelihood

                k_b_c = torch.distributions.Categorical(logits=normed_k_dist).sample().item()

                k = k_b_c // self.Q**2 + left_j + 1
                cat_bc = k_b_c % self.Q**2
                b_cat = cat_bc // self.Q
                c_cat = cat_bc % self.Q

                # print( a_likelihood, torch.logsumexp(all_k_dist, dim=0))
                # assert  a_likelihood.item() == torch.logsumexp(all_k_dist).item()
                #
                # for k in range(working_node.i + 1, working_node.j):
                #
                #     # np.outer(viterbi_chart[i,k].expm1(), viterbi_chart[k,j].expm1(),
                #     #          out=kron_temp_vector_2d_view)
                #     np.add(self.chart[working_node.i, k][sent_index][:, None], self.chart[k, working_node.j][sent_index],
                #            out=kron_temp_vector_2d_view)
                #
                #     joint_k_B_C = cur_G_row + temp_kron_vector + cur_G_row_mask
                #     joint_k_B_C = joint_k_B_C.astype(np.float64)
                #     total_likelihood_k = logsumexp(joint_k_B_C)
                #     k_marginal += np.exp(total_likelihood_k - a_likelihood)
                #
                #     if k_marginal > k_dart:
                #         # print(kth_node, k_dart, k_marginal, working_node.i, working_node.j, k)
                #         kth_node += 1
                #         p_bc = joint_k_B_C - total_likelihood_k
                #         np.exp(p_bc,out=p_bc)
                #         bc = self.prng.multinomial(1, np.ravel(p_bc))
                #
                #         cat_bc = np.nonzero(bc)[0][0]
                #
                #         b_cat = cat_bc // self.Q
                #         c_cat = cat_bc % self.Q
                        # print(b_cat, c_cat, cat_bc)
                expanded_nodes.append(working_node)
                node_b = Node(b_cat, working_node.i, k, self.D, self.K, parent=working_node)
                node_c = Node(c_cat, k, working_node.j, self.D, self.K, parent=working_node)
                # print(node_b, node_c)
                if node_b.d == self.D and node_b.j - node_b.i != 1:
                    print(node_b)
                    raise Exception
                if node_b.s != 0 and node_c.s != 1:
                    raise Exception("{}, {}".format(node_b, node_c))
                if node_b.is_terminal():
                    expanded_nodes.append(node_b)
                    # rules.append(Rule(node_b.k, sent[working_node.i]))
                else:
                    expanding_nodes.append(node_b)
                if node_c.is_terminal():
                    expanded_nodes.append(node_c)
                    # rules.append(Rule(node_c.k, sent[k]))
                else:
                    expanding_nodes.append(node_c)

            # logprobs.append(logprob)
            nodes_list.append(expanded_nodes)
        return nodes_list, logprobs #, rules

    # @profile
    def compute_viterbi_inside(self, sent): #sparse
        self.this_sent_len = len(sent)
        batch_size = 1
        # sents_tensor = torch.tensor(sents).to('cuda')
        sent_len = self.this_sent_len
        num_points = sent_len + 1
        self.left_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to('cuda')
        self.right_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to('cuda')

        # viterbi_chart = np.full((num_points, num_points, batch_size, self.Q), -np.inf, dtype=np.float)
        backtrack_chart = {}

        # print('lex')
        _, max_subcats = self.get_lexis_prob(sent, self.left_chart)
        self._find_max_prob_words_within_vit(sent, self.left_chart[0])
        self.right_chart[0] = self.left_chart[0]

        # kron_temp_mat = np.zeros((batch_size, self.Q ** 2), dtype=np.float)
        # kron_temp_mat_3d_view = kron_temp_mat.reshape((batch_size, self.Q, self.Q))

        for ij_diff in range(1, sent_len):
            # print('ijdiff', ij_diff)
            left_min = 0
            left_max = sent_len - ij_diff
            right_min = ij_diff
            right_max = sent_len
            height = ij_diff

            b = self.left_chart[0:height, left_min:left_max] # (all_ijdiffs, i-j, batch, Q) a square of the left chart
            c = torch.flip(self.right_chart[0:height, right_min:right_max], dims=[0])
            #
            dot_temp_mat = ( b[...,None]+c[...,None,:] ).view(height, left_max, batch_size, -1)
            # dot temp mat is all_ijdiffs, i-j, batch, Q2

            # dense
            filtered_kron_mat = self.log_G + dot_temp_mat[:, :, :, None, :]
            # filtered temp mat is all_ijdiffs, i-j, batch, Q, Q2

            filtered_kron_mat = filtered_kron_mat.permute(1,2,3,0,4).contiguous().view(left_max, batch_size, self.Q,
                                                                                   -1)
            # permute the dims to get i-j, batch, Q, all_ijdiffs * Q2

            max_kbc, argmax_kbc = torch.max(filtered_kron_mat, dim=3) # i-j, batch, Q
            ks = argmax_kbc // self.Q**2 + torch.arange(1, left_max+1)[:, None, None].to('cuda')
            bc_cats = argmax_kbc % self.Q**2
            b_cats =  bc_cats // self.Q
            c_cats = bc_cats % self.Q
            self.left_chart[height, left_min:left_max] = max_kbc
            self.right_chart[height, right_min:right_max] = max_kbc
            # print(ks.shape, b_cats.shape, c_cats.shape)
            k_b_c = torch.stack((ks, b_cats,c_cats), dim=3).to('cpu')

            backtrack_chart[ij_diff] = k_b_c
        self.right_chart = None
        return backtrack_chart, max_subcats

    # @profile
    def viterbi_backtrack(self, backtrack_chart, sent, max_cats=None):
        sent_index = 0
        nodes_list = []
        sent_len = self.this_sent_len
        topnode_pdf = self.left_chart[sent_len-1, 0]
        if max_cats is not None:
            max_cats = max_cats.squeeze()
            max_cats = max_cats.tolist()

        # draw the top node
        p_topnode = topnode_pdf + self.log_p0
        A_ll, top_A = torch.max(p_topnode, dim=-1)
        # top_A = top_A.squeeze()
        # A_ll = A_ll.squeeze()

        expanding_nodes = []
        expanded_nodes = []
        # rules = []
        assert self.this_sent_len > 0, "must call inside pass first!"

        A_cat = top_A[sent_index].item()

        assert not ( torch.isnan(A_ll[sent_index]) or torch.isinf(A_ll[sent_index]) or A_ll[sent_index].item() == 0 ), \
            'something wrong with viterbi parsing. {}'.format(A_ll[sent_index])

        # prepare the downward sampling pass
        top_node = Node(A_cat, 0, sent_len, self.D, self.K)
        if sent_len > 1:
            expanding_nodes.append(top_node)
        else:
            expanded_nodes.append(top_node)
        # rules.append(Rule(None, A_cat))
        # print(backtrack_chart)
        while expanding_nodes:
            # print(sent_len, expanding_nodes)
            working_node = expanding_nodes.pop()
            ij_diff = working_node.j - working_node.i - 1

            k_b_c = backtrack_chart[ij_diff][ working_node.i, sent_index,
                                                        working_node.cat]
            split_point, b, c = k_b_c[0].item(), k_b_c[1].item(), k_b_c[2].item()

            expanded_nodes.append(working_node)
            # print(expanding_nodes)
            node_b = Node(b, working_node.i, split_point, self.D, self.K, parent=working_node)
            node_c = Node(c, split_point, working_node.j, self.D, self.K, parent=working_node)
            # print(node_b, node_c)
            if node_b.d == self.D and node_b.j - node_b.i != 1:
                print(node_b)
                raise Exception
            if node_b.s != 0 and node_c.s != 1:
                raise Exception("{}, {}".format(node_b, node_c))
            if node_b.is_terminal():
                if max_cats is not None:
                    node_b.k = str(node_b.k) + '|' + str(max_cats[node_b.i][node_b.k])
                expanded_nodes.append(node_b)
                # rules.append(Rule(node_b.k, sent[working_node.i]))
            else:
                expanding_nodes.append(node_b)
            if node_c.is_terminal():
                if max_cats is not None:
                    node_c.k = str(node_c.k) + '|' + str(max_cats[node_c.i][node_c.k])
                expanded_nodes.append(node_c)
                # rules.append(Rule(node_c.k, sent[k]))
            else:
                expanding_nodes.append(node_c)
            # rules.append(Rule(working_node.cat, node_b.k, node_c.k))
        nodes_list.append(expanded_nodes)
        return nodes_list

    def get_lexis_prob(self, sents, left_chart):
        lexical_l1 = None
        if sents.dim() > 1:
            sent_len = len(sents[0])
        else:
            sent_len = len(sents)
            sents = sents.unsqueeze(0)
        max_subcats = None
        for i in range(0, sent_len):


            if self.embedding_type == 'none':
                wordlist = sents[:, i]
                left_chart[0, i] = torch.index_select(self.log_lexis, 1, wordlist).t()
            else:
                if self.embedding_type == 'word':
                    if i != 0: break
                    wordlist = sents.t()
                    word_embs = self.embeddings[wordlist, :] # sentlen, batch, emb
                    # word_embs = word_embs[:, None,:]
                    if isinstance(self.log_lexis, torch.distributions.Distribution):
                        word_embs = word_embs.unsqueeze(-2)
                    if self.pcfg_split is not None:
                        lexis_probs = self.log_lexis.log_prob(word_embs) + self.pcfg_split[:, 1] # sentlen, batch, prob
                    else:
                        lexis_probs = self.log_lexis.log_prob(word_embs)
                    # logging.info('max logprob: {}'.format(lexis_probs.max().cpu().item()))
                    left_chart[0] = lexis_probs

                    processed_words = set()
                    lexical_l1 = 0
                    for ii in range(wordlist.shape[0]):
                        for jj in range(wordlist.shape[1]):
                            if wordlist[ii, jj].item() not in processed_words:
                                processed_words.add(wordlist[ii, jj].item())
                                lexical_l1 = lexical_l1 + torch.norm(lexis_probs[ii, jj].exp(), 1)

                elif self.embedding_type == 'context':
                    if i != 0: break
                    sent_embs = []
                    # print('-----')
                    for index, sent_index in enumerate(self.sent_indices):
                        sent_embs.append(self.embeddings[str(sent_index)].to('cuda'))
                        # print(sent_index, sent_embs[-1].shape, sents[index].shape)
                    sent_embs = torch.stack(sent_embs, dim=0).transpose(1, 0) # sentlen, batch, emb
                    if isinstance(self.log_lexis, torch.distributions.Distribution):
                        sent_embs = sent_embs.unsqueeze(-2) # sentlen, batch, 1, emb
                    lexis_probs = self.log_lexis.log_prob(sent_embs)
                    if self.k2 is not None:
                        K, K2 = self.k2.shape
                        lexis_probs = torch.reshape(lexis_probs, list(lexis_probs.shape[:-1])+[K, K2]) + self.k2
                        max_subcats = torch.argmax(lexis_probs, dim=-1)
                        lexis_probs = torch.logsumexp(lexis_probs, -1)

                    if self.pcfg_split is not None:
                        lexis_probs = lexis_probs + self.pcfg_split[:, 1] # sentlen, batch, p

                    lexical_l1 = torch.logsumexp(lexis_probs.flatten(), 0).exp()
                    left_chart[0] = lexis_probs
        return lexical_l1, max_subcats

    @staticmethod
    def calc_Q(K=0, D=0):
        if D == -1:
            return K
        return (D+1)*(K)*2

    def find_max_prob_words(self, num_words=10):
        # with torch.no_grad():
        #     embs = self.log_lexis.unsqueeze(-2)
        #     lexis_probs = self.log_lexis.log_prob(embs)
        #     _, max_cats = torch.topk(lexis_probs, num_words, dim=0)
        #     return max_cats.detach().t()
        with torch.no_grad():
            word_indices, prob_vals = zip(*self.vocab_prob_list)
            word_indices, prob_vals = torch.tensor(word_indices).cuda(), torch.stack(prob_vals, dim=0)
            # all_data = torch.cat([word_indices.unsqueeze(1), prob_vals], dim=1)
            best_word_for_cat = {}

            max_probs, max_indices = torch.topk(prob_vals, num_words, dim=0)
            for cat in range(prob_vals.shape[1]):
                best_word_for_cat[cat] = torch.stack((word_indices[max_indices[:,cat]].float(), max_probs[:,cat]), dim=1)
            return best_word_for_cat

    def _find_max_prob_words_within_vit(self, sents, left_chart_bottom_row):
        with torch.no_grad():
            bottom_row = left_chart_bottom_row.transpose(1, 0) # batch, sentlen, p
            flatten_sents = torch.flatten(sents) # batch*sentlen
            flatten_bottom = torch.flatten(bottom_row, end_dim=-2) # batch*sentlen, p
            # vals, indices = torch.max(flatten_bottom, dim=1) # batch*sentlen, max_p
            for word_index, word in enumerate(flatten_sents):
                raw_word = word.item()
                if raw_word not in self.finished_vocab:
                    self.finished_vocab.add(raw_word)
                    self.vocab_prob_list.append((raw_word, flatten_bottom[word_index].detach()))

    def clear_vocab_prob_list(self):
        self.vocab_prob_list = []
        self.finished_vocab = set()