import numpy as np
from collections import defaultdict
from itertools import product
from scipy import sparse

def compute_Q(K=0, D=0):
    if D == -1:
        return K
    return (D + 1) * (K) * 2

def get_root_mask(D, K):
    Q = ((D+1)*2*K)
    mask = np.zeros((Q,), dtype=np.float)
    mask[:K] = 1

    return mask

def get_lex(pcfg_model, D, lex_scale=20):
    num_words = pcfg_model.len_vocab
    # print(num_words)
    lexis = []
    for nonterm in pcfg_model.unannealed_dists:
        if nonterm.symbol() != '0':
            lexis.append(pcfg_model.unannealed_dists[nonterm][-num_words:])
    lexis = np.vstack(lexis) * lex_scale
    lexis = lexis.T
    # print(lexis.shape)
    lexis = np.tile(lexis, (1, (D+1)*2))
    return lexis

# for testing only:
def compile_nonterms(gammas, K, D):
    # side -> D -> K
    Q = compute_Q(K, D)
    nonterms = np.zeros((K * 2 * (D+1), K * 2 * (D+1), K * 2 * (D+1)))
    nonterms_dict = defaultdict(float)
    lhses = []
    # print(gammas)
    for side in range(2):
        for d in range(D):
            if gammas:
                for lhs in gammas[side][d]:
                    lhs_sym = int(lhs.symbol()) - 1
                    if lhs.symbol() != '0':
                        for rhs, val in gammas[side][d][lhs].items():
                            rhs_1, rhs_2 = int(rhs[0].symbol()) - 1, int(rhs[1].symbol()) - 1
                            lhs_mat = side * (K * (D+1)) + d * K + lhs_sym
                            if side == 1:
                                depth = d + 1
                            else:
                                depth = d
                            rhs_1_mat = 0 * (K * (D+1))+ (depth * K) + rhs_1
                            rhs_2_mat = 1 * (K * (D+1))+ (d * K) + rhs_2
                            # print(lhs, rhs, val,  lhs_mat, rhs_1_mat, rhs_2_mat, side, d)
                            # print(lhs, rhs, lhs, lhs_mat, rhs_1_mat, rhs_2_mat, side, d, val)
                            nonterms[lhs_mat, rhs_1_mat, rhs_2_mat] = val
                            nonterms_dict[(side, d, lhs_sym), (0, depth, rhs_1), (1, d, rhs_2)] = val
            else:
                for k in range(K+1):
                    lhs_sym = k - 1
                    lhses.append(lhs_sym)
                    if k != 0:
                        for rhs_1, rhs_2 in product(range(K), range(K)):
                            lhs_mat = side * (K * (D+1)) + d * K + lhs_sym
                            if side == 1:
                                depth = d + 1
                            else:
                                depth = d
                            rhs_1_mat = 0 * (K * (D+1))+ (depth * K) + rhs_1
                            rhs_2_mat = 1 * (K * (D+1))+ (d * K) + rhs_2
                            # print(lhs, rhs, val,  lhs_mat, rhs_1_mat, rhs_2_mat, side, d)
                            # print(lhs, rhs, lhs, lhs_mat, rhs_1_mat, rhs_2_mat, side, d, val)
                            val = np.random.random()
                            nonterms[lhs_mat, rhs_1_mat, rhs_2_mat] = val
                            nonterms_dict[(side, d, lhs_sym), (0, depth, rhs_1), (1, d, rhs_2)] = val
    for lhs in lhses:
        nonterms[lhs_mat] /= np.sum(nonterms[lhs_mat])
    nonterms = np.reshape(nonterms, (K * 2 * (D+1), -1)).astype(np.float)
    sp_nonterms = sparse.csr_matrix(nonterms)
    return sp_nonterms, nonterms_dict

def compute_hd_size(D, K):
    G_D_size = ((D+1)* (K) * 2) ** 3
    G_D_actual = 2 * (D) * (K) ** 3
    HD_size_dense = G_D_size * 32 / 8 / 1024 / 1024
    HD_size_sparse = G_D_actual * 32 / 8 / 1024 / 1024 * 3
    print('Dense size: ',G_D_size, ' Dense HD size: ', HD_size_dense, ' Sparse size: ', G_D_actual,\
                          ' Sparse HD size:', HD_size_sparse)
    return G_D_size, HD_size_dense, G_D_actual, HD_size_sparse

def compute_ijk_vecs_num(sent_len):
    return int((sent_len - 1) * sent_len * 0.5 * (sent_len + 1) / 3)

def compute_ij_vecs_num(sent_len):
    return (sent_len - 1) * sent_len * 0.5

def compute_delta_num(sent_len, delta):
    return (sent_len + 1 - delta) * (delta - 1)

def compute_incr_sum(num_points, i):
    return int((num_points - 1 + i) * abs(num_points - i) / 2)

def compute_decr_sum(j):
    return int((j-1) * j / 2)

def compute_tri_num(n):
    return (n + 1) * n / 2