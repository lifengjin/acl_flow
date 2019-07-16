import os.path
import time
from .cky_utils import compute_Q
import bidict
from .metrics import *
from .metric_groups import *
import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
from .matrix_initialization import matrix_initialization
import logging

class _PCFG_model(nn.Module):
    def __init__(self, K, D, len_vocab, num_sents, num_words, alpha, log_dir='.',
                 word_dict_file=None, hyperparam_collector=None,
                 init_method=None, **kwargs):
        super().__init__()
        self.hyp_col = hyperparam_collector
        self.K = K
        self.K2 = K**2
        self.len_vocab = len_vocab
        self.alpha = alpha
        self.nonterm_expansion_size = self.K**2
        self.term_expansion_size = self.len_vocab
        self.num_sents = num_sents
        self.num_words = num_words
        self.nonterms = list(range(self.K))
        self.iter_index = 0
        self.log_dir = log_dir
        self.log_mode = 'w'
        self.hypparams_log_path = os.path.join(log_dir, 'running_status.txt')
        self.pcfg_split_log_path = os.path.join(log_dir, 'pcfg_split_probs.txt')
        self.cat_word_log_path = os.path.join(log_dir, 'cat_best_word.txt')
        self.word_dict = self._read_word_dict_file(word_dict_file)
        self.possible_params = ['_p0', '_emission', '_emission_mu', '_emission_sigma', '_expansion', '_pcfg_split']

    def set_log_mode(self, mode):
        self.log_mode = mode  # decides whether append to log or restart log

    def start_logging(self):

        self.pcfg_split_log = open(self.pcfg_split_log_path, self.log_mode, encoding='utf8')

        self.hypparam_log = open(self.hypparams_log_path, self.log_mode, encoding='utf8')
        self.cat_word_log = open(self.cat_word_log_path, self.log_mode, encoding='utf8')

        if self.log_mode == 'w':
            self.hypparam_log.write('\t'.join(SimpleIterMetrics.metrics) + '\n')

    def end_logging(self):
        self.hypparam_log.close()
        self.counts_log.close()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['counts_log']
        del state['hypparam_log']
        return state

    def _translate_model_to_pcfg(self):
        pcfg = {x: {} for x in range(self.K)}
        for parent in pcfg:
            # print(parent)
            dist = torch.cat([self.expansion[parent], self.emission[parent]], dim=0)
            for index, value in enumerate(dist):
                if index < self.K2:
                    rhs = (index // self.K, index % self.K)
                else:
                    rhs = index - self.K2
                pcfg[parent][rhs] = value
        return pcfg

    def get_current_pcfg(self):
        pcfg = self._translate_model_to_pcfg()
        return pcfg, self.p0

    def save(self, dnn):
        t0 = time.time()
        log_dir = self.log_dir
        save_model_fn = 'pcfg_model_' + str(self.iter_index) + '.pkl'
        past_three = os.path.join(log_dir, 'pcfg_model_' + str(self.iter_index - 3) + '.pkl')
        if os.path.exists(past_three): # and (self.this_metric.iter_index - 3) % 100:
            os.remove(past_three)
        this_f = os.path.join(log_dir, save_model_fn)
        with open(this_f, 'wb') as ffh:#, open(this_pcfg_dict_f, 'wb') as dfh:
            torch.save((self,dnn), ffh)

        t1 = time.time()
        logging.info('Dumping out the pcfg model takes {:.3f} secs.'.format(t1 - t0))

    def _read_word_dict_file(self, word_dict_file):
        f = open(word_dict_file, 'r', encoding='utf-8')
        word_dict = bidict.bidict()
        for line in f:
            (word, index) = line.rstrip().split(" ")
            word_dict[int(index)] = word
        return word_dict

    def save_grammar_params(self, iter_index, batch_index, only_mu=False):
        params = set(['_expansion', '_pcfg_split', '_emission_mu', '_emission_sigma', '_p0', '_emission', '_grammar_val'])
        param_dict = {}
        for param_name in params:
            if only_mu and param_name != '_emission_mu': continue
            if hasattr(self, param_name):
                if isinstance(getattr(self, param_name), torch.Tensor):
                    param_dict[param_name] = getattr(self, param_name).data.cpu()
                elif isinstance(getattr(self, param_name), torch.nn.ParameterDict):
                    param_dict[param_name] = {}
                    self_param_dict = getattr(self, param_name)
                    for key in self_param_dict:
                        param_dict[param_name][key] = self_param_dict[key].data.cpu()
        if hasattr(self, 'neural_exp') and only_mu:
            param_dict['neural_exp'] = [x.cpu() for x in self.neural_exp]
        # if self.flow_type is not None:
        #     param_dict['flow'] = self.flow.state_dict()
        if only_mu: mu_marker = 'mu_'
        else: mu_marker = ''
        save_model_fn = 'pcfg_model_' + mu_marker + str(iter_index) + '_' + str(batch_index) + '.trh'
        model_path = 'model_params'
        if not os.path.exists(os.path.join(self.log_dir, model_path)):
            os.mkdir(os.path.join(self.log_dir, model_path))
        save_model_path = os.path.join(self.log_dir, model_path, save_model_fn)
        torch.save(param_dict, save_model_path)
        if hasattr(self, 'neural_exp') and only_mu:
            param_dict['neural_exp'] = [x.cuda() for x in self.neural_exp]

class PCFG_model(_PCFG_model):
    def __init__(self, K, D, len_vocab, num_sents, num_words, alpha, log_dir='.',
                 word_dict_file=None, saved_params=None, hyperparam_collector=None,
                 init_method=None, init_branching_tendency=None, ev_seqs=None):
        super().__init__(K, D, len_vocab, num_sents, num_words, alpha, log_dir=log_dir,
                         word_dict_file=word_dict_file, hyperparam_collector=hyperparam_collector, init_method=init_method)
        self.init_model(alpha, saved_params, init_method=init_method,
                        init_branching_tendency=init_branching_tendency, ev_seqs=ev_seqs)
        self.embeddings = None

    def init_model(self, alpha, saved_params=None, init_method='uniform', init_branching_tendency=None
                   , ev_seqs=None):
        assert saved_params is None or len(saved_params) == 4
        logging.info('Initialization with seed {}'.format(torch.initial_seed()))
        if saved_params is None:
            expansion_alpha = torch.full((self.K, self.nonterm_expansion_size), alpha)
            emission_alpha = torch.full((self.K, self.term_expansion_size), alpha)
            p0_alpha_tensor = torch.full((self.K,), alpha)
            pcfg_split_alpha = torch.full((self.K, 2), alpha)
            self.hyp_col.add_param('Model init', init_method)
            if init_method == 'uniform':
                self._expansion = torch.nn.init.uniform_(expansion_alpha, 1e-2, 1e-1)
                self._emission = torch.nn.init.uniform_(emission_alpha, 1e-2, 1e-1)
                self._p0 = torch.nn.init.uniform_(p0_alpha_tensor, 1e-2, 1e-1)
                self._pcfg_split = torch.nn.init.uniform_(pcfg_split_alpha, 1e-2, 1e-1)
            elif init_method == 'dirichlet':
                self._expansion = torch.distributions.dirichlet.Dirichlet(expansion_alpha).sample().log()
                self._emission = torch.distributions.dirichlet.Dirichlet(emission_alpha).sample().log()
                self._p0 = torch.distributions.dirichlet.Dirichlet(p0_alpha_tensor).sample().log()

                self._pcfg_split = torch.distributions.dirichlet.Dirichlet(pcfg_split_alpha).sample().log() # 0: expansion 1:
            elif init_method == 'dirichlet_whole':
                grammar_alpha = torch.cat([expansion_alpha, emission_alpha], dim=1)
                _grammar = torch.distributions.Dirichlet(grammar_alpha).sample().log()
                self._p0 = torch.distributions.dirichlet.Dirichlet(p0_alpha_tensor).sample().log()

                if init_branching_tendency is not None:
                    _grammar, self._p0 = matrix_initialization(_grammar, self._p0, ev_seqs, self.word_dict,
                                                               branching='right')

                _expansion = _grammar[:, :self.nonterm_expansion_size]
                _emission = _grammar[:, self.nonterm_expansion_size:]
                self._pcfg_split = torch.cat([torch.logsumexp(_expansion, dim=1, keepdim=True), torch.logsumexp(_emission,
                                                                                                             dim=1,
                                                                                                             keepdim=True
                                                                                                             ) ], dim=1)
                self._expansion = _expansion - self._pcfg_split[:, 0].unsqueeze(1)
                self._emission = _emission - self._pcfg_split[:, 1].unsqueeze(1)

        else:
            for param_name in saved_params:

                if param_name in self.possible_params:
                    setattr(self, param_name, saved_params[param_name])

        # self.hooks = []
        self._pcfg_split = nn.Parameter(data=self._pcfg_split)
        # self.hooks.append(self._pcfg_split.register_hook(add_gauss_noise))
        self._p0 = nn.Parameter(data=self._p0)
        # self.hooks.append(self._p0.register_hook(add_gauss_noise))
        self._emission = nn.Parameter(data=self._emission)
        # self.hooks.append(self._emission.register_hook(add_gauss_noise))
        self._expansion = nn.Parameter(data=self._expansion)
        # self.hooks.append(self._expansion.register_hook(add_gauss_noise))
        self.normalization_transform = F.log_softmax
        # self.normalization_transform = relu_normalize

    def normalize_models(self):
        self.pcfg_split = self.normalization_transform(self._pcfg_split, dim=1)
        self.emission = self.normalization_transform(self._emission, dim=1) + self.pcfg_split[:, 1][..., None]
        self.expansion = self.normalization_transform(self._expansion, dim=1) + self.pcfg_split[:, 0][..., None]
        self.p0 = self.normalization_transform(self._p0, dim=0)

    def normalized_parameters(self):
        pcfg_split = self.normalization_transform(self._pcfg_split, dim=1)
        emission = self.normalization_transform(self._emission, dim=1) + self.pcfg_split[:, 1][..., None]
        expansion = self.normalization_transform(self._expansion, dim=1) + self.pcfg_split[:, 0][..., None]
        p0 = self.normalization_transform(self._p0, dim=0)
        # return [pcfg_split, emission, expansion, p0]
        return [emission, expansion, p0]

    def sparsity(self, eps=1e-4):
        with torch.no_grad():
            pcfg_split = F.log_softmax(self._pcfg_split, dim=1)
            expansion = F.log_softmax(self._expansion, dim=1)
            expansion = self.expansion + pcfg_split[:, 0][..., None]
            emission = F.log_softmax(self._emission, dim=1) + pcfg_split[:, 1][..., None]
            sparsity_measure = (expansion.exp() > eps).sum() + (emission.exp() > eps).sum()
        return sparsity_measure.item() / (expansion.numel() + emission.numel())

class PCFG_model_bayes(_PCFG_model):
    def __init__(self, K, D, len_vocab, num_sents, num_words, alpha, log_dir='.',
                 word_dict_file=None, full_initialization_tensors=None, hyperparam_collector=None,
                 init_method=None):
        super().__init__(K, D, len_vocab, num_sents, num_words, alpha, log_dir=log_dir,
                 word_dict_file=word_dict_file, full_initialization_tensors=full_initialization_tensors,
                                         hyperparam_collector=hyperparam_collector, init_method=init_method)
        self.init_model(alpha, full_initialization_tensors, init_method=init_method)
        self.preterm_nodes = self.num_words
        self.nonterm_nodes = self.num_words - self.num_sents
        self.register_backward_hook(self.not_first_iter)

    def not_first_iter(self, x, dx):
        self.first_iter = False

    def init_model(self, alpha, full_initialization_tensors=None, init_method='uniform'):
        assert full_initialization_tensors is None or len(full_initialization_tensors) == 2
        if full_initialization_tensors is None:
            self.grammar_alpha = torch.full((self.K, self.nonterm_expansion_size+self.term_expansion_size), alpha)
            self.p0_alpha = torch.full((self.K,), 1)
            self.hyp_col.add_param('Model init', init_method)
            if init_method == 'uniform':
                _grammar = torch.nn.init.uniform_(grammar_alpha)
                _grammar = _grammar / _grammar.sum(dim=1, keepdim=True)
                _p0 = torch.nn.init.uniform_(p0_alpha_tensor)
                _p0 = _p0 / _p0.sum()
            elif init_method == 'dirichlet':
                _grammar = torch.distributions.dirichlet.Dirichlet(self.grammar_alpha).sample()
                _p0 = torch.distributions.dirichlet.Dirichlet(self.p0_alpha).sample()
        else:
            _p0, _grammar = full_initialization_tensors
            self.grammar_alpha = torch.full((self.K, self.nonterm_expansion_size+self.term_expansion_size), alpha)
            self.p0_alpha = torch.full((self.K,), 1)

        _p0.uniform_(0, 1e-1)
        self._p0 = nn.Parameter(data=_p0)
        # self.hooks.append(self._p0.register_hook(add_gauss_noise))
        _grammar[:, :self.nonterm_expansion_size] /= _grammar[:,:self.nonterm_expansion_size].sum()
        _grammar[:, self.nonterm_expansion_size:] /= _grammar[:, self.nonterm_expansion_size:].sum()

        _grammar.uniform_(0, 1e-1)

        self._grammar = nn.Parameter(data=_grammar)
        self.count_transform = torch.distributions.transform_to(torch.distributions.constraints.simplex)
        # self.count_transform = relu_normalize
        self.first_iter = True


    def normalize_models(self):
        expansion_part_of_grammar = self._grammar[:, :self.nonterm_expansion_size]
        emission_part_of_grammar = self._grammar[:, self.nonterm_expansion_size:]
        expansion_part_of_grammar_size = expansion_part_of_grammar.shape
        emission_part_of_grammar_size = emission_part_of_grammar.shape

        simplex_expansion = self.count_transform(expansion_part_of_grammar.flatten()).reshape(expansion_part_of_grammar_size)
        simplex_emission = self.count_transform(emission_part_of_grammar.flatten()).reshape(emission_part_of_grammar_size)
        simplex_p0 = self.count_transform(self._p0)
        # if not self.first_iter:
        #     simplex_expansion = self.count_transform(self._grammar[:, :self.nonterm_expansion_size])
        #     simplex_emission = self.count_transform(self._grammar[:, self.nonterm_expansion_size:])
        # else:
        #     simplex_expansion = self._grammar[:, :self.nonterm_expansion_size]
        #     simplex_emission = self._grammar[:, self.nonterm_expansion_size:]
        self.expansion_pseudo_counts = simplex_expansion * self.nonterm_nodes + self.alpha
        self.emission_pseudo_counts = simplex_emission * self.preterm_nodes + self.alpha
        self.pseudo_grammar_counts = torch.cat([self.expansion_pseudo_counts, self.emission_pseudo_counts], dim=1)
        self.pseudo_grammar_counts = self.pseudo_grammar_counts.cpu()
        self.grammar = torch.distributions.Dirichlet(self.pseudo_grammar_counts).rsample()
        self.grammar = self.grammar.cuda().log()
        self.expansion = self.grammar[:, :self.nonterm_expansion_size]
        self.emission = self.grammar[:, self.nonterm_expansion_size:]
        self.p0_pseudo_counts = simplex_p0 * self.num_sents + 1
        self.p0_pseudo_counts = self.p0_pseudo_counts.cpu()
        self.p0 = torch.distributions.Dirichlet(self.p0_pseudo_counts).rsample()
        self.p0 = self.p0.cuda().log()
        self.pcfg_split = torch.cat([torch.logsumexp(self.expansion, dim=1,keepdim=True), torch.logsumexp(
                    self.emission, dim=1, keepdim=True)], dim=1)

    def normalized_parameters(self):
        return [self._grammar, self._p0]

    def soft_count_loss(self):
        expansion_part_of_grammar = self._grammar[:, :self.nonterm_expansion_size]
        emission_part_of_grammar = self._grammar[:, self.nonterm_expansion_size:]
        expansion_part_of_grammar_size = expansion_part_of_grammar.shape
        emission_part_of_grammar_size = emission_part_of_grammar.shape

        simplex_expansion = self.count_transform(expansion_part_of_grammar.flatten()).reshape(expansion_part_of_grammar_size)
        simplex_emission = self.count_transform(emission_part_of_grammar.flatten()).reshape(emission_part_of_grammar_size)

        simplex_p0 = self.count_transform(self._p0)

        no_alpha_expansion_counts = simplex_expansion.sum()
        no_alpha_emission_counts = simplex_emission.sum()
        no_alpha_p0_counts = simplex_p0.sum()
        quadratic_loss = (no_alpha_emission_counts - 1) ** 2 + (no_alpha_expansion_counts -
                           1)** 2 + (no_alpha_p0_counts - 1) ** 2
        # quadratic_loss = (no_alpha_emission_counts - self.preterm_nodes) ** 2 + (no_alpha_expansion_counts -
        #                    self.nonterm_nodes)** 2 + (no_alpha_p0_counts - self.num_sents) ** 2
        return quadratic_loss


def add_gauss_noise(grad, scale=1):
    # print(grad.mean(), grad.std())
    # print(scale)
    new_grad = grad + torch.zeros_like(grad).normal_(mean=grad.mean(), std=grad.std()) / scale
    return new_grad

def relu_normalize(inputs, dim=-1):
    inputs = torch.nn.functional.relu(inputs) + 1e-10
    return (inputs   / inputs.sum(dim=dim, keepdim=True)).log()
