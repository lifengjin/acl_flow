import os.path
import time, random, math
from .cky_utils import compute_Q
import bidict
from .metrics import *
from .metric_groups import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from .flow import NICETrans, FLOWDist, RealNVP, OriginalNICETrans
from .flows_ext import FLOWextDist
from .pcfg_model import _PCFG_model
EPSILON = 1e-200

def normalize_a_tensor(tensor):
    return tensor / (np.sum(tensor, axis=-1, keepdims=True) + 1e-20)  # to supress zero division warning

class PCFG_model_Gauss(_PCFG_model):
    def __init__(self, K, D, len_vocab, num_sents, num_words, alpha, log_dir='.',
                 word_dict_file=None, saved_params=None, embeddings=None, flow_type=None,
                 tune_scale=False, hyperparam_collector=None, init_method=None,
                 embedding_type='word', num_flow_blocks=8, std=None, tune_embeddings=False):
        super().__init__(K, D, len_vocab, num_sents, num_words, alpha, log_dir=log_dir,
                 word_dict_file=word_dict_file, init_method=init_method)

        self.flow_type = flow_type
        self.tune_scale = tune_scale
        self.hyp_col = hyperparam_collector
        # self.init_counts()
        assert embeddings is not None, "must have embeddings!"
        # self.embeddings = None
        if embedding_type == 'word':
            self.register_buffer('embeddings', embeddings)
            self.feature_size = self.embeddings.shape[1]
        elif embedding_type == 'context':
            self.embeddings = {}
            for key in embeddings:
                self.embeddings[str(key)] = embeddings[key].to('cuda')
            self.feature_size = self.embeddings['0'].shape[1]
        if self.flow_type:
            assert self.feature_size % 2 == 0, 'embedding size must be dividable by 2!'
        hyperparam_collector.add_param('Embedding size', self.feature_size)
        self.init_model(alpha, saved_params=saved_params, init_method=init_method,
                        embedding_type=embedding_type, num_flow_blocks=num_flow_blocks, std=std, tune_embeddings=tune_embeddings)

    def init_model(self, alpha, saved_params=None, init_method='dirichlet',
                   embedding_type='word', num_flow_blocks=8, std=None, tune_embeddings=False):
        expansion_alpha = torch.full((self.K, self.nonterm_expansion_size), alpha)
        p0_alpha_tensor = torch.full((self.K,), alpha)
        pcfg_split_alpha = torch.full((self.K, 2), alpha)
        logging.info('Initialization with seed {}'.format(torch.initial_seed()))

        if init_method == 'uniform':
            self._expansion = torch.nn.init.kaiming_uniform_(expansion_alpha, a=math.sqrt(5))
            self._p0 = torch.nn.init.kaiming_uniform_(p0_alpha_tensor.unsqueeze(0), a=math.sqrt(5)).squeeze()
            self._pcfg_split = torch.nn.init.kaiming_uniform_(pcfg_split_alpha, a=math.sqrt(5))
        elif init_method == 'dirichlet':
            self._expansion = torch.distributions.dirichlet.Dirichlet(expansion_alpha).sample().log()
            self._p0 = torch.distributions.dirichlet.Dirichlet(p0_alpha_tensor).sample().log()
            self._pcfg_split = torch.distributions.dirichlet.Dirichlet(pcfg_split_alpha).sample().log()  # 0: expansion 1:
        else:
            raise ValueError('unknown init method')

        if embedding_type == 'word':
            counter = 5000
            self.hyp_col.add_param('Word embedding counter', counter)
            indices = random.sample(range(0, self.embeddings.shape[0]), counter)
            embeddings_for_init = self.embeddings[indices, :]
        elif embedding_type == 'context': # randomly sample a few sentences for calculation of means and vars
            # for key in self.embeddings:
            #     self.embeddings[key] = self.embeddings[key].to('cuda')
            counter = 5000
            self.hyp_col.add_param('Context embedding counter', counter)
            embs_tmp = []
            sampled_keys = random.sample(self.embeddings.keys(), counter)
            for key in sampled_keys:
                embs_tmp.append(self.embeddings[str(key)])
            embeddings_for_init = torch.cat(embs_tmp, dim=0)
        if std is None:
            logging.info('STD is None. Calculate STD from embeddings.')
            _emission_sigma = torch.std(embeddings_for_init, dim=0).cpu() # / 2  # .expand(__emission_mu.size())
        else:
            logging.info('STD is not None. Use STD from file.')
            _emission_sigma = std
        _emission_mu = torch.full((self.K, self.feature_size), alpha).normal_().mul_(0.04)
        _emission_mu += embeddings_for_init.mean(dim=0, keepdim=True).cpu()

        # self.hooks = []
        self._pcfg_split = nn.Parameter(data=self._pcfg_split.detach())
        # self.hooks.append(self._pcfg_split.register_hook(add_gauss_noise))
        self._p0 = nn.Parameter(data=self._p0.detach())
        # self.hooks.append(self._p0.register_hook(add_gauss_noise))
        self._emission_mu = nn.Parameter(data=_emission_mu.detach())
        self.register_buffer('_emission_sigma', _emission_sigma)
        # self.hooks.append(self._emission.register_hook(add_gauss_noise))
        self._expansion = nn.Parameter(data=self._expansion.detach())
        # self.hooks.append(self._expansion.register_hook(add_gauss_noise))
        self._emission_sigma = self._emission_sigma.repeat(self.K, 1).to('cuda') # log?

        if saved_params is not None:
            self.load_state_dict(saved_params)

        self.base_params = [self._pcfg_split, self._p0, self._expansion]

        if self.flow_type is not None and self.flow_type == 'nice':
            self.flow = NICETrans(num_flow_blocks, 1, self.feature_size, self.feature_size)
            self.emission = FLOWDist(self.flow, self._emission_mu, self._emission_sigma)
        # elif self.flow_type is not None and self.flow_type == 'nvp':
        #     self.flow = RealNVP(num_flow_blocks, 1, self.feature_size, self.feature_size)
        elif self.flow_type is not None and (self.flow_type == 'realnvp' or self.flow_type == 'maf'):
            self.emission = FLOWextDist(self._emission_mu, self._emission_sigma, model_type=self.flow_type,
                                        num_blocks=num_flow_blocks, num_inputs=self.feature_size,
                                        num_hidden=self.feature_size)
        else:
            self.emission = torch.distributions.Independent(torch.distributions.Normal(self._emission_mu,
                                                                                       self._emission_sigma), 1)

        if self.flow_type is not None and self.flow_type == 'nice':
            self.emit_params = [self._emission_mu] + list(self.flow.parameters())
        elif self.flow_type is not None and self.flow_type != 'nice':
            self.emit_params = [self._emission_mu] + list(self.emission.model.parameters())
        else:
            self.emit_params = [self._emission_mu]

        if tune_embeddings:
            if embedding_type == 'word':
                self.embeddings = nn.Parameter(data=self.embeddings)
            elif embedding_type == 'context':
                self.embeddings = nn.ParameterDict({x:nn.Parameter(self.embeddings[x]) for x in self.embeddings})

            self.emit_params += list(self.embeddings.values())
        logging.info(self.emission)

    def normalize_models(self):

        self.pcfg_split = F.log_softmax(self._pcfg_split, dim=1)
        self.expansion = F.log_softmax(self._expansion, dim=1)
        self.expansion = self.expansion + self.pcfg_split[:, 0][..., None]
        self.p0 = F.log_softmax(self._p0, dim=0)

    def normalized_parameters(self):
        expansion = F.log_softmax(self._expansion, dim=1)
        p0 = F.log_softmax(self._p0, dim=0)

        return [ expansion, p0] # self.pcfg_split

    def sparsity(self, eps=1e-4):
        with torch.no_grad():
            pcfg_split = F.log_softmax(self._pcfg_split, dim=1)
            expansion = F.log_softmax(self._expansion, dim=1)
            expansion = self.expansion + self.pcfg_split[:, 0][..., None]
            sparsity_measure = (expansion.exp() > eps).sum()
        return sparsity_measure.item() / expansion.numel()

    def gaussian_mean_distance_loss(self):

        distances = 0
        for x in range(self._emission_mu.shape[0]):
            for y in range(self._emission_mu.shape[0]):
                if x < y:
                    v1 = self._emission_mu[x].unsqueeze(0)
                    v2 = self._emission_mu[y].unsqueeze(0)
                    distance = torch.nn.functional.pairwise_distance(v1, v2)
                    distances = distances + distance
        loss = - distances
        return loss

    def branching_total_loss(self):
        loss = 0
        expansion = F.log_softmax(self._expansion, dim=1)

        for category_1 in range(expansion.shape[0]):
            left_branching_rule_indices = [category_1 * self.K + i for i in range(self.K)]
            right_branching_rule_indices = [category_1 + i* self.K for i in range(self.K)]
            loss = loss + expansion[category_1, left_branching_rule_indices].logsumexp(0)
            loss = loss + expansion[category_1, right_branching_rule_indices].logsumexp(0)
        return - loss

    def branching_difference_loss(self):
        loss = 0
        expansion = F.log_softmax(self._expansion, dim=1)
        for category_1 in range(expansion.shape[0]):
            left_branching_rule_indices = [category_1 * self.K + i for i in range(self.K)]
            right_branching_rule_indices = [category_1 + i* self.K for i in range(self.K)]
            loss = ( expansion[category_1, left_branching_rule_indices].logsumexp(0) - \
                   expansion[category_1, right_branching_rule_indices].logsumexp(0) ).abs() + loss

        return - loss

    def save_gauss_params(self, iter_index, batch_index):
        assert self.flow_type is None, 'can only save Gaussian emission params with this function'
        # if self.flow_type is not None:
        #     param_dict['flow'] = self.flow.state_dict()
        save_model_fn = 'pcfg_model_' + str(iter_index) + '_' + str(batch_index) + '.trh'
        model_path = 'model_params'
        if not os.path.exists(os.path.join(self.log_dir, model_path)):
            os.mkdir(os.path.join(self.log_dir, model_path))
        save_model_path = os.path.join(self.log_dir, model_path, save_model_fn)
        torch.save(self.state_dict(), save_model_path)

class PCFG_model_Gauss_Neuralexp_Flowemit(_PCFG_model):
    def __init__(self, K, D, len_vocab, num_sents, num_words, alpha, log_dir='.',
                 word_dict_file=None, saved_params=None, embeddings=None, flow_type=None,
                 tune_scale=False, hyperparam_collector=None, init_method=None,
                 embedding_type='word', num_flow_blocks=8, std=None, tune_embeddings=False, exp_type='rnn',
                 drop_out_rate=0, num_rnn_layers=1, bidirectional_flag=False, full_neural=True):
        super().__init__(K, D, len_vocab, num_sents, num_words, alpha, log_dir=log_dir,
                 word_dict_file=word_dict_file, init_method=init_method)

        self.flow_type = flow_type
        self.tune_scale = tune_scale
        self.hyp_col = hyperparam_collector
        # self.init_counts()
        assert embeddings is not None, "must have embeddings!"
        # self.embeddings = None
        if embedding_type == 'word':
            self.register_buffer('embeddings', embeddings)
            self.feature_size = self.embeddings.shape[1]
        elif embedding_type == 'context':
            self.embeddings = {}
            for key in embeddings:
                self.embeddings[str(key)] = embeddings[key].to('cuda')
            self.feature_size = self.embeddings['0'].shape[1]
        if self.flow_type:
            assert self.feature_size % 2 == 0, 'embedding size must be dividable by 2!'
        hyperparam_collector.add_param('Embedding size', self.feature_size)
        self.exp_type = exp_type
        self.full_neural = full_neural
        hyperparam_collector.add_param('Full neural?', self.full_neural)
        self.init_model(alpha, saved_params=saved_params, init_method=init_method,
                        embedding_type=embedding_type, num_flow_blocks=num_flow_blocks, std=std,
                        tune_embeddings=tune_embeddings, drop_out_rate=drop_out_rate, num_rnn_layers=num_rnn_layers,
                        bidirectional_flag=bidirectional_flag, exp_type=exp_type, full_neural=full_neural)

    def init_model(self, alpha, saved_params=None, init_method='uniform',
                   embedding_type='word', num_flow_blocks=8, std=None, tune_embeddings=False,
                   drop_out_rate=0, num_rnn_layers=1, bidirectional_flag=False, exp_type='rnn',
                   full_neural=False):
        # expansion_alpha = torch.full((self.K, self.nonterm_expansion_size), alpha)
        p0_alpha_tensor = torch.full((self.K,), alpha)
        pcfg_split_alpha = torch.full((self.K, 2), alpha)
        logging.info('Initialization with seed {}'.format(torch.initial_seed()))
        if saved_params is not None:
            for param_name in saved_params:
                if param_name in self.possible_params:
                    setattr(self, param_name, saved_params[param_name])
        else:
            if not full_neural:
                if init_method == 'uniform':
                    # self._expansion = torch.nn.init.kaiming_uniform_(expansion_alpha, a=math.sqrt(5))
                    self._p0 = torch.nn.init.kaiming_uniform_(p0_alpha_tensor.unsqueeze(0), a=math.sqrt(5)).squeeze()
                    self._pcfg_split = torch.nn.init.kaiming_uniform_(pcfg_split_alpha, a=math.sqrt(5))
                elif init_method == 'dirichlet':
                    # self._expansion = torch.distributions.dirichlet.Dirichlet(expansion_alpha).sample().log()
                    self._p0 = torch.distributions.dirichlet.Dirichlet(p0_alpha_tensor).sample().log()
                    self._pcfg_split = torch.distributions.dirichlet.Dirichlet(pcfg_split_alpha).sample().log()  # 0: expansion 1:
                else:
                    raise ValueError('unknown init method')
            else:
                self._p0 = torch.nn.Linear(self.feature_size, 1)

                self._pcfg_split = torch.nn.Sequential(torch.nn.Linear(self.feature_size, self.feature_size),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(self.feature_size, 2),
                                                       torch.nn.LogSoftmax())

        if embedding_type == 'word':
            counter = 5000
            self.hyp_col.add_param('Word embedding counter', counter)
            indices = random.sample(range(0, self.embeddings.shape[0]), counter)
            embeddings_for_init = self.embeddings[indices, :]
        elif embedding_type == 'context': # randomly sample a few sentences for calculation of means and vars
            # for key in self.embeddings:
            #     self.embeddings[key] = self.embeddings[key].to('cuda')
            counter = 5000
            self.hyp_col.add_param('Context embedding counter', counter)
            embs_tmp = []
            sampled_keys = random.sample(self.embeddings.keys(), counter)
            for key in sampled_keys:
                embs_tmp.append(self.embeddings[str(key)])
            embeddings_for_init = torch.cat(embs_tmp, dim=0)
        if std is None:
            logging.info('STD is None. Calculate STD from embeddings.')
            _emission_sigma = torch.std(embeddings_for_init, dim=0).cpu() # / 2  # .expand(__emission_mu.size())
        else:
            logging.info('STD is not None. Use STD from file.')
            _emission_sigma = std
        _emission_mu = torch.full((self.K, self.feature_size), alpha).normal_().mul_(0.04)
        _emission_mu += embeddings_for_init.mean(dim=0, keepdim=True).cpu()

        # self.hooks = []
        if not full_neural:
            self._pcfg_split = nn.Parameter(data=self._pcfg_split.detach())
            self._p0 = nn.Parameter(data=self._p0.detach())
            self.base_params = [self._pcfg_split, self._p0]
        else:
            self.base_params = list(self._pcfg_split.parameters()) + list(self._p0.parameters())

        self._emission_mu = nn.Parameter(data=_emission_mu.detach())
        self.register_buffer('_emission_sigma', _emission_sigma)
        # self.hooks.append(self._emission.register_hook(add_gauss_noise))
        # self._expansion = nn.Parameter(data=self._expansion.detach())
        # self.hooks.append(self._expansion.register_hook(add_gauss_noise))
        self._emission_sigma = self._emission_sigma.repeat(self.K, 1).to('cuda') # log?
        # self.min_std = self._emission_sigma.min()
        # self.min_std_multiplied = self.min_std * 1e-1
        if self.tune_scale:
            self._emission_sigma = self._emission_sigma + torch.zeros_like(self._emission_sigma).normal_() * \
                                   self._emission_sigma.mean()
            self._emission_sigma = nn.Parameter(self._emission_sigma)

        # self._emission_sigma = nn.Parameter(data=self._emission_sigma.detach())
        # self.emission = torch.distributions.MultivariateNormal(self._emission_mu, scale_tril=emission_sigma)

        if self.flow_type is not None and self.flow_type == 'nice':
            self.flow = NICETrans(num_flow_blocks, 1, self.feature_size, self.feature_size)
            self.emission = FLOWDist(self.flow, self._emission_mu, self._emission_sigma)
        # elif self.flow_type is not None and self.flow_type == 'nvp':
        #     self.flow = RealNVP(num_flow_blocks, 1, self.feature_size, self.feature_size)
        elif self.flow_type is not None and (self.flow_type == 'realnvp' or self.flow_type == 'maf'):
            self.emission = FLOWextDist(self._emission_mu, self._emission_sigma, model_type=self.flow_type,
                                        num_blocks=num_flow_blocks, num_inputs=self.feature_size,
                                        num_hidden=self.feature_size)
        else:
            self.emission = torch.distributions.Independent(torch.distributions.Normal(self._emission_mu,
                                                                                       self._emission_sigma), 1)

        if self.flow_type is not None and self.flow_type == 'nice':
            self.emit_params = [self._emission_mu] + list(self.flow.parameters())
        elif self.flow_type is not None and self.flow_type != 'nice':
            self.emit_params = [self._emission_mu] + list(self.emission.model.parameters())
        else:
            self.emit_params = [self._emission_mu]

        if tune_embeddings:
            if embedding_type == 'word':
                self.embeddings = nn.Parameter(data=self.embeddings)
            elif embedding_type == 'context':
                self.embeddings = nn.ParameterDict({x:nn.Parameter(self.embeddings[x]) for x in self.embeddings})

            self.emit_params += list(self.embeddings.values())
        if exp_type == 'rnn':
            self.bottom_layer = torch.nn.RNN(input_size=self.feature_size, hidden_size=100, num_layers=num_rnn_layers,
                                             bidirectional=bidirectional_flag, batch_first=True, nonlinearity='relu')
            self.top_linear = torch.nn.Linear(100*(int(bidirectional_flag)+1), 1)
            self.neural_exp = [self.bottom_layer, self.top_linear]
        elif exp_type == 'linear':
            self.bottom_linear = torch.nn.Sequential(torch.nn.Linear(self.feature_size*3, self.feature_size),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(self.feature_size, 100),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(100, 1))
        elif exp_type == 'flow':
            self.bottom_layer_left =  NICETrans(num_flow_blocks, 1, self.feature_size, self.feature_size)
            self.bottom_layer_right = NICETrans(num_flow_blocks, 1, self.feature_size, self.feature_size)
            self.bottom_left_dist = FLOWDist(self.bottom_layer_left, self._emission_mu, self._emission_sigma)
            self.bottom_right_dist = FLOWDist(self.bottom_layer_right, self._emission_mu, self._emission_sigma)

        if hasattr(self, 'bottom_layer'):
            self.base_params += list(self.bottom_layer.parameters())
        if hasattr(self, 'top_linear'):
            self.base_params += list(self.top_linear.parameters())
        # self.base_params.append(self.auxiliary_weights)
        self.drop_out = torch.nn.Dropout(0)

        logging.info(self)

    def normalize_models(self):

        if not self.full_neural:
            self.pcfg_split = F.log_softmax(self._pcfg_split, dim=1)
            self.p0 = F.log_softmax(self._p0, dim=0)
        else:
            self.pcfg_split = self._pcfg_split(self._emission_mu)
            self.p0 = F.log_softmax(self._p0(self._emission_mu).squeeze())
        # self.expansion = self.calc_expansion()
        self.expansion = self.calc_expansion()
        self.expansion = self.expansion + self.pcfg_split[:, 0][..., None]
        # logging.info('expansion total diff: ')
        expansion_diff = 0
        if hasattr(self, 'previous_expansion'):
            with torch.no_grad():
                expansion_diff = torch.nn.functional.kl_div(self.previous_expansion, self.expansion.exp(),
                                                            reduction='batchmean')
        else:
            pass
        logging.info('expansion total diff: {}'.format(expansion_diff))

        self.previous_expansion = self.expansion.data

    def normalized_parameters(self):
        # expansion = self.calc_expansion()
        expansion = self.calc_expansion()
        if not self.full_neural:
            p0 = F.log_softmax(self._p0, dim=0)
        else:
            p0 = F.log_softmax(self._p0(self._emission_mu).squeeze())
        return [ expansion, p0] # self.pcfg_split

    def sparsity(self, eps=1e-4):
        with torch.no_grad():
            if not self.full_neural:
                pcfg_split = F.log_softmax(self._pcfg_split, dim=1)
            else:
                pcfg_split = self._pcfg_split(self._emission_mu)
            # expansion = self.calc_expansion()
            self.expansion = self.calc_expansion()
            expansion = self.expansion + self.pcfg_split[:, 0][..., None]
            sparsity_measure = (expansion.exp() > eps).sum()
        return sparsity_measure.item() / expansion.numel()

    def gaussian_mean_distance_loss(self):

        distances = 0
        for x in range(self._emission_mu.shape[0]):
            for y in range(self._emission_mu.shape[0]):
                if x < y:
                    v1 = self._emission_mu[x].unsqueeze(0)
                    v2 = self._emission_mu[y].unsqueeze(0)
                    distance = torch.nn.functional.pairwise_distance(v1, v2)
                    distances = distances + distance
        loss = - distances
        return loss

    def calc_expansion(self):
        expansion = []
        mus = self._emission_mu  # + self.auxiliary_weights
        if self.exp_type != 'flow':
            repeated_mus = mus[:, None, :].expand(-1, mus.shape[0], -1).reshape(-1, self.feature_size)
            tiled_mus = mus.expand(mus.shape[0], mus.shape[0], self.feature_size).reshape(-1, self.feature_size)
            children_mus = torch.stack((repeated_mus, tiled_mus), dim=-2)
            for cat in range(self.K):
                this_mu = mus[cat]
                expanded_this_mu = this_mu.expand(children_mus.shape[0], 1, self.feature_size)
                all_mus = torch.cat((expanded_this_mu, children_mus), dim=1)
                all_mus = self.drop_out(all_mus)
                probs = self._single_cat_exp_probs(all_mus, self.exp_type)
                expansion.append(probs)
            expansion = torch.stack(expansion, dim=0)

        else:
            expansion = self._single_cat_exp_probs(mus, self.exp_type)
        return expansion

    def calc_expansion_dependency(self):
        expansion = []
        mus = self._emission_mu  # + self.auxiliary_weights
        if self.exp_type != 'flow':
            for cat in range(self.K):
                this_mu = mus[cat]
                expanded_this_mu = this_mu.expand(mus.shape[0]-1, self.feature_size)
                parent_mus = this_mu.expand(mus.shape[0]*2-2, 1, self.feature_size)

                valid_cats = [i for i in range(self.K) if i != cat]
                left_child_mus = torch.cat((expanded_this_mu, mus[valid_cats]), dim=0)[:, None, :]
                right_child_mus = torch.cat((mus[valid_cats], expanded_this_mu), dim=0)[:, None, :]
                all_mus = torch.cat((parent_mus, left_child_mus, right_child_mus), dim=1)
                # all_mus = torch.cat((expanded_this_mu, children_mus), dim=1)
                all_mus = self.drop_out(all_mus)
                probs = self._single_cat_exp_probs(all_mus, self.exp_type)
                true_probs = torch.full((self.K**2, ), -float('inf'))
                counter = 0
                for direction in ['l', 'r']:
                    for child_cat in range(self.K):
                        if child_cat == cat: continue
                        else:
                            if direction == 'l':
                                index = child_cat * self.K + cat
                            else:
                                index = cat * self.K + child_cat
                            true_probs[index] = probs[counter]
                            counter += 1
                expansion.append(true_probs.cuda())
            expansion = torch.stack(expansion, dim=0)

        else:
            expansion = self._single_cat_exp_probs(mus, self.exp_type)
        return expansion

    def _single_cat_exp_probs(self, mus, exp_type):
        if exp_type == 'rnn':
            last_h, _ = self.bottom_layer(mus)
            last_h = last_h[:,-1,:].squeeze()
            last_h = last_h.reshape(last_h.shape[0], -1)
            scores = self.top_linear(last_h).squeeze()
            probs = F.log_softmax(scores, dim=0)

        elif exp_type == 'linear':
            scores = self.bottom_layer(mus)
            probs = F.log_softmax(scores, dim=0)

        # not correct
        elif exp_type == 'flow':
            left_probs = self.bottom_left_dist.log_prob(mus)
            right_probs = self.bottom_right_dist.log_prob(mus)
            assert left_probs.numel() == right_probs.numel() == self.K ** 2
            probs = left_probs[...,None] + right_probs[:,None,:]

        elif exp_type == 'euclidean':
            left_logit = torch.bmm(mus[:, 0].unsqueeze(1), mus[:, 1].unsqueeze(2))
            right_logit = torch.bmm(mus[:, 0].unsqueeze(1), mus[:, 2].unsqueeze(2))
            total_logit = left_logit + right_logit
            probs = F.log_softmax(total_logit.squeeze(), dim=0)
        return probs
