#!/usr/bin/env python3.4

import os, time, socket, gzip
import sys
import torch
import multiprocessing
import bidict
import uuid
from sklearn.metrics import v_measure_score

from .pcfg_model import PCFG_model, PCFG_model_bayes
from .pcfg_model_gauss import PCFG_model_Gauss, PCFG_model_Gauss_Neuralexp_Flowemit
from .pcfg_translator import *
from .dimi_io import write_linetrees_file, read_gold_pcfg_file, write_out_category_related_data
from .metrics import *
from .delete_punc_tree import *
from .cky_parser_sgd import *
from .batcher import *
from .hyperparameter_collector import HyperParameterCollector
from .metric_groups import SimpleIterMetrics

# Has a state for every word in the corpus
# What's the state of the system at one Gibbs sampling iteration?
class Sample:
    def __init__(self):
        self.hid_seqs = []
        self.models = None
        self.log_prob = 0


def wrapped_sample_beam(*args, **kwargs):

    try:

        sample_beam(*args, **kwargs)
    except Exception as e:
        # print(e)
        logging.info('Sampling beam function has errored out!')
        logging.info('Hostname is '+socket.gethostname())
        raise e

# This is the main entry point for this module.
# Arg 1: ev_seqs : a list of lists of integers, representing
# the EVidence SEQuenceS seen by the user (e.g., words in a sentence
# mapped to ints).
# emission_type = [gaussian, multinomial, niceflow]
def sample_beam(ev_seqs, params, working_dir, punct_dict_file=None,
                word_dict_file=None, resume=False):
    global K
    hypparam_collector = HyperParameterCollector()

    train_max_len = hypparam_collector.add_param('Training Max length', int(params.get('train_max_len', 1e2)))

    debug = params.get('debug', 'INFO')
    logfile = params.get('logfile', 'log.txt.gz')
    logfile_fh = gzip.open(os.path.join(working_dir, logfile), 'wt', encoding='utf8')
    filehandler = logging.StreamHandler(logfile_fh)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level=getattr(logging, debug), format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    ## prepare the dictionaries for words, puncts
    with open(word_dict_file, 'r', encoding='utf-8') as f:
        word_dict = bidict.bidict()
        for line in f:
            (word, vocab_index) = line.rstrip().split(" ")
            word_dict[int(vocab_index)] = word
    if punct_dict_file is not None:
        with open(punct_dict_file, 'r', encoding='utf-8') as f:
            punct_dict = bidict.bidict()
            for line in f:
                (punct, punct_index) = line.rstrip().split("\t")
                punct_dict[punct_index] = punct
    else:
        punct_dict = {}

    # dev eval file
    dev_eval_file = params.get('dev_eval_file', None)
    if dev_eval_file is not None:
        hypparam_collector.add_param('Dev eval', 'True')
        dev_seqs = []
        with open(dev_eval_file, encoding='utf8') as df:
            for line in df:
                line = line.strip().split(' ')
                line = [int(x) for x in line]
                # for index, word in enumerate(line):
                #     if punct_dict and word in punct_dict.inv:
                #         line[index] = word_dict.inv[punct_dict.inv[word]]
                #     else:
                #         line[index] = word_dict.inv[word]
                dev_seqs.append(line)
        dev_sent_lens = list(map(len, dev_seqs))
        sorted_dev_seqs_index = np.argsort(dev_sent_lens).tolist()
    else:
        hypparam_collector.add_param('Dev eval', 'False')
        dev_seqs = None

    # get gold trees and depunced trees for eval
    gold_pcfg_file = params.get('gold_pcfg_file', None)
    if not gold_pcfg_file:
        raise ValueError("must have a gold pcfg file!")
    else:
        gold_trees = read_gold_pcfg_file(gold_pcfg_file, word_dict)
        # gold_models = get_gold_dists_abbabbc(gold_trees, num_words) # used for testing with the synth data
        gold_models = None
    gold_spans = []
    gold_nopunc_pos_seqs = []
    gold_withpunc_pos_seqs = []

    dropped_sents_num = 0
    for sent_index, sent in enumerate(ev_seqs):
        if len(sent) >= train_max_len:
            gold_trees[sent_index] = None
            ev_seqs[sent_index] = []
            dropped_sents_num += 1

    logging.info('Length filtered number of sentences: {} with max {}'.format(dropped_sents_num, train_max_len))

    for tr in gold_trees:
        l_spans = []
        # for subtree in tr.subtrees(lambda t : t.height() == 2):
        if tr is None:
            continue
        for word, pos in tr.pos():
            # pos = subtree.label()
            if '+' in pos:
                pos = pos.split('+')[-1]
            gold_withpunc_pos_seqs.append(pos)
    depunced_gold_trees = delete_puncs(gold_trees, word_dict)
    for tr in depunced_gold_trees:
        if tr is None:
            gold_spans.append([])
            continue
        l_spans = []
        for subtree in tr.subtrees(lambda t : t.height() >= 2):
            if subtree.height() > 2:
                l_spans.append(' '.join(subtree.leaves()))
            # else:
        for word, pos in tr.pos():
            # pos = subtree.label()
            if '+' in pos:
                pos = pos.split('+')[-1]
            gold_nopunc_pos_seqs.append(pos)
        gold_spans.append(l_spans)

    # calc sentence lengths
    sent_lens = list(map(len, ev_seqs))
    sorted_ev_seqs_index = np.argsort(sent_lens)
    sorted_ev_seqs_index = sorted_ev_seqs_index[dropped_sents_num:].tolist()

    # collect hyperparams for logging
    K = hypparam_collector.add_param('K', int(params.get('k')))
    max_len = hypparam_collector.add_param('Max length', max(map(len, ev_seqs)))

    vocab_size = hypparam_collector.add_param('Number of word types', len(word_dict))
    num_sents = hypparam_collector.add_param('Number of word types', len(ev_seqs))
    num_tokens = hypparam_collector.add_param('Number of word types', sum(sent_lens))
    hypparam_collector.add_param_list(['Average sentence length', 'Sentence length variance'],
                                      [np.mean(sent_lens), np.var(sent_lens)])
    ## Set debug first so we can use it during config setting:

    word_vecs_file = hypparam_collector.add_param('Word vec file', params.get('word_vecs_file', None))
    dev_word_vecs_file = hypparam_collector.add_param('Word vec file (dev)', params.get('dev_word_vecs_file', None))

    logging.info('Run #ID# : {}'.format(str(uuid.uuid4())))
    logging.info('Working folder : {}'.format(working_dir))
    logging.info('Embedding file : {}'.format(word_vecs_file))
    logging.info('Embedding file (dev) : {}'.format(dev_word_vecs_file))
    logging.info('GPU is {}'.format(torch.cuda.is_available()))
    emission_type = hypparam_collector.add_param('Type of Emission model', params.get('emission_type','gaussian'))
    ## emission type: multinomial, gaussian, flow
    init_method = hypparam_collector.add_param('Init method', params.get('init_method', 'dirichlet')) # dirichlet, uniform
    init_branching_tendency = hypparam_collector.add_param('Init branching tendency', params.get(
        'init_branching_tendency', None)) # none, right, left
    embedding_type = hypparam_collector.add_param('Embedding type', params.get('embedding_type', 'none')) # none, word,
    # context
    flow_type = hypparam_collector.add_param('Flow type', params.get('flow_type', None)) # nice, nvp, maf
    # if emission_type == 'flow': assert flow_type is not None, "FLOW TYPE must not be None if FLOW is chosen as emit."
    # auxiliary loss: vas, sim
    if 'flow' in emission_type:
        num_flow_blocks = hypparam_collector.add_param('Number of Flow blocks', int(params.get('num_flow_blocks', 4)))

    aux_losses = hypparam_collector.add_param('Auxiliary loss', params.get('aux_losses', None))
    if aux_losses is not None:
        aux_losses = aux_losses.split(',')
        for aux_loss in aux_losses:
            if aux_loss == 'sim':
                sim_scaler = hypparam_collector.add_param('Sim scaler', float(params.get('sim_scaler', 1e-5)))
            elif aux_loss == 'mu_distance':
                mu_distance_scaler = hypparam_collector.add_param('MuDistance scaler', float(params.get('mu_distance_scaler',
                                                                                                 1e-5)))
            elif aux_loss == 'branching_total':
                branching_total_scaler = hypparam_collector.add_param('Branching total scaler', float(params.get(
                    'branching_total_scaler', 1e-5)))
            elif aux_loss == 'branching_difference':
                branching_difference_scaler = hypparam_collector.add_param('Branching difference scaler', float(params.get(
                    'branching_difference_scaler', 1e-5)))
            elif aux_loss == 'sim_penalty':
                sim_penalty_scaler = hypparam_collector.add_param('Sim penalty scaler', float(params.get(
                    'sim_penalty_scaler', 5)))
            else:
                raise ValueError('Unknown loss {}.'.format(aux_loss))
    else:
        pass

    saved_param_fn = hypparam_collector.add_param('Saved param file', params.get('saved_params', None))
    init_eval_flag = hypparam_collector.add_param('Eval at -1 iter', params.get('init_eval_flag', 'True')=='True')

    if saved_param_fn is not None:
        saved_param_dict = torch.load(saved_param_fn)
    else:
        saved_param_dict = None
    tune_scale = hypparam_collector.add_param('Tune the scale of Gauss?', params.get('tune_scale', 'False')=='True') # tune
    #  the
    # scale
    D = hypparam_collector.add_param('Depth', int(params.get('d', -1)))
    iters = hypparam_collector.add_param('Total iterations', int(params.get('iters')))
    max_num_per_batch = hypparam_collector.add_param('Max batch size', max(1, int(params.get('max_num_per_batch', 1))))

    num_batches_per_update = hypparam_collector.add_param('Number of batches per update', int(params.get(
        'num_batches_per_update', 1)))
    viterbi_batch_size = hypparam_collector.add_param('Viterbi batch size', int(params.get('viterbi_batch_size', 64)))

    resume_iter = int(params.get("resume_iter", -1))
    num_batches_per_eval = hypparam_collector.add_param('Number of Batches per eval', int(params.get(
        'num_batches_per_eval',50)))
    if emission_type == 'multinomial_autoencoding':
        max_num_per_batch = hypparam_collector.add_param('Max batch size', 1)
        viterbi_batch_size = hypparam_collector.add_param('Viterbi batch size', 1)

    init_alpha = hypparam_collector.add_param('Init alpha', float(params.get("init_alpha", 1)))
    # output settings:
    print_out_first_n_sents = int(params.get('first_n_sents', -1))
    aug_number = hypparam_collector.add_param('Augmented sentence number', int(params.get('aug_number', 0)))
    gold_number = hypparam_collector.add_param('Gold sentence number', num_sents-aug_number)
    save_gaussian_params = hypparam_collector.add_param('Save Gauss params?', params.get('save_gaussian_params',
                                                                                        'False')=='True')

    save_debugging_params = hypparam_collector.add_param('Save debugging params?', params.get('save_debugging_params',
                                                                                         'False')=='True')# save params or

    seed = hypparam_collector.add_param('Seed', int(params.get('seed', -1))) # for seeding torch
    if seed > -1:
        seed = seed
    else:
        seed = random.randint(0, sys.maxsize)
    torch.manual_seed(seed)
    random.seed(seed)
    logging.info('Python and Torch random seed is {}'.format(seed))
    hypparam_collector.add_param('Seed', seed)

    samples = []
    start_ind = 0
    end_ind = num_sents

    if hypparam_collector.add_param('STD', params.get('std', None)):
        std = torch.load(hypparam_collector.add_param('STD', params.get('std', None)))
    else:
        std = None
    save_sent_embs = hypparam_collector.add_param('Save sent embs', params.get('save_sent_embs', 'False') == 'True')
    if emission_type == 'multinomial':
        embeddings = None
        pcfg_model = PCFG_model(K, D, vocab_size, num_sents, num_tokens, init_alpha, log_dir=working_dir,
                                word_dict_file=word_dict_file, saved_params=saved_param_dict,
                                hyperparam_collector=hypparam_collector, init_method=init_method,
                                init_branching_tendency=init_branching_tendency, ev_seqs=ev_seqs)

    elif emission_type == 'gaussian':
        embeddings = compile_embeddings(word_vecs_file, word_dict, punct_dict, embedding_type)
        if dev_word_vecs_file is not None and embedding_type == 'context':
            dev_embeddings = compile_embeddings(dev_word_vecs_file, word_dict, punct_dict, embedding_type)
            for dev_sent_index in dev_embeddings:
                embeddings[dev_sent_index+num_sents] = dev_embeddings[dev_sent_index]
        pcfg_model = PCFG_model_Gauss(K, D, vocab_size, num_sents, num_tokens, init_alpha, log_dir=working_dir,
                                      word_dict_file=word_dict_file, saved_params=saved_param_dict,
                                      embeddings=embeddings, flow_type=flow_type, tune_scale=tune_scale,
                                      hyperparam_collector=hypparam_collector, init_method=init_method,
                                      embedding_type=embedding_type, std=std)
    elif emission_type == 'flow':
        embeddings = compile_embeddings(word_vecs_file, word_dict, punct_dict, embedding_type)
        tune_embeddings_flag = hypparam_collector.add_param('Tune embeddings', params.get('tune_embeddings_flag',
                                                                                             'False') == 'True')

        if dev_word_vecs_file is not None and embedding_type == 'context':
            dev_embeddings = compile_embeddings(dev_word_vecs_file, word_dict, punct_dict, embedding_type)
            for dev_sent_index in dev_embeddings:
                embeddings[dev_sent_index+num_sents] = dev_embeddings[dev_sent_index]
        pcfg_model = PCFG_model_Gauss(K, D, vocab_size, num_sents, num_tokens, init_alpha, log_dir=working_dir,
                                      word_dict_file=word_dict_file, saved_params=saved_param_dict,
                                      embeddings=embeddings, flow_type=flow_type, tune_scale=tune_scale,
                                      hyperparam_collector=hypparam_collector, init_method=init_method,
                                      embedding_type=embedding_type, num_flow_blocks=num_flow_blocks, std=std,
                                      tune_embeddings=tune_embeddings_flag)

    elif emission_type == 'flow-neuralexp':
        embeddings = compile_embeddings(word_vecs_file, word_dict, punct_dict, embedding_type)

        tune_embeddings_flag = hypparam_collector.add_param('Tune embeddings', params.get('tune_embeddings_flag',
                                                                                             'False') == 'True')
        drop_out_rate = hypparam_collector.add_param('Drop out rate', int(params.get('drop_out_rate', 0)))
        num_rnn_layers = hypparam_collector.add_param('Number RNN layers', int(params.get('num_rnn_layers', 1)))
        bidirectional_flag = hypparam_collector.add_param('Bidirectional RNN?', params.get('bidirectional_flag',
                                                                                             'False') == 'True')
        exp_type = hypparam_collector.add_param('Expansion type', params.get('exp_type', 'rnn'))

        if dev_word_vecs_file is not None and embedding_type == 'context':
            dev_embeddings = compile_embeddings(dev_word_vecs_file, word_dict, punct_dict, embedding_type)
            for dev_sent_index in dev_embeddings:
                embeddings[dev_sent_index+num_sents] = dev_embeddings[dev_sent_index]
        pcfg_model = PCFG_model_Gauss_Neuralexp_Flowemit(K, D, vocab_size, num_sents, num_tokens, init_alpha, log_dir=working_dir,
                                                         word_dict_file=word_dict_file, saved_params=saved_param_dict,
                                                         embeddings=embeddings, flow_type=flow_type, tune_scale=tune_scale,
                                                         hyperparam_collector=hypparam_collector, init_method=init_method,
                                                         embedding_type=embedding_type, num_flow_blocks=num_flow_blocks, std=std,
                                                         tune_embeddings=tune_embeddings_flag, drop_out_rate=drop_out_rate, num_rnn_layers=num_rnn_layers,
                                                         bidirectional_flag=bidirectional_flag, exp_type=exp_type)

    pcfg_model = pcfg_model.to('cuda')
    pcfg_model.normalize_models()
    cky_parser = batch_CKY_parser(pcfg_model.K, D, max_len)

    cky_parser.set_models(pcfg_model.p0, pcfg_model.expansion, pcfg_model.emission, pcfg_model.embeddings,
                          pcfg_model.pcfg_split, embedding_type=embedding_type)

    batch_metrics = SimpleIterMetrics()

    word_dict = pcfg_model.word_dict
    # print(bounded_pcfg_model.K)

    if not resume:

        dnn_obs_model = None

        hid_seqs = [None] * num_sents
        viterbi_hid_seqs = [None] * num_sents
        logprobs = [None] * num_sents
        fixed_logprobs = torch.zeros(num_sents)
        average_fixed_logprobs = torch.zeros_like(fixed_logprobs)

        pcfg_model.start_logging()

        cur_iter = 0

    else:
        try:
            if resume_iter > 0:
                num_iter = resume_iter
            else:
                pcfg_runtime_stats = open(os.path.join(working_dir, 'pcfg_hypparams.txt'), encoding='utf8')
                num_iter = int(pcfg_runtime_stats.readlines()[-1].split('\t')[0])
            pcfg_model, dnn_obs_model = torch.load(open(os.path.join(working_dir, 'pcfg_model_'+str(
                num_iter)+'.pkl'), 'rb'))
        except:
            pcfg_model, dnn_obs_model = torch.load(open(os.path.join(working_dir, 'pcfg_model_'+str(
                num_iter-1)+'.pkl'), 'rb'))

        dnn_obs_model = None
        logging.info("Continuing from iteration {}".format(num_iter))
        pcfg_model.set_log_mode('a')
        pcfg_model.start_logging()

        # sample_pcfg_dict, sample_p0 = pcfg_replace_model(None, None, bounded_pcfg_model, pcfg_model, resume=True, dnn=dnn_obs_model)

        hid_seqs = [None] * num_sents
        viterbi_hid_seqs = [None] * num_sents
        logprobs = [0] * num_sents
        fixed_logprobs = torch.zeros(num_sents)
        average_fixed_logprobs = torch.zeros_like(fixed_logprobs)

        cur_iter = pcfg_model.iter

    batches = batcher(ev_seqs, sorted_ev_seqs_index, max_num_per_batch) # (tensor, indices)
    viterbi_batches = batcher(ev_seqs, sorted_ev_seqs_index, viterbi_batch_size, viterbi=True) # for viterbi parsing
    if dev_seqs is not None:
        dev_batches = batcher(dev_seqs, sorted_dev_seqs_index, viterbi_batch_size) # for dev eval
    # NN related params
    hypparam_collector.add_param('Number of batches', len(batches))

    optimizer_name = hypparam_collector.add_param('Optimizer', params.get('optimizer', 'Adam'))
    batch_average = hypparam_collector.add_param('Batch average loss', params.get('batch_average', 'sentence')) # batch,
    # sentence, none
    loss_multiplier = hypparam_collector.add_param('Loss multiplier', int(params.get('loss_multiplier', 1))) # batch,

    # sentence
    hypparam_collector.add_param('Embedding window size', params.get('emb_window_size', 0))

    lr_base = hypparam_collector.add_param('Learning rate (Base)', float(params.get('lr_base', 0.1)))
    lr_emit = hypparam_collector.add_param('Learning rate (Emit)', float(params.get('lr_emit', 1e-3)))

    l1_reg = hypparam_collector.add_param('L1 regularization', float(params.get('l1_reg', 0.)))
    lexical_l1_reg = hypparam_collector.add_param('Lexical L1 reg', float(params.get('lexical_l1_reg', 0.)))

    max_gradient_clipping = hypparam_collector.add_param('Max gradient for clipping', float(params.get(
        'max_gradient_clipping', -1)))

    shuffle_batches = hypparam_collector.add_param('Shuffle batches?', params.get('shuffle_batches', 'True')=='True')
    amsgrad_flag = hypparam_collector.add_param('AMSgrad flag', params.get('amsgrad_flag', 'False')=='True')

    if optimizer_name == 'SGD':
        # lr_emit = hypparam_collector.add_param('Learning rate (Emit)', lr_base) # only the base lr is working
        optimizer = torch.optim.SGD(pcfg_model.parameters(), lr=lr_base)
        hypparam_collector.remove_param('Learning rate (Emit)')
    elif optimizer_name == 'Adam':
        # lr_emit = hypparam_collector.add_param('Learning rate (Emit)', lr_base)
        optimizer = torch.optim.Adam(pcfg_model.parameters(), lr=lr_base, amsgrad=amsgrad_flag)
        hypparam_collector.remove_param('Learning rate (Emit)')
    elif optimizer_name == 'Adam_split':
        assert emission_type != 'multinomial'
        param_index = 0
        param_group_index = 0
        for params in [pcfg_model.base_params, pcfg_model.emit_params]:
            for param in params:
                logging.info('{} {} | {}'.format(param_group_index, param_index, list(param.shape)))
                param_index += 1
            param_group_index += 1
            param_index = 0
        optimizer = torch.optim.Adam([{'params':pcfg_model.base_params, 'lr':lr_base},
                                      {'params':pcfg_model.emit_params, 'lr':lr_emit}], amsgrad=amsgrad_flag)


    ### print out all hyperparameters
    logging.info(hypparam_collector.write_out())
    batch_index = -1
    iter_index = -1
    ## first iter, before optimization, eval the init model
    if init_eval_flag:
        pcfg_model.eval()
        with torch.no_grad():
            # pcfg_model.save_grammar_params(iter_index, batch_index)
            total_branches, right_branches = 0, 0
            if dev_seqs is not None:
                dev_logprobs = torch.zeros((len(dev_seqs),))
                dev_as = torch.zeros((len(dev_seqs),))
                for batch, sent_indices in dev_batches:
                    print()
                    if embedding_type == 'context':
                        sent_indices_added = [x+num_sents for x in sent_indices]
                    else:
                        sent_indices_added = sent_indices
                    _, logprob_list_fixed, _, _, \
                    vtree_list, _, vlr_branches_list, _ = cky_parser.marginal(batch, viterbi_flag=False, only_viterbi=False,
                                                                           sent_indices=sent_indices_added)
                    for index_index, sent_index in enumerate(sent_indices):
                        dev_logprobs[sent_index] = logprob_list_fixed[index_index].item()
                        dev_as[sent_index] = dev_logprobs[sent_index] / dev_sent_lens[sent_index]
                dev_logprobs = dev_logprobs.sum().item()
                dev_vas = dev_as.var().item()
            kk = 0

            for batch, sent_indices in viterbi_batches:
                # print(batch.shape, sum(sent_indices))

                # logging.info('{}, len {}, size {}'.format(kk / len(viterbi_batches), len(batch[0]),
                #                                           len(sent_indices) ))
                kk += 1
                _, logprob_list_fixed, _, _, \
                vtree_list, _, vlr_branches_list, _ = cky_parser.marginal(batch, viterbi_flag=True, only_viterbi=False,
                                                                       sent_indices=sent_indices)
                left_b, right_b = zip(*vlr_branches_list)
                total_branches += sum(left_b) + sum(right_b)
                right_branches += sum(right_b)
                for index_index, sent_index in enumerate(sent_indices):
                    fixed_logprobs[sent_index] = logprob_list_fixed[index_index].item()
                    average_fixed_logprobs[sent_index] = fixed_logprobs[sent_index] / sent_lens[sent_index]
                    viterbi_hid_seqs[sent_index] = vtree_list[index_index]
            viterbi_nopunc_pos_seqs = []
            viterbi_withpunc_pos_seqs = []
            for tr_index, tr in enumerate(viterbi_hid_seqs):
                # for subtree in tr.subtrees(lambda t: t.height() == 2):
                if tr_index >= gold_number:
                    break
                if tr is None: continue
                for word, pos in tr.pos():
                    viterbi_withpunc_pos_seqs.append(pos)

            vmeasure_w_punc = v_measure_score(gold_withpunc_pos_seqs, viterbi_withpunc_pos_seqs)
            depunced_viterbi_trees = delete_puncs(viterbi_hid_seqs, word_dict)
            viterbi_recall_val = recall(depunced_viterbi_trees, gold_spans, aug_number)
            for tr_index, tr in enumerate(depunced_viterbi_trees):
                # for subtree in tr.subtrees(lambda t: t.height() == 2):
                if tr_index >= gold_number:
                    break
                if tr is None: continue

                for word, pos in tr.pos():
                    viterbi_nopunc_pos_seqs.append(pos)
            vmeasure = v_measure_score(gold_nopunc_pos_seqs, viterbi_nopunc_pos_seqs)
            best_cat_word_dict = cky_parser.find_max_prob_words()
            write_out_category_related_data(pcfg_model.cat_word_log, best_cat_word_dict, type='best_word',
                                            word_dict=word_dict, batch_index=batch_index, iter_index=iter_index)
            if pcfg_model.pcfg_split is not None:
                write_out_category_related_data(pcfg_model.pcfg_split_log, pcfg_model.pcfg_split, type='pcfg_split',
                                            batch_index=batch_index, iter_index=iter_index)
            cky_parser.clear_vocab_prob_list()
        logging.info("Iter INIT; VAS: {:.4f}; Fixed Logprob: {:.4f}; RB score: {:.4f}; VM-PUNC: {:4f}; VM+PUNC: {:.4f}".format(
            average_fixed_logprobs.var(), fixed_logprobs.sum(),right_branches / total_branches, vmeasure, vmeasure_w_punc))
        if True: # metric group logging
            batch_metric_group = batch_metrics.spawn_batch()
            batch_metric_group.iter_index = iter_index
            batch_metric_group.batch_index = batch_index
            batch_metric_group.logprobs = fixed_logprobs.sum().item()
            batch_metric_group.rb_score = right_branches / total_branches
            batch_metric_group.viterbi_recall = viterbi_recall_val
            batch_metric_group.vas = average_fixed_logprobs.var().item()
            batch_metric_group.vm_nopunc = vmeasure
            batch_metric_group.vm_withpunc = vmeasure_w_punc
            # if pcfg_model.pcfg_split is not None and 'hierarchical' not in emission_type:
            #     batch_metric_group.viterbi_upper = calc_top_vit_loglikelihood(pcfg_model.p0, pcfg_model.expansion,
            #                                                               pcfg_model.pcfg_split, viterbi_hid_seqs)
            batch_metric_group.sparsity = pcfg_model.sparsity()
            if dev_seqs is not None:
                batch_metric_group.dev_logprobs = dev_logprobs
                batch_metric_group.dev_vas = dev_vas
            batch_metrics.last_batch = batch_metric_group
            batch_metrics.write_out_last(pcfg_model.hypparam_log)

    ### Start doing actual optimization:
    last_25_logprobs = torch.zeros(25)
    prev_logprob = 0
    max_logprob = - float('inf')
    p = None
    q = None
    # memory_size_estimator = SizeEstimator(cky_parser, input_size=batches[-1][0].size())
    # print(memory_size_estimator.estimate_size())
    while cur_iter < iters:
        iter_index = cur_iter
        sent_index_list = sorted_ev_seqs_index
        pcfg_model.iter = cur_iter
        last_25_logprob_index = cur_iter % 25

        if shuffle_batches:
            random.shuffle(batches)
        t0 = time.time()

        # torch.autograd.set_detect_anomaly(True)

        for batch_num, (batch, sent_indices) in enumerate(batches):
            pcfg_model.train()
            batch_index += 1

            # if optimizer_name == 'Adam_split' and (batch_index == 1000 or batch_index == 2500):
            #     assert emission_type != 'multinomial'
            #     optimizer = Adam([{'params': pcfg_model.base_params, 'lr': lr_base},
            #                       {'params': pcfg_model.emit_params, 'lr': lr_emit}], amsgrad=True)

            need_eval_and_vit = ((batch_index + 1) % num_batches_per_eval == 0)
            if emission_type == 'multinomial_autoencoding':
                pcfg_model.sample_grammar(batch)
                cky_parser.set_models(pcfg_model.p0, pcfg_model.expansion, pcfg_model.emission)

            _, logprobs_tensor, _, _, \
            _, _, vlr_branches_list, lexical_l1 = cky_parser.marginal(batch, False, sent_indices=sent_indices)
            for index_index, sent_index in enumerate(sent_indices):
                logprobs[sent_index] = logprobs_tensor[index_index].item()

            mml_loss = - sum(logprobs_tensor)
            logging.info('Iter {:3d} | Batch {:5d} | NegLogprob {:10.4f} | Surprisal {:2.4f} | Len {:2d}'.format(
                cur_iter,  batch_index, mml_loss.item(), mml_loss.item() / len(batch) / len(batch[0]),
                        len(batch[0])))
            if batch_average == 'batch':
                mml_loss = mml_loss / len(batch)
            elif batch_average == 'sentence':
                mml_loss = mml_loss / len(batch) / len(batch[0])
            mml_loss = mml_loss * loss_multiplier #/ num_batches_per_update

            loss_dict = {'MML':mml_loss}

            # auxiliary loss
            if aux_losses is not None:
                for aux_loss in aux_losses:
                    if aux_loss == 'sim':
                        with torch.no_grad():
                            mu_sims = pcfg_model._emission_mu @ pcfg_model._emission_mu.t()
                        dist_sims = pcfg_model._expansion[:, None, ...] + pcfg_model._expansion[None, ...]
                        dist_sims = torch.logsumexp(dist_sims, dim=-1).exp()
                        sim_loss = mu_sims @ dist_sims
                        sim_loss = sim_loss.sum() * sim_scaler
                        loss_dict['Sim'] = sim_loss
                    elif aux_loss == 'mu_distance':
                        dis_loss = pcfg_model.gaussian_mean_distance_loss()
                        loss_dict['MuDistance'] = dis_loss * mu_distance_scaler
                    elif aux_loss == 'sim_penalty':
                        if hasattr(pcfg_model.emission, 'sim_penalty'):
                            sim_penalty_loss = pcfg_model.emission.sim_penalty
                        else:
                            sim_penalty_loss = 0
                        loss_dict['SimPenalty'] = sim_penalty_loss * sim_penalty_scaler
            else:
                pass

            if 'flow' in emission_type and flow_type is not None:
                pcfg_model.emission.reset_sim_penalty()

            if l1_reg != 0:
                total_l1 = 0
                for param in pcfg_model.normalized_parameters():
                    total_l1 = torch.norm(param.exp(), p=1) + total_l1
                loss_dict['L1'] = total_l1 * l1_reg * num_batches_per_update * len(batch)

            if lexical_l1_reg != 0 and lexical_l1 is not None:
                loss_dict['L1_lex'] = lexical_l1 * lexical_l1_reg * num_batches_per_update * len(batch)

            string = ''
            all_loss = 0
            for loss in loss_dict:
                # if loss != 'L1': continue
                string += '{} : {:.4f}, '.format(loss, loss_dict[loss].item())
                this_loss = loss_dict[loss]

                all_loss = all_loss + this_loss
            all_loss.backward()
            del loss_dict
            del all_loss
            # for param in pcfg_model.parameters():
            #     logging.info('{}; mean {}, max {}'.format(param.shape, param.grad.mean().item(), param.grad.max().item()))
            logging.info(string)
            pcfg_model.normalize_models()
            if 'hierarchical' in emission_type:
                cky_parser.set_models(pcfg_model.p0, pcfg_model.expansion, pcfg_model.emission, pcfg_model.embeddings,
                                  pcfg_model.pcfg_split, embedding_type=embedding_type, k2=pcfg_model.k2)
            elif emission_type != 'multinomial_autoencoding':
                cky_parser.set_models(pcfg_model.p0, pcfg_model.expansion, pcfg_model.emission, pcfg_model.embeddings,
                                  pcfg_model.pcfg_split, embedding_type=embedding_type)

            if (batch_index + 1) % num_batches_per_update == 0:

                if max_gradient_clipping > 0:
                    torch.nn.utils.clip_grad_value_(pcfg_model.parameters(), max_gradient_clipping)

                optimizer.step()
                optimizer.zero_grad()

            if need_eval_and_vit:
                cky_parser.clear_vocab_prob_list()
                if save_debugging_params:
                    pcfg_model.save_grammar_params(iter_index, batch_index)
                elif emission_type == 'flow-neuralexp':
                    pcfg_model.save_grammar_params(iter_index, batch_index, only_mu=True)
                pcfg_model.eval()
                with torch.no_grad():
                    total_branches, right_branches = 0, 0

                    if dev_seqs is not None:
                        dev_logprobs = torch.zeros((len(dev_seqs),))
                        dev_as = torch.zeros((len(dev_seqs),))
                        dev_hid_seqs = [None] * len(dev_seqs)

                        for batch, sent_indices in dev_batches:
                            if embedding_type == 'context':
                                sent_indices_added = [x + num_sents for x in sent_indices]
                            else:
                                sent_indices_added = sent_indices
                            _, logprob_list_fixed, _, _, \
                            vtree_list, _, vlr_branches_list, _ = cky_parser.marginal(batch, viterbi_flag=True,
                                                                                   only_viterbi=False,
                                                                                   sent_indices=sent_indices_added)
                            for index_index, sent_index in enumerate(sent_indices):
                                dev_logprobs[sent_index] = logprob_list_fixed[index_index].item()
                                dev_as[sent_index] = dev_logprobs[sent_index] / dev_sent_lens[sent_index]
                                dev_hid_seqs[sent_index] = vtree_list[index_index]

                        dev_logprobs = dev_logprobs.sum().item()
                        dev_vas = dev_as.var().item()
                        dev_linetrees_fn = 'iter_' + str(cur_iter) + '_batch_' + str(batch_index) + '.dev.linetrees'
                        dev_full_fn = os.path.join(working_dir, dev_linetrees_fn)
                        q = multiprocessing.Process(target=write_linetrees_file,
                                                    args=([None], pcfg_model.word_dict, dev_full_fn, False,
                                                          dev_hid_seqs, True, punct_dict))
                        q.daemon = True
                        q.start()

                    sent_embs = [None] * num_sents
                    for batch, sent_indices in viterbi_batches:
                        if emission_type == 'flow' and embedding_type == 'context' and save_sent_embs is True:
                            for index, sent_index in enumerate(sent_indices):
                                u, _ = pcfg_model.emission.forward(pcfg_model.embeddings[str(sent_index)].to('cuda'))
                                sent_embs[sent_index] = u.cpu()
                        _, logprob_list_fixed, _, _, \
                        vtree_list, _, vlr_branches_list, _ = cky_parser.marginal(batch, need_eval_and_vit,
                                                                                sent_indices=sent_indices)
                        left_b, right_b = zip(*vlr_branches_list)
                        total_branches += sum(left_b) + sum(right_b)
                        right_branches += sum(right_b)
                        for index_index, sent_index in enumerate(sent_indices):
                            fixed_logprobs[sent_index] = logprob_list_fixed[index_index].item()
                            average_fixed_logprobs[sent_index] = fixed_logprobs[sent_index] / sent_lens[sent_index]
                            viterbi_hid_seqs[sent_index] = vtree_list[index_index]
                    best_cat_word_dict = cky_parser.find_max_prob_words()
                write_out_category_related_data(pcfg_model.cat_word_log, best_cat_word_dict, type='best_word',
                                                word_dict=word_dict, batch_index=batch_index, iter_index=iter_index)
                if pcfg_model.pcfg_split is not None:

                    write_out_category_related_data(pcfg_model.pcfg_split_log, pcfg_model.pcfg_split, type='pcfg_split',
                                                batch_index=batch_index, iter_index=iter_index )
                hidden_embs = os.path.join(working_dir, 'hidden_embs.pkl')
                torch.save(sent_embs, hidden_embs)
                hidden_embs = []

                cky_parser.clear_vocab_prob_list()
                viterbi_nopunc_pos_seqs = []
                viterbi_withpunc_pos_seqs = []
                for tr_index, tr in enumerate(viterbi_hid_seqs):
                    if tr is None: continue

                    if tr_index >= gold_number:
                        break
                    # for subtree in tr.subtrees(lambda t: t.height() == 2):
                    if tr is None: continue
                    for word, pos in tr.pos():
                        viterbi_withpunc_pos_seqs.append(pos)
                vmeasure_w_punc = v_measure_score(gold_withpunc_pos_seqs, viterbi_withpunc_pos_seqs)
                depunced_viterbi_trees = delete_puncs(viterbi_hid_seqs, word_dict)
                for tr_index, tr in enumerate(depunced_viterbi_trees):
                    if tr is None: continue

                    if tr_index >= gold_number:
                        break
                    # for subtree in tr.subtrees(lambda t: t.height() == 2):
                    for word, pos in tr.pos():
                        viterbi_nopunc_pos_seqs.append(pos)
                vmeasure = v_measure_score(gold_nopunc_pos_seqs, viterbi_nopunc_pos_seqs)
                viterbi_recall_val = recall(depunced_viterbi_trees, gold_spans, aug_number)
                logging.info("Iter {:3d} Batch {:5d} eval; Fixed Logprob: {:.4f}".format(cur_iter, batch_index,
                                                                                   fixed_logprobs.sum()))

                t2 = time.time()
                batch_metric_group = batch_metrics.spawn_batch()
                batch_metric_group.iter_index = iter_index
                batch_metric_group.batch_index = batch_index
                batch_metric_group.logprobs = fixed_logprobs.sum().item()
                batch_metric_group.rb_score = right_branches / total_branches
                batch_metric_group.viterbi_recall = viterbi_recall_val
                batch_metric_group.vas = average_fixed_logprobs.var().item()
                batch_metric_group.vm_nopunc = vmeasure
                batch_metric_group.vm_withpunc = vmeasure_w_punc
                # if pcfg_model.pcfg_split is not None and 'hierarchical' not in emission_type:
                #     batch_metric_group.viterbi_upper = calc_top_vit_loglikelihood(pcfg_model.p0, pcfg_model.expansion,
                #                                                               pcfg_model.pcfg_split, viterbi_hid_seqs)
                batch_metric_group.sparsity = pcfg_model.sparsity()
                if dev_seqs is not None:
                    batch_metric_group.dev_logprobs = dev_logprobs
                    batch_metric_group.dev_vas = dev_vas
                batch_metrics.last_batch = batch_metric_group
                batch_metrics.write_out_last(pcfg_model.hypparam_log)

                linetrees_fn = 'iter_' + str(cur_iter) + '_batch_' + str(batch_index) + '.linetrees'
                if batch_metric_group.logprobs > max_logprob and save_gaussian_params:
                    pcfg_model.save_gauss_params(batch_metric_group.iter_index, batch_metric_group.batch_index)
                    max_logprob = batch_metric_group.logprobs


                full_fn = os.path.join(working_dir, linetrees_fn)
                if print_out_first_n_sents != -1:
                    trees = hid_seqs[: print_out_first_n_sents]
                    if need_eval_and_vit:
                        v_trees = viterbi_hid_seqs[: print_out_first_n_sents]
                    else:
                        v_trees = None
                else:
                    trees = hid_seqs
                    if need_eval_and_vit:
                        v_trees = viterbi_hid_seqs
                    else:
                        v_trees = None

                hid_seqs = [None] * num_sents
                viterbi_hid_seqs = [None] * num_sents

                pprint_bool = False

                if cur_iter % 25 == 0 or need_eval_and_vit:
                    anyprint = True
                else:
                    anyprint = False

                p = multiprocessing.Process(target=write_linetrees_file, args=(trees, pcfg_model.word_dict, full_fn, pprint_bool,
                                                                               v_trees, anyprint, punct_dict))
                p.daemon = True
                p.start()

        t1 = time.time()
        time_diff = t1-t0
        logging.info("Iteration {} took {:.4f} secs. Per batch {:.4f} secs".format(cur_iter, time_diff, time_diff/len(
            batches)))

        cur_iter += 1
        if p is not None:
            p.join()
        if q is not None:
            q.join()

    logging.info("Sampling complete.")
    # return samples
    logfile_fh.close()
