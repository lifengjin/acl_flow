import multiprocessing
import pickle

import sys

import itertools

import scripts.dimi_io as io
import configparser
import scripts.dimi as dimi
import os
from random import randint, random
import time

def main(argv):
    if len(argv) < 1:
        sys.stderr.write("One required argument: <Config file|Resume directory>\n")
        sys.exit(-1)

    path = argv[0]
    D, K, init_alpha = 0, 0, 0
    if len(argv) == 3:
        D, K = argv[1], argv[2]
    elif len(argv) == 4:
        D, K, init_alpha = argv[1], argv[2], argv[3]
    if not os.path.exists(path):
        sys.stderr.write("Input file/dir does not exist!\n")
        sys.exit(-1)

    config = configparser.ConfigParser()
    input_seqs_file = None



    time.sleep(random() * 10)
    if os.path.isdir(path):
        ## Resume mode
        config.read(path + "/config.ini")
        out_dir = config.get('io', 'output_dir')
        resume = True
    else:
        config.read(argv[0])
        input_seqs_file = config.get('io', 'init_seqs', fallback=None)
        if not input_seqs_file is None:
            del config['io']['init_seqs']
        out_dir = config.get('io', 'output_dir')
        if not D and not K:
            D = config.get('params', 'd')
            K = config.get('params', 'k')
        if not init_alpha:
            init_alpha = config.get('params', 'init_alpha')
        init_alpha = str(float(init_alpha))
        config['params']['d'] = D
        config['params']['k'] = K
        if init_alpha:
            config['params']['init_alpha'] = init_alpha
        out_dir += '_D'+D+'K'+K+'A'+init_alpha

        counter = itertools.count()
        for i in counter:
            new_out_dir = out_dir + '_{}'.format(i)
            if not os.path.exists(new_out_dir):
                os.makedirs(new_out_dir)
                out_dir = new_out_dir
                config['io']['output_dir'] = out_dir
                sys.stderr.write("The output directory for this run is {}.\n".format(out_dir))
                break
        resume = False


        with open(out_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)

    ## Write git hash of current branch to out directory
    os.system('git rev-parse HEAD > %s/git-rev.txt' % (out_dir))

    input_file = config.get('io', 'input_file')
    working_dir = config.get('io', 'working_dir', fallback=out_dir)
    dict_file = config.get('io', 'dict_file')
    punct_dict_file = config.get('io', 'punct_dict_file', fallback=None)

    ## Read in input file to get sequence for X
    (pos_seq, word_seq) = io.read_input_file(input_file)

    params = read_params(config)
    params['output_dir'] = out_dir

    dimi.wrapped_sample_beam(word_seq, params, working_dir,
                             word_dict_file = dict_file, resume=resume, punct_dict_file=punct_dict_file)


def read_params(config):
    params = {}
    for (key, val) in config.items('io'):
        params[key] = val
    for (key, val) in config.items('params'):
        params[key] = val

    return params


if __name__ == "__main__":

    try:
        multiprocessing.set_start_method("fork")
        # loky.set_start_method("loky")
    except:
        ctx = multiprocessing.get_start_method()
        print(ctx)


    if sys.version_info[0] != 3:
        print("This script requires Python 3")
        exit()

    main(sys.argv[1:])
