This is the repository for our ACL 2019 paper **Unsupervised PCFG Induction with Normalizing Flow**.


#### Dependencies:

Required packages:
- Python 3.6+
- pytorch 1.1+
- gensim
- nltk
- numpy
- scipy
- bidict

Newest versions of the packages should also work.

-----

#### How to use:

1. You will need a *linetrees* file, which is a one-sentence-per-line file with bracketed trees without the ROOT
node. The `make_linetoks.py` and `make_ints_file.py` in `utils` folder will use this file to first generate a `linetoks` file,
one-sentence-per-line with just the terminals, and `dict` and `ints` file where the terminals are replaced
with indices.

2. `embed_with_multilingual_elmo.py` in utils requires [ElmoForManyLang](https://github.com/HIT-SCIR/ELMoForManyLangs).
You can also get pretrained Elmo models from there. Use the script to generate the Elmo embeddings for the dataset.

3. A config file is needed. One sample config file is provided in the `config` folder along with the necessary 
text files in `uyghur_data`. The options are explained in the config file.

4. The running command is `python dimi-trainer.py config/yourconfig.ini`. If the Elmo embeddings are generated for 
the provided Uyghur file, issuing `python dimi-trainer.py config/uyghur.ini` should start the model immediately.
A GPU is required for running the system.

5. Results will be dumped out into the provided output folder. The main diagnostic file is `running_status.txt`, which
includes a whole array of different measurements for grammar quality. The `*.vittrees.gz` files are gzipped files of
Viterbi trees of the dataset.