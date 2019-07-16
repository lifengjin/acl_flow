import sys
sys.path.append('/home/jin.544/dimi_sgd/')
from elmoformanylangs import Embedder
import torch
elmo_model_path = sys.argv[1]
linetoks_fn = sys.argv[2]

e = Embedder(elmo_model_path)

with open(linetoks_fn) as lfh:

    lines = lfh.readlines()
    xlines = [line.strip().split(' ') for line in lines]

    embs = e.sents2elmo(xlines)

    context_embs_dict = {}

    for index, emb in enumerate(embs):
        context_embs_dict[index] = torch.from_numpy(emb)

    torch.save(context_embs_dict, linetoks_fn.replace('.linetoks', '.elmo.tch'))


