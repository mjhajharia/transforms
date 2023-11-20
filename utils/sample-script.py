import argparse
import json
import os
from sample import sample
parser = argparse.ArgumentParser()
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--target_keyword', type=str, required=True)
parser.add_argument('--n_repeat', type=int, default=25, required=False)
parser.add_argument('--n_iter', type=int, default=1000, required=False)
parser.add_argument('--n_chains', type=int, default=4, required=False)
parser.add_argument('--show_progress', type=bool, default=True, required=False)
parser.add_argument('--inits', type=str, default=None, required=False)
args = parser.parse_args()

if args.inits is not None:
    inits = json.load(open(args.inits))

sample(transform=args.transform,
target_keyword=args.target_keyword,
n_repeat=args.n_repeat,
n_iter=args.n_iter,
n_chains=args.n_chains,
show_progress=args.show_progress)
