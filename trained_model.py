import json
import torch
import argparse
import numpy as np
import torch.nn as nn

from pathlib import Path
from modules.train import *
from modules.dataset import *
from modules.network import *
from torch import cuda, device, save

# Set local paths
res_path = "./data/results"
out_dir = res_path + "18_240_64_4_4_1200"


if __name__ == "__main__":

    # Retrieve parameters
    parser = argparse.ArgumentParser(description='PyTorc Text generation Model, based on The Bible.')
    parser.add_argument('--n', type=int,  default='100',  help='number of words to generate')
    parser.add_argument('--seed', type=str, default='the', help='initial seed')
    args = parser.parse_args()

    # Set device
    device = device("cuda") if cuda.is_available() else device("cpu")
    #print('Selected device:', device)

    # Set the random seed manually for reproducibility.
    _ = torch.manual_seed(42)

    # Fix minimum sentence length
    min_len = 18
    # Tokenize data
    dataset = Bible('./data/bible.txt', min_len = min_len)

    # Define transformation
    dataset.transform = transforms.Compose([
        WordToIndex(dataset.words),
        ToTensor()
    ])

    # Initialize decoder ad encoder
    w2i = WordToIndex(dataset.words)
    # Set number of word to be generated
    n = args.n
    # Number of words known
    ntokens = len(dataset.words)
    # Retrieve training arguments
    training_args = json.load(open('{}/training_json'.format(out_dir)))

    # Build the model
    net = TransformerModel(ntokens,
                          training_args['emsize'],
                          training_args['nhead'],
                          training_args['nhid'],
                          training_args['nlayers'],
                          training_args['dropout']).to(device)

    # Update the weights
    net.load_state_dict(torch.load('{}/net_params.pth'.format(out_dir), map_location='cpu'))
    net.to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()


    # Set the model in evaluation state
    net.eval()

    # Set initial word
    state = args.seed

    # Text generation function
    generate_text(n, state, dataset.words, net, w2i, ntokens, device)
    print()
