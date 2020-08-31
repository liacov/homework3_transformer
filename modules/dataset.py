import re
import torch
import random
import numpy as np
import unidecode as ud
from torch import device, cuda
from torchvision import transforms
from torch.utils.data import Dataset

### CLASSES ###

class WordToIndex(object):

    def __init__(self, words):
        # Store list of words
        self.words = set(words)
        # Define mapping from words to integers
        self.encoder = {e: i for i, e in enumerate(words)}
        # Define mapping from integers to words
        self.decoder = {i: e for i, e in enumerate(words)}

    # Return word as its index
    def __call__(self, sentence):
        # Make list of labels from words
        labels = [self.encoder[w] for w in sentence if w in self.words]
        # Return list of labels
        return labels

    # Return vector of indices as its corresponding words
    def decode(self, sentence):
        # Make list of words from labels
        words = [self.decoder[i] for i in sentence if i in self.decoder.keys()]
        # retrun list of words
        return words


class ToTensor(object):

    def __call__(self, sentence):

        return torch.tensor(sentence).int()


class Bible(Dataset):

    def __init__(self, file_path, min_len = 4, transform=None):

        # Load data
        with open(file_path, 'r') as file:
            text = file.read()
        ## Text preprocessing
        # Remove non-unicode characters
        text = ud.unidecode(text)
        # Lowercase
        text = text.lower()
        # Remove single newlines
        text = re.sub(r'(?<!\n)\n', ' ', text)
        # Remove undesired punctuation and numbers
        text = re.sub(r'[\*\=\/\d]', '', text)
        # Remove undesired symbols between words
        text = re.sub(r'(?<=\D)[-]+(?=(\D))', ' ', text)
        # Remove double spaces
        text = re.sub(r'[\t ]+', ' ', text)
        # Split text into sentences
        sentences = list(re.findall(r'([^\.\!\?\n]+[\.\!\?]+["]{,1}[ ]{,1})', text))
        # Split sentences according to words
        words = []
        sentences_clean = []
        for sentence in sentences:
            # Split sentence into words according to punctuation
            tokens = list(re.split(r'([ \n\:\;\"\,\(\)\.\!\?])', sentence))
            # Remove useless characters
            tokens = [w for w in tokens if re.search('[^| ]', w)]
            # Remove punctuation at the beginning of sentences
            if tokens[0] == ':' and  tokens[1] == ':':
                tokens = tokens[2:]
            elif tokens[0] == ':' and  tokens[1]!= ':':
                tokens = tokens[1:]
            # Filter short sentences
            if len(tokens) >= min_len:
                # Substitute entire sentence with splited one
                sentences_clean.append(tokens)
                # Save words
                words.extend(tokens)
        # Store words
        self.words = set(words)
        # Store sentences transformation pipeline
        self.transform = transform
        # Return sentences
        self.sentences = sentences_clean

    def __len__(self):

        return len(self.sentences)

    def __getitem__(self, i):

        # Case index i is greater or equal than number of sequences
        if i >= len(self.sentences):
            # Raise key error
            raise IndexError('Chosen index exceeds sentences indices')

        # Case transform is not set
        if self.transform is None:
            # Just return i-th sentence
            return self.sentences[i]

        # Otherwise, transform it and return transformation result
        return self.transform(self.sentences[i])


class Mobydick(Dataset):

    def __init__(self, file_path, min_len=4, transform=None):

        # Load data
        with open(file_path, 'r') as file:
            text = file.read()

        ## Text preprocessing
        # Remove non-unicode characters
        text = ud.unidecode(text)
        # Lowarcase
        text = text.lower()
        # Remove single newlines
        text = re.sub(r'(?<!\n)\n', ' ', text)
        # Remove punctuation and numbers
        text = re.sub(r'[\*\=\/\d]', '', text)
        # Remove underscores
        text = re.sub('_', '', text)
        # Remove symbols between words
        text = re.sub(r'(?<=\D)[-]+(?=(\D))', ' ', text)
        # Remove double spaces
        text = re.sub(r'[\t ]+', ' ', text)

        # Split text into sentences
        sentences = list(re.findall(r'([^\.\!\?\n]+[\.\!\?]+["]{,1}[ ]{,1})', text))

        # Split sentences according to words
        words = []
        sentences_clean = []
        for sentence in sentences:
            # Split sentence into words according to punctuation
            tokens = list(re.split(r'([ \n\:\;\"\,\(\)\.\!\?])', sentence))
            # Remove useless characters
            tokens = [w for w in tokens if re.search('[^| ]', w)]
            # filter short sentences
            if len(tokens) >= min_len:
                # Substitute entire sentence with splited one
                sentences_clean.append(tokens)
                # Save words
                words.extend(tokens)

        # Store words
        self.words = set(words)
        # Store sentences
        self.sentences = sentences_clean
        # Store sentences transformation pipeline
        self.transform = transform

    def __len__(self):

        return len(self.sentences)

    def __getitem__(self, i):

        # Case index i is greater or equal than number of sequences
        if i >= len(self.sentences):
            # Raise key error
            raise IndexError('Chosen index exceeds sentences indices')

        # Case transform is not set
        if self.transform is None:
            # Just return i-th sentence
            return self.sentences[i]

        # Otherwise, transform it and return transformation result
        return self.transform(self.sentences[i])


### FUNCTIONS ###

def split_train_test(dataset, train_prc=0.8):
    # Define dataset length
    n = len(dataset)
    # Define number of training dataset indices
    m = round(train_prc * n)
    # Split datasets in two
    train_idx = np.random.choice(n, m)
    train = [dataset[i] for i in range(n) if i in train_idx]
    test = [dataset[i] for i in range(n) if i not in train_idx]
    return torch.cat(train), torch.cat(test)

def batchify(data, bsz):
    """
    From Pytorch doc:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data
