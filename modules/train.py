import time
import torch
import numpy as np
from torch import nn
from torch import cuda, device, save


def get_batch(source, i, bptt = 35):
    """
    Generates the input and target sequence for the transformer model. It
    subdivides the source data into chunks of length bptt. For the language
    modeling task, the model needs the following words as Target.
    (From Pytorch doc:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    """

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    (From Pytorch doc:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    """

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(net, loss_fn, data_source, ntokens, bptt = 35):
    """
    From Pytorch doc:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    # Turn on evaluation mode which disables dropout.
    net.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            inputs, targets = get_batch(data_source, i)
            inputs = inputs.long()
            targets = targets.long()
            output = net(inputs)
            output_flat = output.view(-1, ntokens)
            total_loss += len(inputs) * loss_fn(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train(net, optimizer, loss_fn, data, ntokens, clip, epoch, bptt = 35, log_interval = 100):

    # Turn on training mode which enables dropout.
    net.train()

    total_loss = 0.
    start_time = time.time()
    temp_loss = []

    for batch, i in enumerate(range(0, data.size(0) - 1, bptt)):
        inputs, targets = get_batch(data, i)
        inputs = inputs.long()
        targets = targets.long()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the net would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        output = net(inputs)

        loss = loss_fn(output.view(-1, ntokens), targets)
        loss.backward()

        optimizer.step()

        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch, len(data) // bptt,
                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
            temp_loss.append(float(cur_loss))
    return float(np.mean(temp_loss))


def generate_text(n, state, words, net, w2i, ntokens, device = 'cuda'):
    # Extract last word
    word = state.split()[-1]
    # Handle the situation where the seed is not contained in the dictionary
    if word in words:
        input = torch.tensor(np.reshape(w2i(word), (1, -1))).long().to(device)
    else:
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    # Generate next word
    with torch.no_grad():  # no tracking history
        for i in range(n):
            # Get output
            output = net(input, False)
            word_weights = output[-1].squeeze().exp().cpu()

            # Sample word from output distribution
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)

            # Concatenate the word predicted with the current state
            input = torch.cat([input, word_tensor], 0)
            word = w2i.decoder[word_idx.item()]
            state = '{} {}'.format(state, word)

    # Set punctuations signs and upper case signs
    punc = ['!', '?', '.', ';', ':', ',',"'"]
    upcase = ['?',  '!',  '.']

    # Set initial params
    after_point = False
    new_line_counter = 0
    previous = '_'

    # Print initial state
    print('TEXT:')
    print('{}'.format(state.split()[0]), end = '')

    # Print next word following some given rules
    for i in state.split()[1:]:
        # Avoid loops
        if i == previous:
            continue

        # Update
        previous = i

        # Increment
        new_line_counter += 1

        # Flag: next word capitalized
        if i in upcase:
          after_point = True

        # Flag: start newline after a full point
        if i == '.' and new_line_counter > 10:
          new_line_counter = 0
          print('.')

        # Flag: do not add whitespace, there is punctuation
        elif i in punc:
          print(i, end='')
          new_line_counter -= 1

        # Print new word following flags
        else:
          if after_point:
            if new_line_counter > 1:
                print(' {}'.format(i.capitalize()), end='')
                after_point=False
            # After newline, no whitespace added
            else:
                print('{}'.format(i.capitalize()), end='')
                after_point=False
          else:
            print(' {}'.format(i), end='')
