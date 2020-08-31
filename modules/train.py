import time
import torch
import torch.onnx
from torch import nn
from torch import cuda, device, save

'''
From Pytorch doc:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

def get_batch(source, i, bptt = 35):

    """
    Generates the input and target sequence for the transformer model. It
    subdivides the source data into chunks of length bptt. For the language
    modeling task, the model needs the following words as Target.
    """

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(net, data_source, ntokens):
    # Turn on evaluation mode which disables dropout.
    net.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            inputs, targets = get_batch(data_source, i)
            output = net(inputs)
            output_flat = output.view(-1, ntokens)
            total_loss += len(inputs) * loss_fn(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train(net, optimizer, loss_fn, data, ntokens, bptt = 35, log_interval = 100):
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
    training_loss.append(float(np.mean(temp_loss)))

def export_onnx(path, batch_size, seq_len):
    print('The net is also exported in ONNX format at {}'.
          format(path))
    net.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = net.init_hidden(batch_size)
    torch.onnx.export(net, (dummy_input, hidden), path)
