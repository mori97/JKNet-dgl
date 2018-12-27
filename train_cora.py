"""Train Jump Knowledge Network with Cora dataset.
"""
import argparse

import dgl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from modules import JKNetConcat

L2_PENALTY = 0.0005


def preprocessing(cites_filename, content_filename, device):
    """Parse the .cites file and .content file.
    Return the graph, features of nodes and the dataset.

    Args:
        cites_filename (str): .cites file.
        content_filename (str): content_file.
        device (torch.Device): The desired device of returned tensor.
    """
    class2index = {}
    paper2index = {}
    xs = []
    ts = []
    with open(content_filename, 'r') as f:
        for line in f:
            words = line.strip().split('\t')
            paper_id = words[0]
            word_attributes = list(map(float, words[1:-1]))
            class_label = words[-1]

            if paper_id not in paper2index:
                paper2index[paper_id] = len(paper2index)
            if class_label not in class2index:
                class2index[class_label] = len(class2index)

            xs.append(word_attributes)
            ts.append(class2index[class_label])

    graph = dgl.DGLGraph()
    graph.add_nodes(len(xs))

    with open(cites_filename, 'r') as f:
        for line in f:
            words = line.strip().split('\t')
            try:
                src = paper2index[words[0]]
                dst = paper2index[words[1]]
                graph.add_edge(src, dst)
            except KeyError:
                continue

    xs = torch.Tensor(xs).to(device)
    idx = np.array(range(graph.number_of_nodes()))
    idx_train, idx_test, ts_train, ts_test = \
        train_test_split(idx, ts, test_size=0.2)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    ts_train = torch.LongTensor(ts_train).to(device)
    ts_test = torch.LongTensor(ts_test).to(device)

    return graph, xs, idx_train, idx_test, ts_train, ts_test


def train(graph, model, xs, idx_train, ts_train, optimizer):
    model.train()
    optimizer.zero_grad()
    ys = F.log_softmax(model(graph, xs), dim=1)
    loss = F.nll_loss(ys[idx_train], ts_train)
    loss.backward()
    optimizer.step()


def evaluate(graph, model, xs, idx_test, ts_test):
    model.eval()
    with torch.no_grad():
        ys = model(graph, xs)[idx_test]
        predict = ys.max(1, keepdim=True)[1]
        n_correct = predict.eq(ts_test.view_as(predict)).sum().item()
        accuracy = n_correct / ts_test.shape[0]
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aggregation',
                        help='The way to aggregate neighbourhoods',
                        type=str, choices=('sum', 'mean', 'max'),
                        default='sum')
    parser.add_argument('--cites-file',
                        help='.cites file',
                        type=str, default='./datasets/cora/cora.cites')
    parser.add_argument('--content-file',
                        help='.content file',
                        type=str, default='./datasets/cora/cora.content')
    parser.add_argument('--epochs', '-e',
                        help='number of epochs to train',
                        type=int, default=100)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning rate',
                        type=float, default=0.005)
    parser.add_argument('--n-layers',
                        help='Number of convolution layers',
                        type=int, default=6)
    parser.add_argument('--n-units',
                        help='Size of middle layers.',
                        type=int, default=16)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    graph, xs, idx_train, idx_test, ts_train, ts_test =\
        preprocessing(args.cites_file, args.content_file, device)

    in_features = xs.shape[1]
    out_features = torch.max(ts_train).item() + 1

    model = JKNetConcat(in_features, out_features, args.n_layers, args.n_units,
                        args.aggregation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=L2_PENALTY)

    accuracy_list = []
    for epoch in range(1, args.epochs + 1):
        train(graph, model, xs, idx_train, ts_train, optimizer)
        accuracy = evaluate(graph, model, xs, idx_test, ts_test)
        accuracy_list.append(accuracy)
        print('Epoch: {}\tAccuracy: {:.2%}'.format(epoch, accuracy))
    print('Best accuracy: {:.2%}'.format(max(accuracy_list)))


if __name__ == '__main__':
    main()
