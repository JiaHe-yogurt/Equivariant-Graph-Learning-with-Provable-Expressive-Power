import networkx as nx
from sklearn import preprocessing
import collections
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations, combinations
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from numpy.linalg import matrix_power
import torch
import torch.optim as optim
import torch.nn.functional as F
import os, sys
from tqdm import tqdm
import argparse
import time
import random
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize



NUM_LABELS = {'ENZYMES':3, 'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'DD':89}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
            in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
            the second arrayy is labels with correspons shape
    """
    sizes = list(map(lambda t: t.shape[1], graphs))
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    labels = labels[indexes]
    r_graphs = []
    r_labels = []
    one_size = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels):
    r_graphs, r_labels = [], []
    for i in range(len(labels)):
        curr_graph, curr_labels = shuffle(graphs[i], labels[i])
        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)
    return r_graphs, r_labels


def split_to_batches(graphs, labels, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs = []
    r_labels = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
    return np.array(r_graphs), np.array(r_labels)


# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    #np.random.seed(1)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf]



def noramlize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)






def k_minus_1_order_k(adj, order=3):
  #adj = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
  nn = adj.shape[0]
  A_shape = tuple([order-1] + [nn]*(order))
  A = np.zeros((A_shape), dtype='float32')
  for feature in range(A.shape[0]):
    for i in range(nn):
      for j in range(nn):
        for k in range(nn):
          if i!=j and j!=k and i!=k:
            cur=[adj[i,j],adj[j,k]]
            A[feature,i,j,k]=cur[feature]
  return A






## load dataset
def hierarchy_load_dataset(ds_name, input_order):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name]+1), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0])+1] = 1.
                for k in range(2,len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            curr_graph = noramlize_graph(curr_graph)
            graphs.append(curr_graph)

    graphs = np.array(graphs)
    #dim = [graph.shape[0] for graph in graphs]
    #sort = (sorted([(x, i) for (i, x) in enumerate(dim)], reverse=True)[:3000])
    #graphs = np.delete(graphs, ([sort[i][1] for i in range(len(sort))]), axis=0)
    #labels = np.delete(labels, ([sort[i][1] for i in range(len(sort))]), axis=0)

    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2,0,1])
       # if input_order == 2:
       #     A_tower = A_power(graphs[i][0])
       #     graphs[i] = np.concatenate([graphs[i], A_tower])

        if input_order == 3:
            G = nx.from_numpy_array(graphs[i][0])
         #   tri = np.expand_dims(construct_upperA3(G, length_clique=3), axis=0)
            nodal_features = np.zeros((graphs[i].shape[0]-1, graphs[i].shape[1], graphs[i].shape[1], graphs[i].shape[1]))
            for j in range(nodal_features .shape[0]):
                np.fill_diagonal(nodal_features [j - 1], graphs[i][j].diagonal())
            graphs[i] = k_minus_1_order_k(graphs[i][0], order=3)
            graphs[i] = np.concatenate([graphs[i], nodal_features])

    return graphs, np.array(labels)

