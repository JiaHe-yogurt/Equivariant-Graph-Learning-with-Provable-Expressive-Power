import networkx as nx
from sklearn import preprocessing
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import random
from itertools import permutations, combinations
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from numpy.linalg import matrix_power
from scipy import sparse
import pickle
import copy
import math
from networkx.algorithms import *

tf.disable_eager_execution()

NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38, 'PROTEINS': 3, 'PTC': 22, 'DD': 89}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx = []
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx = []
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels, graphs3d):
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
    graphs3d = graphs3d[indexes]
    r_graphs = []
    r_labels = []
    r_graphs3d = []
    one_size = []
    one_size_node = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
            one_size_node.append(np.expand_dims(graphs3d[i], axis=0))

        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_graphs3d.append(np.concatenate(one_size_node, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            one_size_node = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
            one_size_node.append(np.expand_dims(graphs3d[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    r_graphs3d.append(np.concatenate(one_size_node, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels, r_graphs3d


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels, graphs3d):
    r_graphs, r_labels, r_graphs3d = [], [], []
    for i in range(len(labels)):
        curr_graph, curr_labels, curr_nodefeature = shuffle(graphs[i], labels[i], graphs3d[i])
        r_graphs.append(curr_graph)
        r_graphs3d.append(curr_nodefeature )
        r_labels.append(curr_labels)
    return r_graphs, r_labels, r_graphs3d





def split_to_batches(graphs, labels, graphs3d, size):
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
    r_graphs3d = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        r_graphs3d = r_graphs3d + np.split(graphs3d[k], [j for j in range(size, graphs3d[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
    return np.array(r_graphs), np.array(r_labels), np.array(r_graphs3d)


# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels, graphs3d):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    #np.random.seed(1)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf], np.array(graphs3d)[shf]





def get_cliques_by_length(G, length_clique):
    """ Return the list of all cliques in an undirected graph G with length
    equal to length_clique. """
    cliques = []
    for c in nx.enumerate_all_cliques(G):
        if len(c) <= length_clique:
            if len(c) == length_clique:
                cliques.append(c)
        else:
            return cliques
    # return empty list if nothing is found
    return cliques




def construct_upperA3(G):
    tri = get_cliques_by_length(G, 3)
    # print(tri)
    nn = G.number_of_nodes()
    A3 = np.zeros((nn, nn, nn), dtype='float32')
    for i in tri:
        A3[tuple(i)] = 1
    return A3


def motif(shape, directed=False, star_node=None):
    if directed:
       target = nx.DiGraph()
    else:
       target = nx.Graph()
    if shape == 'tree':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
    if shape == 'triangle':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(1, 3)
    if shape == 'tail_triangle':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(1, 3)
        target.add_edge(1, 4)
    if shape == 'star':
        target = nx.star_graph(star_node)
    if shape == 'chain':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
    if shape == 'box':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(1, 4)
    if shape == 'semi_clique':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(1, 4)
        target.add_edge(1, 3)
    if shape == '4_clique':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(1, 4)
        target.add_edge(1, 3)
        target.add_edge(2, 4)
    if shape == '5_nodes':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(1, 5)
    if shape == '5_nodes_1':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 1)
        target.add_edge(4, 3)
        target.add_edge(3, 5)
    if shape == '5_nodes_2':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 1)
        target.add_edge(4, 3)
        target.add_edge(4, 2)
        target.add_edge(5, 2)

    return target



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



def multihead_GCN(ds_name, input_order, num_hop=3):
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs3d, graphs2d,  labels =[], [], []
    cn0 = [] #, [], [], [], []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
            labels.append(int(graph_meta[1]))  # ori
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0]) + 1] = 1.
                for k in range(2, len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            curr_graph = noramlize_graph(curr_graph)
            graphs2d.append(curr_graph)
    graphs2d = np.array(graphs2d)
    labels = np.array(labels)


    for i in range(graphs2d.shape[0]):
        graphs2d[i] = np.transpose(graphs2d[i], [2,0,1])
        nodals = np.array([np.diagonal(nodal) for nodal in graphs2d[i][1:]]).transpose()
        gsp0 = convolved_features(graphs2d[i][0], nodals, num_hop)  ## H=1
        cn0.append(gsp0) #, cn1.append(gsp1), cn2.append(gsp2), cn3.append(gsp3), cn4.append(gsp4)
        if input_order == 2:
            graphs2d[i] = np.expand_dims(graphs2d[i][0],axis=0)
        if input_order == 3:
            graphs3d.append(k_minus_1_order_k(graphs2d[i][0], order=3).transpose(1,2,3,0))
    cn0 = np.array(cn0) #, np.array(cn1), np.array(cn2), np.array(cn3), np.array(cn4)
    if input_order == 3:
         graphs3d = np.array(graphs3d)
    for i in range(cn0.shape[0]):
        cn0[i]= np.transpose(cn0[i], [1,0,2])
        #cn1[i] cn2[i], cn3[i], cn4[i] = np.transpose(cn2[i], [1,0,2]), np.transpose(cn3[i], [1,0,2]), np.transpose(cn4[i], [1,0,2])
        if input_order == 3:
          graphs3d[i] = np.transpose(graphs3d[i], [3,0,1,2])
    if input_order == 3:
        return np.array(labels), cn0, graphs3d
    else:
        return np.array(labels), cn0, graphs2d


def convolved_features(GSO,inputs,num_hop):
    #D_inv = np.diag(np.sum(GSO, axis=0) ** -0.5)
    #GSO = np.matmul(np.matmul(D_inv, GSO), D_inv)
    supports = [inputs]
    for i in range(1, num_hop + 1):
        aggregate = np.matmul(GSO ** i,inputs)
        supports.append(aggregate)
    return  np.array(supports).transpose(1,0,2)









