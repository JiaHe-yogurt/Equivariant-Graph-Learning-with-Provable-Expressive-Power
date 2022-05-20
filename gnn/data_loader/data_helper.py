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

from subgraphcount.dataset_synthetic import DglSyntheticDataset, collate_lrp_dgl_light
from torch.utils.data import DataLoader

from subgraphcount.model_synthetic import *



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



def get_cliques_by_length(G, length_clique):
    """ Return the list of all cliques in an undirected graph G with length
    equal to length_clique. """
    cliques = []
    for c in nx.enumerate_all_cliques(G) :
        if len(c) <= length_clique:
            if len(c) == length_clique:
                cliques.append(c)
        else:
            return cliques
    # return empty list if nothing is found
    return cliques


def construct_A3(G, length_clique=3):
      tri=get_cliques_by_length(G,3)
      #print(tri)
      nn = G.number_of_nodes()
      A3=np.zeros((nn,nn,nn), dtype='float32')
      for i in tri:
        perm = permutations(i)
        for j in list(perm):
          A3[j]=1
      return A3



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

def k_times_k_plus_1_order_k_minus_1(adj,order=2):
  #adj = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
  nn = adj.shape[0]
  A_shape = tuple([order+1]+[order] + [nn]*(order))
  A = np.zeros((A_shape), dtype='float32')
  for feature1 in range(order+1):
    for feature2 in range(order):
      for i in range(nn):
        for j in range(nn):
          for k in range(nn):
            if i!=j and j!=k and i!=k:
              cur1=[(i,j),(j,k),(i,k)]
              cur=[[adj[i,k],adj[j,k]],[adj[i,k],adj[j,i]],[adj[i,j],adj[j,k]]]
              A[feature1,feature2][cur1[feature1]]=cur[feature1][feature2]
  A = A.reshape((order*(order+1),nn,nn))
  return A





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
    if shape == '6_nodes_1_1_left':   ## 1.1 left
        target.add_edge(0, 1)
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(0, 5)
    if shape == '6_nodes_1_1_right':   ## 1.1 right
        target.add_edge(0, 1)
        target.add_edge(0, 2)
        target.add_edge(2, 1)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(3, 5)
    if shape == '6_nodes_1_2_left':   ## 1.2 left
        target.add_edge(0, 1)
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(0, 5)
        target.add_edge(2, 5)
    if shape == '6_nodes_1_2_right':    ## 1.2 right
        target.add_edge(0, 1)
        target.add_edge(0, 2)
        target.add_edge(2, 1)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(3, 5)
        target.add_edge(3, 0)
    if shape == '10_nodes_1_1_left':    ## 5.1 left
        target.add_edge(0, 1)
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(5, 6)
        target.add_edge(6, 7)
        target.add_edge(7, 8)
        target.add_edge(8, 9)
        target.add_edge(9, 0)
    if shape == '10_nodes_1_1_right':  ## 5.1 right
        target.add_edge(0, 1)
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(0, 4)

        target.add_edge(5, 6)
        target.add_edge(6, 7)
        target.add_edge(7, 8)
        target.add_edge(8, 9)
        target.add_edge(9, 5)
    if shape == '10_nodes_1_2_left':    ## 5.2 left
        target.add_edge(0, 1)
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(4, 5)
        target.add_edge(5, 6)
        target.add_edge(6, 7)
        target.add_edge(7, 8)
        target.add_edge(8, 9)
        target.add_edge(9, 0)
        target.add_edge(5, 0)

    if shape == '10_nodes_1_2_right':  ## 5.1 right
        target.add_edge(0, 1)
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(0, 4)
        target.add_edge(5, 6)
        target.add_edge(6, 7)
        target.add_edge(7, 8)
        target.add_edge(8, 9)
        target.add_edge(9, 5)
        target.add_edge(9, 0)



    if shape == 'rook':
        target = nx.Graph()
        target.add_nodes_from(np.arange(0, 16))
        for i in range(0, 4):
            for subset in combinations([i, i + 4, i + 2 * 4, i + 3 * 4], 2):
                target.add_edge(subset[0], subset[1])

        for i in range(0, 16, 4):
            for subset in combinations(np.arange(i, i + 4), 2):
                target.add_edge(subset[0], subset[1])
    if shape == 'shrik':
        target=Shrikhande_graph()
    return target

def  Shrikhande_graph():
    target = nx.Graph()
    target.add_nodes_from(np.arange(0,16))
    for j in [1,2,6,7,8,13]:
        target.add_edge(0,j)
    for j in [0,3,6,7,9,11]:
        target.add_edge(1,j)
    for j in [0,4,6,8,10,12]:
        target.add_edge(2,j)
    for j in [1,5,6,9,10,14]:
        target.add_edge(3,j)
    for j in [2,5,6,11,12,15]:
        target.add_edge(4,j)
    for j in [4,3,6,13,14,15]:
        target.add_edge(5,j)
    for j in [5,4,3,2,1,0]:
        target.add_edge(6,j)
    for j in [1,0,14,13,12,11]:
        target.add_edge(7,j)
    for j in [2,0,15,13,10,9]:
        target.add_edge(8,j)
    for j in [8,3,1,15,11,10]:
        target.add_edge(9,j)
    for j in [9,8,3,2,14,12]:
        target.add_edge(10,j)
    for j in [9,7,4,1,15,12]:
        target.add_edge(11,j)
    for j in [11,10,7,4,2,14]:
        target.add_edge(12,j)
    for j in [8,7,5,0,15,14]:
        target.add_edge(13,j)
    for j in [13,12,10,7,5,3]:
        target.add_edge(14,j)
    for j in [13,11,9,8,5,4]:
        target.add_edge(15,j)
    return target

def isotest(ds_name):
    graphs,  labels = [], []
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(2, 1)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(3, 5)

    G2 = nx.Graph()
    G2.add_edge(0, 1)
    G2.add_edge(1, 2)
    G2.add_edge(2, 3)
    G2.add_edge(3, 4)
    G2.add_edge(4, 5)
    G2.add_edge(0, 5)
    # G.add_edge(0,1)
    # G.add_edge(0,0)
    # G.add_edge(0,2)
    # G.add_edge(2,1)
    # G.add_edge(1,3)
    # G.add_edge(2,3)
    # G.add_edge(4,5)
    # G.add_edge(4,6)
    # G.add_edge(5,6)
    # G.add_edge(5,7)
    # G.add_edge(7,6)

    # G2=nx.Graph()
    # G2.add_edge(0,5)
    # G2.add_edge(0,0)
    # G2.add_edge(0,2)
    # G2.add_edge(2,1)
    # G2.add_edge(1,3)
    # G2.add_edge(2,3)
    # G2.add_edge(1,4)
    # G2.add_edge(4,6)
    # G2.add_edge(5,6)
    # G2.add_edge(5,7)
    # G2.add_edge(7,6)

    # target = motif(target_shape)
    GRAPH = [G, G2]
    # graph_dict=dict(zip([5,6,6, 6, 7,8, 9, 9, 10,10], [0.7,0.4,0.5, 0.6, 0.4,0.4,0.4, 0.3, 0.4, 0.3]))
    # graph_dict = dict(zip([8, 9, 9, 10, 10, 11, 11, 12, 13], [0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.2, 0.2]))
    # graph_dict = dict(zip([8, 8,8,8,8], [0.3,0.4,0.5,0.6,0.2]))
    num_rep = [500, 500, 1000, 1000, 500, 500, 500, 500, 500, 500]
    for num, g in zip(num_rep, GRAPH):
        # _, label, _ = high_order(g, target)
        for s in range(num):
            # if nx.is_connected(G):
            node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
            g_new = nx.relabel_nodes(g, node_mapping)
            #graph = three_order_four(g_new)
            adj_new = nx.to_numpy_matrix(g_new, nodelist=list(range(len(g_new.nodes))))
            #graphs.append(np.expand_dims(adj_new,axis=0))
            graph = k_minus_1_order_k(adj_new, order=3)
            #graph = k_times_k_plus_1_order_k_minus_1(adj_new, order=2)
            graphs.append(graph)
            # graphs3d2.append(graph3d2)
            # labels.append(label)
            labels.append(int(nx.is_isomorphic(g_new, G2)))
    # for i in range(len(graphs3d)):
    #     graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
    # graphs3d2[i] = np.expand_dims(graphs3d2[i], axis=0)
    graphs = np.array(graphs)
    # graphs3d2 = np.array(graphs3d2)
    return graphs, np.array(labels)




def test_case_1_and_2( target_shape, directed=False, input_order=3):
    d={'6_nodes_1_1_left':  '6_nodes_1_1_right', '6_nodes_1_1_right':'6_nodes_1_1_left', '6_nodes_1_2_left':'6_nodes_1_2_right',
       '6_nodes_1_2_right': '6_nodes_1_2_left' }
    target = motif(target_shape, directed, star_node=3)
    K, V = [target.number_of_nodes()]*4 , [0.3,0.4,0.5,0.6]
    num_rep = [1000] * len(K)
    graphs, labels= [], []
    for num, k, v in zip(num_rep, K, V):
        for s in range(num):
            #g = nx.erdos_renyi_graph(k, v, seed=s, directed=directed)
            #graph3d, label, _ = high_order(G, target)
            g = nx.erdos_renyi_graph(6, 0.3, seed=1, directed=directed)
            adj=nx.to_numpy_array(g)
            if input_order==2:
                graphs.append(np.expand_dims(adj,axis=0))
            elif input_order == 3:
                graph = k_minus_1_order_k(adj, order=3)
                graphs.append(graph)
            else:
                graph = construct_A3(g)
                graphs.append(graph)
            label = 0 if nx.is_isomorphic(g, target)==False else 1
            labels.append(label)
    for num in range(1000):
            node_mapping = dict(zip(target.nodes(), sorted(target.nodes(), key=lambda k: random.random())))
            g= nx.relabel_nodes(target, node_mapping)
            adj_new = nx.to_numpy_matrix(g, nodelist=list(range(len(g.nodes))))
            #g= target
            #adj_new = nx.to_numpy_matrix(g)

            if input_order==2:
                graphs.append(np.expand_dims(adj_new,axis=0))
            elif input_order == 3:
                graph = k_minus_1_order_k(adj_new, order=3)
                graphs.append(graph)
            else:
                graph = construct_A3(g)
                graphs.append(graph)
            label = 0 if nx.is_isomorphic(g, target) == False else 1
            labels.append(label)
    for num in range(0):
            g = motif(d[target_shape], directed, star_node=3)
            node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
            g= nx.relabel_nodes(g, node_mapping)
            adj_new = nx.to_numpy_matrix(g, nodelist=list(range(len(g.nodes))))
            if input_order==2:
                graphs.append(np.expand_dims(adj_new,axis=0))
            elif input_order == 3:
                graph = k_minus_1_order_k(adj_new, order=3)
                graphs.append(graph)
            else:
                graph = construct_A3(g)
                graphs.append(graph)
            label = 0 if nx.is_isomorphic(g, target) == False else 1
            labels.append(label)

    labels = np.array(labels)
    #graphs3d=tf.ragged.constant(graphs3d).to_tensor().eval(session=tf.Session())
    graphs = np.array(graphs)
    return  graphs, labels


def test_case_3( target_shape, directed=False, input_order=3):
    target=test3_graphs(target_shape)
    K, V = [target.number_of_nodes()]*4 , [0.3,0.4,0.5,0.6]
   # num_rep = [100] * len(K)
    num_rep = [250] * len(K)
    graphs, labels= [], []
    for num, k, v in zip(num_rep, K, V):
        for s in range(num):
            g = nx.erdos_renyi_graph(k, v, seed=s, directed=directed)
            #graph3d, label, _ = high_order(G, target)
            adj=nx.to_numpy_array(g)
            if input_order==2:
                graphs.append(np.expand_dims(adj,axis=0))
            elif input_order == 3:
                graph = k_minus_1_order_k(adj, order=3)
                graphs.append(graph)
            else:
                graph = construct_A3(g)
                graphs.append(graph)
            label = 0 if nx.is_isomorphic(g, target)==False else 1
            labels.append(label)
    for num in range(num_rep[0]):
            node_mapping = dict(zip(target.nodes(), sorted(target.nodes(), key=lambda k: random.random())))
            g= nx.relabel_nodes(target, node_mapping)
            # graph = three_order_four(g_new)
            adj_new = nx.to_numpy_matrix(g, nodelist=list(range(len(g.nodes))))
            if input_order==2:
                graphs.append(np.expand_dims(adj_new,axis=0))
            elif input_order == 3:
                graph = k_minus_1_order_k(adj, order=3)
                graphs.append(graph)
            else:
                graph = construct_A3(g)
                graphs.append(graph)
            label = 0 if nx.is_isomorphic(g, target) == False else 1
            labels.append(label)
    for num in range(0):
        if target_shape == 'left':
            opposite_shape = 'right'
        elif target_shape == 'right':
            opposite_shape = 'left'
        g = test3_graphs(opposite_shape)
        node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
        g = nx.relabel_nodes(g, node_mapping)
        adj_new = nx.to_numpy_matrix(g, nodelist=list(range(len(g.nodes))))
        if input_order == 2:
            graphs.append(np.expand_dims(adj_new, axis=0))
        elif input_order == 3:
            graph = k_minus_1_order_k(adj, order=3)
            graphs.append(graph)
        else:
            graph = construct_A3(g)
            graphs.append(graph)
        label = 0 if nx.is_isomorphic(g, target) == False else 1
        labels.append(label)

    labels = np.array(labels)
    #graphs3d=tf.ragged.constant(graphs3d).to_tensor().eval(session=tf.Session())
    graphs = np.array(graphs)
    return  graphs, labels


### test3
def test3_graphs(left_or_right):
    target=nx.Graph()
    target.add_nodes_from([0,40])
    if left_or_right=='left':
      edges=[(0,36),(0,37),(1,5),(1,20),(1,18),(2,5),(2,21),(2,19),(3,22),(3,21),(3,18),
           (4,22),(4,20),(4,19),(5,1),(5,2),(5,6),(6,7),(6,9),(7,24),(7,11),
           (8,23),(8,25),(8,11),(9,25),(9,26),(10,23),(10,26),(10,24),
           (11,12),(12,13),(12,14),(13,17),(13,28),(14,29),(14,30),(15,17),
           (15,29),(15,27),(16,27),(16,28),(16,30),(17,18),(19,30),(22,23),
           (24,34),(25,33),(26,27),(28,32),(29,31),(31,35),(31,37),(32,36),
           (32,38),(33,35),(33,36),(34,37),(34,38),(35,40),(38,40),(20,40),(21,0)]
    elif left_or_right=='right':
      edges = [(0, 36), (0, 37), (1, 5), (1, 20), (1, 18), (2, 5), (2, 21), (2, 19), (3, 22), (3, 21), (3, 18),
                 (4, 22), (4, 20), (4, 19), (5, 1), (5, 2), (5, 6), (6, 7), (6, 9), (7, 24), (7, 11),
                 (8, 23), (8, 25), (8, 11), (9, 25), (9, 26), (10, 23), (10, 26), (10, 24),
                 (11, 12), (12, 13), (12, 14), (13, 17), (13, 28), (14, 29), (14, 30), (15, 17),
                 (15, 29), (15, 27), (16, 27), (16, 28), (16, 30), (17, 18), (19, 30), (22, 23),
                 (24, 34), (25, 33), (26, 27), (28, 32), (29, 31), (31, 35), (31, 37), (32, 36),
                 (32, 38), (33, 35), (33, 36), (34, 37), (34, 38), (35, 40), (38, 40), (20, 0), (21, 40)]
    for edge in edges:
        target.add_edge(edge[0],edge[1])
    return target



## benchmark dataset
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

