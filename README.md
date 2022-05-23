# Equivariant-Graph-Learning-with-Provable-Expressive-Power
A TensorFlow implementation of paper "Equivariant Graph Learning with Provable Expressive Power"
## Abstract
Graph neural network (GNN) models have been widely used for learning graph-
structured data. Due to the permutation-invariant requirement of graph learning
tasks, a basic element in graph neural networks is the invariant and equivariant
linear layers. Previous work [24] provided a maximal collection of invariant
and equivariant linear layers and a simple deep neural network model, called k-
IGN, for graph data defined on k-tuples of nodes. It is shown that the expressive
power of k-IGN is equivalent to k-Weisfeiler-Lehman (WL) algorithm in graph
isomorphism tests. However, the dimension of the invariant layer and equivariant
layer is the k-th and 2k-th bell numbers, respectively. Such high complexity makes
it computationally infeasible for k-IGNs with k > 3.
In this paper, we show that a much smaller dimension for the linear layers is
sufficient to achieve the same expressive power. We provide two sets of orthogonal
bases for the linear layers, each with only 3(2k-1)^k basis elements. Based on
these linear layers, we develop neural network models GNN-a and GNN-b, and
show that for the graph data defined on k-tuples of data, GNN-a and GNN-b achieve
the expressive power of the k-WL algorithm and the (k + 1)-WL algorithm in
graph isomorphism tests, respectively. In molecular prediction tasks on benchmark
datasets, we demonstrate that low-order neural network models consisting of the
proposed linear layers achieve better performance than other neural network models.
In particular, order-2 GNN-b and order-3 GNN-a both have 3-WL expressive power,
but use a much smaller basis and hence much less computation time than known
neural network models.

## Data
Benchmark dataset can be downloaded from folder named "data". Data for graph isomorphism test
can be generated using python file data_help.py under folder "gnn/data_loader".

## Code

### Prerequisites

python3

TensorFlow.




### Running the tests

The folder main_scripts contains scripts that run different experiments:

1. To run training and evaluation order-k GNN-a and order-k GNN-b on benchmark data sets run the  gnn/main_scripts/main benchmark.py script
2. To run training and evaluation GCN + order-2 GNN-b and GCN + order-3 GNN-a on benchmark data sets run the gnn gcn/main_scripts/main benchmark.py scriptscript



