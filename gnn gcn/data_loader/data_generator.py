import numpy as np
import data_loader.data_helper as helper
import Utils.config

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.batch_size
        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        labels, graphs3d, graphs= helper.multihead_GCN(self.config.dataset_name,self.config.input_order,self.config.num_hop)  # count subgraph
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels, graphs3d = helper.shuffle(graphs, labels, graphs3d)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels,  self.train_graphs3d, self.val_graphs, self.val_labels, self.val_graphs3d  = graphs[idx:], labels[idx:], graphs3d[idx:], graphs[:idx], labels[:idx], graphs3d[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels,  self.train_graphs3d, self.val_graphs, self.val_labels , self.val_graphs3d = graphs[train_idx], labels[
                train_idx],  graphs3d[train_idx], graphs[test_idx], labels[test_idx], graphs3d[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(1, self.config.dataset_name)
            self.train_graphs, self.train_graphs3d ,self.train_labels, self.val_graphs, self.val_graphs3d, self.val_labels = graphs[train_idx], graphs3d[train_idx], labels[train_idx], graphs[test_idx], graphs3d[test_idx], labels[test_idx]
        # change validation graphs to the right shape
        self.val_graphs = [np.expand_dims(g, 0) for g in self.val_graphs]
        self.val_graphs3d = [np.expand_dims(g, 0) for g in self.val_graphs3d]
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        return next(self.iter)

    # initialize an iterator from the data for one training epoch
    def initialize(self, is_train):
        if is_train:
            self.reshuffle_data()
        else:
            self.iter = zip(self.val_graphs, self.val_labels, self.val_graphs3d)

    # resuffle data iterator between epochs
    def reshuffle_data(self):
        graphs, labels, graphs3d = helper.group_same_size(self.train_graphs, self.train_labels, self.train_graphs3d)
        graphs, labels, graphs3d = helper.shuffle_same_size(graphs, labels, graphs3d)
        graphs, labels, graphs3d = helper.split_to_batches(graphs, labels, graphs3d, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels, graphs3d = helper.shuffle(graphs, labels, graphs3d)
        self.iter = zip(graphs, labels, graphs3d)










