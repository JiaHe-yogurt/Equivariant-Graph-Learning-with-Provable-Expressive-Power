import numpy as np
import data_loader.data_helper as helper
import Utils.CONFIG


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.batch_size
        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        #graphs, labels = helper.isotest(self.config.dataset_name)
        #graphs, labels = helper.test_case_1_and_2(self.config.target_shape, False, self.config.input_order)
        #graphs, labels = helper.test_case_3(self.config.target_shape, False, self.config.input_order)
        graphs, labels = helper.hierarchy_load_dataset(self.config.dataset_name, self.config.input_order)
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(graphs, labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            #train_idx = [idx for idx in train_idx if idx not in deleted]
            #test_idx = [idx for idx in test_idx if idx not in deleted]
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
        # change validation graphs to the right shape
        self.val_graphs = [np.expand_dims(g, 0) for g in self.val_graphs]
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        return next(self.iter)

    # initialize an iterator from the data for one training epoch
    def initialize(self, is_train):
        if is_train:
            self.reshuffle_data()
        else:
            self.iter = zip(self.val_graphs, self.val_labels)

    # resuffle data iterator between epochs
    def reshuffle_data(self):
        graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
        graphs, labels = helper.shuffle_same_size(graphs, labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = helper.shuffle(graphs, labels)
        self.iter = zip(graphs, labels)

