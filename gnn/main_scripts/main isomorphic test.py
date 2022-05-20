    from data_loader.data_generator import DataGenerator
    from models.invariant_basic import invariant_basic
    from trainers.trainer import Trainer
    from Utils.CONFIG import process_config
    from Utils.dirs import create_dirs
    import numpy as np
    from collections import Counter
    from Utils.utils import get_args
    from Utils import CONFIG
    import warnings
    import networkx as nx
    warnings.filterwarnings('ignore')
    import importlib
    import collections
    import data_loader.data_helper as helper
    from Utils.utils import get_args
    import os
    import time
    import json
    from sklearn.metrics import accuracy_score as acc
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import mean_squared_error as mse

    # capture the config path from the run arguments
    # then process the json configuration file
    config = os.getcwd()+str('/CONFIGS/example.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpuconfig.gpu_options.visible_device_list = config.gpus_list
    gpuconfig.gpu_options.allow_growth = True


    results=collections.defaultdict(list)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    subgraph_shape = '6_nodes_1_1_left'
    target_shape = '6_nodes_1_1_right'
    g = helper.motif(target_shape, directed=False, star_node=3)
    input_graph = helper.motif(subgraph_shape, directed=False, star_node=3)
    adj = nx.to_numpy_array(g)
    graph1 = np.expand_dims(helper.k_minus_1_order_k(adj, order=3), axis=0)
    adj = nx.to_numpy_array(input_graph)
    graph2 = np.expand_dims(helper.k_minus_1_order_k(adj, order=3), axis=0)
    graphs = np.concatenate([graph1, graph2])


    config = process_config('/Users/jiahe/PycharmProjects/colab gn/CONFIGS/example.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # create tensorflow session
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpuconfig.gpu_options.visible_device_list = config.gpus_list
    gpuconfig.gpu_options.allow_growth = True

    base_summary_folder = config.summary_dir
    base_exp_name = config.exp_name

    def test_graph_nonisomorphic(subgraph_shape, input_order):
                if subgraph_shape == '6_nodes_1_1_left':
                    target_shape = '6_nodes_1_1_right'
                elif subgraph_shape == '6_nodes_1_1_right':
                    target_shape = '6_nodes_1_1_left'
                elif subgraph_shape == '6_nodes_1_2_left':
                    target_shape = '6_nodes_1_2_right'
                elif subgraph_shape == '6_nodes_1_2_right':
                    target_shape = '6_nodes_1_2_left'
                elif subgraph_shape == '10_nodes_1_1_left':
                    target_shape = '10_nodes_1_1_right'
                elif subgraph_shape == '10_nodes_1_1_right':
                    target_shape = '10_nodes_1_1_left'
                elif subgraph_shape == '10_nodes_1_2_left':
                    target_shape = '10_nodes_1_2_right'
                elif subgraph_shape == '10_nodes_1_2_right':
                    target_shape = '10_nodes_1_2_left'
                elif subgraph_shape == 'rook':
                    target_shape = 'shrik'
                elif subgraph_shape == 'left':     ### this is for test3
                    target_shape = 'right'
                elif subgraph_shape == 'right':
                    target_shape = 'left'
                if target_shape  == 'left' or target_shape  == 'right':
                    g = helper.test3_graphs(target_shape)
                    input_graph = helper.test3_graphs(subgraph_shape)
                else:
                    g = helper.motif(target_shape, directed=False, star_node=3)
                    input_graph = helper.motif(subgraph_shape, directed=False, star_node=3)
                adj = nx.to_numpy_array(g)
                graphs = []
                if input_order == 2:
                    graph = np.expand_dims(np.expand_dims(adj, axis=0), axis=0)
                elif input_order == 3:
                    graph = np.expand_dims(helper.k_minus_1_order_k(adj, order=3), axis=0)

                label = 0 if nx.is_isomorphic(g, input_graph)==False else 1
                label = np.array([label]*2)
                graphs = [graph]*2
                return graphs, label

    def test_graph_isomorphic(subgraph_shape, input_order):
                target_shape =  subgraph_shape
                if target_shape  == 'left' or target_shape  == 'right':
                    g = helper.test3_graphs(target_shape)
                    input_graph = helper.test3_graphs(subgraph_shape)
                else:
                    g = helper.motif(target_shape, directed=False, star_node=3)
                    input_graph = helper.motif(subgraph_shape, directed=False, star_node=3)
                adj = nx.to_numpy_array(g)
                graphs = []
                if input_order == 2:
                    graph = np.expand_dims(np.expand_dims(adj, axis=0), axis=0)
                elif input_order == 3:
                    graph = np.expand_dims(helper.k_minus_1_order_k(adj, order=3), axis=0)

                label = 0 if nx.is_isomorphic(g, input_graph)==False else 1
                label = np.array([label]*2)
                graphs = [graph]*2
                return graphs, label

    data = DataGenerator(config)
    data.config.num_classes = len(collections.Counter(data.train_labels))
    create_dirs([config.summary_dir, config.checkpoint_dir])

    with open(config.summary_dir + 'config', 'w') as f:
                json.dump(config, f)
    sess = tf.Session(config=gpuconfig)
            # create an instance of the model you want
    model = invariant_basic(config, data)
    trainer = Trainer(sess, model, data, config)
    trainer.train()

    data.val_graphs, data.val_labels = test_graph_nonisomorphic(config.target_shape, config.input_order)
    data.val_size = len(data.val_graphs)
    test_acc, test_loss, pred, true = trainer.test(trainer.best_epoch, load_best_model=True)

    sess.close()
    tf.reset_default_graph()











