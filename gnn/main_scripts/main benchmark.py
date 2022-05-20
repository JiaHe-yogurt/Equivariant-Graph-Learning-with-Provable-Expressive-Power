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

    ### benchmark data
    data = DataGenerator(config)
    data.config.num_classes = len(collections.Counter(data.train_labels))
    # config.dataset_size = data.train_graphs.shape[2]
    with open(config.summary_dir+'config', 'w') as f:
                json.dump(config, f)
    sess = tf.Session(config=gpuconfig)

    # create an instance of the model you want
    model = invariant_basic(config, data)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config)
    trainer.train()
    test_acc, test_loss, pred, true = trainer.test(trainer.best_epoch, load_best_model=True)

    if config.task == 'classification':
        res = acc(pred.reshape(-1, ), data.val_labels)
    if config.task == 'regression':
        res = mse(pred.reshape(-1, ), data.val_labels)

    sess.close()
    tf.reset_default_graph()



