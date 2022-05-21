    from data_loader.data_generator import DataGenerator
    from models.invariant_basic import invariant_basic
    from trainers.trainer import Trainer
    from Utils.config import process_config
    from Utils.dirs import create_dirs
    import numpy as np
    from  collections import Counter
    from Utils.utils import get_args
    from Utils import config
    import warnings
    warnings.filterwarnings('ignore')
    import importlib
    import collections
    import data_loader.data_helper as helper
    from Utils.utils import get_args
    import os
    import time
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config('../gnn multiple inputs/configs/example.json')
   # config.num_classes=4
     """reset config.num_classes if it's syn data"""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    # create the experiments dirs
    tf.set_random_seed(1)
    np.random.seed(1)
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
   # gpuconfig = tf.ConfigProto(intra_op_parallelism_threads=2, allow_soft_placement=True, log_device_placement=True)
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    gpuconfig.gpu_options.allow_growth = True
    # create your data generator
    np.random.seed(1)
    #config.num_fold = 1
    data = DataGenerator(config)
    #config.dataset_size = int(data.train_graphs.shape[2])
    data.config.num_epochs = 1
    sess = tf.Session(config=gpuconfig)

    # create an instance of the model you want
    model = invariant_basic(config, data)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config)
    # load model if exists
    # here you train your model
    stt = time.time()
    trainer.train()
    end = time.time()
    sess.close()
    tf.reset_default_graph()





