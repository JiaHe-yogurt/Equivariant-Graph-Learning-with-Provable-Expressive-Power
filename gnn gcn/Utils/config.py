import json
from easydict import EasyDict
import os
import datetime

NUM_LABELS = {'syn': 0,'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'ENZYMES':3,'PROTEINS':3, 'PTC':22, 'DD':89, 'QM9': 18}
NUM_FEATURES = {'syn': 0,'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':62, 'NCI1':1372, 'NCI109':38, 'ENZYMES': 0, 'PROTEINS':319, 'PTC':539, 'DD':89, 'QM9': 18}
NUM_CLASSES = {'syn':1, 'COLLAB':3, 'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2,'ENZYMES': 6, 'PROTEINS':2, 'PTC':2, 'QM9': 12}
LEARNING_RATES = {'syn':0.001, 'COLLAB': 0.00256, 'IMDBBINARY': 0.005, 'IMDBMULTI': 0.005, 'ENZYMES': 0.005,'MUTAG': 0.01, 'NCI1':0.00256, 'NCI109':0.00256, 'PROTEINS': 0.00064, 'PTC': 0.00128,'QM9': 0.00128}
DECAY_RATES = {'syn': 0.7, 'COLLAB': 0.7, 'IMDBBINARY': 0.8, 'IMDBMULTI': 0.8, 'MUTAG': 0.7, 'NCI1':0.7, 'NCI109':0.7, 'ENZYMES': 0.7, 'PROTEINS': 0.7, 'PTC': 0.6, 'QM9': 0.6}
CHOSEN_EPOCH = {'syn':30, 'COLLAB': 100, 'IMDBBINARY': 40, 'IMDBMULTI': 150, 'MUTAG': 130, 'NCI1': 99, 'NCI109': 99,'ENZYMES': 50, 'PROTEINS': 20, 'PTC': 9, 'QM9': 10}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
NUM_CLIQUE= {'syn':6,'COLLAB': 6, 'IMDBBINARY':17, 'IMDBMULTI': 21,  'MUTAG':62,  'NCI1': 2,  'NCI109':2, 'PROTEINS':4, 'PTC':2,   'DD': 4}

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(json_file):
    config =get_config_from_json(json_file)
    config.num_classes = NUM_CLASSES[config.dataset_name]
    if config.dataset_name == 'QM9' and config.target_param is not False:
        config.num_classes = 1
    config.num_features = NUM_FEATURES[config.dataset_name]
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join("../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.parent_dir, "checkpoint/")

    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.learning_rate = LEARNING_RATES[config.dataset_name]
        config.decay_rate = DECAY_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    return config

if __name__ == '__main__':
    config = process_config('../configs/example.json')
    print(config.values())
