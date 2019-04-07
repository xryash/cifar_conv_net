import pickle

import matplotlib.pyplot as plt
import yaml
import os.path

from datasets import preprocess_datasets


def loss_plot(train_loss, test_loss):
    """Draw test and train loss plot"""
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


def accuracy_plot(train_accuracy, test_accuracy):
    """Draw test and train accuracy plot"""
    plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_accuracy)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


def _load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    return config


def load_hyperparams():
    """Load neural network configuration"""
    config_path = 'application.yaml'
    config = _load_config(config_path)
    hyperparams = config['hyperparams']
    epochs = hyperparams['epochs']
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    model_replica_path = hyperparams['model_replica_path']
    dropout_rate = hyperparams['dropout_rate']
    return epochs, batch_size, learning_rate, model_replica_path, dropout_rate


def _load_datasets_config():
    """Load datasets configuration"""
    config_path = 'application.yaml'
    config = _load_config(config_path)
    datasets_config = config['datasets']
    cifar10_dataset_folder_path = datasets_config['cifar10_dataset_folder_path']
    new_folder_path = datasets_config['new_folder_path']
    ratio = datasets_config['ratio']
    files = datasets_config['files']
    preprocessed_test_file_name = datasets_config['preprocessed_test_file_name']
    preprocessed_train_file_name = datasets_config['preprocessed_train_file_name']
    return cifar10_dataset_folder_path, new_folder_path, ratio, files, preprocessed_test_file_name, preprocessed_train_file_name


def load_datasets():
    """Load train and test datasets"""
    cifar10_dataset_folder_path, new_folder_path, ratio, files, preprocessed_test_file_name, preprocessed_train_file_name = _load_datasets_config()

    test_path = new_folder_path + preprocessed_test_file_name
    train_path = new_folder_path + preprocessed_train_file_name

    if not (os.path.exists(test_path) and os.path.exists(train_path)):
        preprocess_datasets(cifar10_dataset_folder_path, new_folder_path, files, ratio, preprocessed_test_file_name,
                            preprocessed_train_file_name)

    with open(test_path, mode='rb') as file:
        test_dataset = pickle.load(file)

    with open(train_path, mode='rb') as file:
        train_dataset = pickle.load(file)

    train_x, train_y = train_dataset[0], train_dataset[1]
    test_x, test_y = test_dataset[0], test_dataset[1]

    return train_x, train_y, test_x, test_y
