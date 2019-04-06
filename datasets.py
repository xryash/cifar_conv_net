import pickle

import numpy as np


def _load_cifar10_batch(cifar10_dataset_folder_path, file):
    with open(cifar10_dataset_folder_path + str(file), mode='rb') as file:
        batch = pickle.load(file, encoding='latin')
    features = batch['data']
    labels = batch['labels']

    return features, labels


def _convert_to_matrix(x):
    """Convert labels to matrix"""
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded


def _normalize(x):
    """Normalize features"""
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def _split_dataset(features, labels, ratio):
    """Split dataset with the ratio"""
    arr = np.arange(len(labels))
    np.random.shuffle(arr)
    validation_ratio = int(ratio * len(features))
    test_features = features[arr[0:validation_ratio]]
    test_labels = labels[arr[0:validation_ratio]]

    train_features = features[arr[validation_ratio:features.size]]
    train_labels = labels[arr[validation_ratio:features.size]]

    return train_features, test_features, train_labels, test_labels


def _save(features, labels, new_folder_path, filename):
    """Save datasets to file"""
    pickle.dump((features, labels), open(str(new_folder_path) + str(filename), 'wb'))


def _preprocess(features, labels):
    """Normalize and preprocess data"""
    features = features.reshape((len(features), 3, 32, 32)).transpose(0, 2, 3, 1)
    features = _normalize(features)
    labels = _convert_to_matrix(labels)
    return features, labels


def preprocess_datasets(cifar10_dataset_folder_path, new_folder_path, files, ratio, preprocessed_test_file_name,
                        preprocessed_train_file_name):
    """Preprocess and save datasets to files"""

    # init buffer arrays for test datasets
    test_features = []
    test_labels = []

    # init buffer arrays for train datasets
    train_features = []
    train_labels = []

    for file in files:
        # load data
        features, labels = _load_cifar10_batch(cifar10_dataset_folder_path, file)

        # normalize data
        features, labels = _preprocess(features, labels)

        # split train and test datasets
        train_features_batch, test_features_batch, train_labels_batch, test_labels_batch = _split_dataset(features,
                                                                                                          labels, ratio)

        # save test datasets in the buffer
        test_features.extend(test_features_batch)
        test_labels.extend(test_labels_batch)

        # save train datasets in the buffer
        train_features.extend(train_features_batch)
        train_labels.extend(train_labels_batch)

    # convert to numpy arrays
    test_features = np.array(test_features)
    train_features = np.array(train_features)
    test_labels = np.array(test_labels)
    train_labels = np.array(train_labels)

    # save test datasets to file
    _save(test_features, test_labels, new_folder_path, preprocessed_test_file_name)

    # save train datasets to file
    _save(train_features, train_labels, new_folder_path, preprocessed_train_file_name)

