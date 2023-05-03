import logging
import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label

def split_train_test(dataset, idxs, batch_size):
    # split train, and test
    # idxs_train = idxs
    train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
    return train


def _data_transforms_emnist():

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ])


    valid_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform

def dirichlet_cifar_noniid(degree_noniid, dataset, num_users):
    
    train_labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    
    label_distribution = np.random.dirichlet([degree_noniid]*num_users, num_classes)
    
    # print(label_distribution)
    # print(sum(label_distribution), sum(np.transpose(label_distribution)), sum(sum(label_distribution)))
    
    class_idcs = [np.argwhere(train_labels==y).flatten() for y in range(num_classes)]
    
    dict_users = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            dict_users[i] += [idcs]

    # print(dict_users, np.shape(dict_users))
    
    dict_users = [set(np.concatenate(idcs)) for idcs in dict_users]
    
    return dict_users

def load_partition_data_emnist( data_dir, partition_method, partition_alpha, client_number, batch_size, logger):
    transform_train, transform_test = _data_transforms_emnist()
    dataset_train = EMNIST(data_dir, train=True, download=False,
                        transform=transform_train, split = 'letters')
    dataset_test = EMNIST(data_dir, train=False, download=False,
                        transform=transform_test, split = 'letters')


    dict_train = dirichlet_cifar_noniid(partition_alpha, dataset_train, client_number) # non-iid
    dict_test = dirichlet_cifar_noniid(partition_alpha, dataset_test, client_number) # non-iid\

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        local_data = dict_train[client_idx] 
        data_local_num_dict[client_idx] = len(local_data)
        train_data_local_dict[client_idx] = split_train_test(dataset_train, list(dict_train[client_idx]),batch_size)
        # test_data_local_dict[client_idx]  = DataLoader(DatasetSplit(dataset_test ,  dict_test[client_idx]), batch_size=len(dict_test[client_idx]), shuffle=False)
        test_data_local_dict[client_idx]  = split_train_test(dataset_test , list(dict_test[client_idx]), batch_size)
        # split_train_test(dataset_test , list(dict_test[client_idx]),batch_size)

    return data_local_num_dict, train_data_local_dict, test_data_local_dict




   