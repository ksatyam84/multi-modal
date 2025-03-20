"""
Filename: data_setup.py

Assignment: Movie Poster Genre Classification Multi-Model project
Class: Basics of AI

Authors: Kristopher Kodweis & Kumar Satyam

Contains functions for creating the iterable data for each of the models and data types.
"""

import os
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

# This setting the number of CPU cores are available to compute on
NUM_WORKERS = os.cpu_count()

# This is developing the dataloaders for training Image Classification models
def create_image_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers = NUM_WORKERS):

    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform = transform)

    classes = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers = NUM_WORKERS, pin_memory= True)

    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers = NUM_WORKERS, pin_memory= True)

    return train_dataloader, test_dataloader, classes


