"""
Filename: utils.py

Assignment: Movie Poster Genre Classification Multi-Model project
Class: Basics of AI

Authors: Kristopher Kodweis & Kumar Satyam

Contains functions for creating the iterable data for each of the models and data types.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import List
import torchvision
from torch import nn

import os

import zipfile

import requests

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from pathlib import Path
from torchvision import transforms


### REUSABLE TRANSFORMS DEFINED HERE ###
class baseV0:
    
    def __init__(self, height: int = 128, width: int = 128):
        self.data_transform = transforms.Compose([
            transforms.v2.ToDtype(torch.uint8, scale=True),
            transforms.Resize(size=(height, width), antialias=True),
            transforms.Grayscale(),
            transforms.v2.ToImage(),
            transforms.v2.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Adds a slight random perspective to image datasets to help simulate reworld picture taking 
class perspectiveV0:    

    def __init__(self, height: int = 64, width: int = 64):
        self.data_transform = transforms.Compose([
            transforms.v2.ToDtype(torch.uint8, scale=True),
            transforms.Resize(size=(height, width), antialias=True),
            transforms.RandomPerspective(distortion_scale=0.1, p=.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.v2.ToImage(),
            transforms.v2.ToDtype(torch.float32, scale=True),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])


# Function to save a PyTorch model
def save_model(model: torch.nn.Module, target_dir: str, model_name: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    print(f"[INFO] Saving model to: {model_save_path}")

    torch.save(obj=model.state_dict(), f=model_save_path)


### THESE ARE TO HELP VISUALIZE THE DATA AND COMPUTATIONS BEING PERFORMED
def create_confusion_matrix(y_pred_tensor, test_data, class_names):

    conf_mat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    conf_mat_tensor = conf_mat(preds=y_pred_tensor, target=test_data.targets)
    
    fig, ax = plot_confusion_matrix(conf_mat=conf_mat_tensor.numpy(), class_names=class_names, figsize=(10, 7))

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    model.to("cpu") 
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min ()-0.1, X[:, 0].max() + 0.1
    y_min, y_max = y[:, 1].min ()-0.1, y[:, 1].max() + 0.1

    xx , yy = np.meshgrid(np.linspace(x_min, x_max, 101, np.linspace(y_min, y_max, 101)))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode:
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contour(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=.07)
    plt.scatter(X[:, 0], X[:, 1], c = y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(prop={"size":14})

def print_train_time(start, end, device=None):
    total_time = end-start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def pred_and_plot_img(model: torch.nn.Module, image_path: str, class_names:List[str] = None, transform=None, device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    target_image = target_image / 255.0

    if transform:
        target_image = transform(target_image)

    model.to(device)

    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)

        target_image_pred = model(target_image.to(device))

    
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.imshow(target_image.squeeze().permute(1, 2, 0))

    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]}  |  Prob: {target_image_pred_probs.max().cpu():.3f}"

    else:
         title = f"Pred: {target_image_pred_label}  |  Prob: {target_image_pred_probs.max().cpu():.3f}"

    plt.title(title)
    plt.axis(False)

def plot_loss_curves(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy ")
    plt.xlabel("Epochs")
    plt.legend()