"""
Filename: utils.py

Assignment: Movie Poster Genre Classification Multi-Model project
Class: Basics of AI

Authors: Kristopher Kodweis & Kumar Satyam

Contains functions for creating the iterable data for each of the models and data types.
"""
import torch

from pathlib import Path
from torchvision import transforms



class baseV0:
    
    def __init__(self, height: int = 128, width: int = 128):
        self.data_transform = transforms.Compose([
            transforms.v2.ToDtype(torch.uint8, scale=True),
            transforms.Resize(size=(height, width), antialias=True),
            transforms.RandomCrop(size=64, padding=2),
            transforms.v2.ToImage(),
            transforms.v2.ToDtype(torch.float32, scale=True),
            transforms.Normalize([.5, .5, .5], [.1, .1, .1])
        ])

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

