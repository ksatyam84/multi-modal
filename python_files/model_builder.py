"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

import torch
from torch import nn


# The TinyVGG model is a simple convolutional neural network (CNN) architecture.
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate the flattened size after the convolutional layers
        self.flattened_size = self._get_flattened_size(input_shape, hidden_units)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flattened_size, out_features=output_shape)
        )

    def _get_flattened_size(self, input_shape: int, hidden_units: int) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape, 224, 224)  # Assuming input images are 224x224
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    

