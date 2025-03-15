"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from tqdm.auto import tqdm
import data_setup, engine, model_builder, utils

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 25
LEARNING_RATE = 0.01

torch.manual_seed(42)  # Set seed for reproducibility


if __name__ == '__main__':

    train_dir = "/Users/kkodweis/Github-Repos/EAS510-BasicsAI/multi-modal/datasets/SampleV0/Train"
    test_dir = "/Users/kkodweis/Github-Repos/EAS510-BasicsAI/multi-modal/datasets/SampleV0/Test"

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    data_transform = utils.perspectiveV0()

    train_dataloader, test_dataloader, class_labels = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform.data_transform,
        batch_size=BATCH_SIZE
    )

    model = model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_labels)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=NUM_EPOCHS
    )

    utils.save_model(model=model,target_dir="models", model_name="tinyvgg_model_V2.pth")

    
    loaded_modelV2 = model_builder.TinyVGG()
    loaded_modelV2.load_state_dict(torch.load("models/tinyvgg_model_V2.pth"))

    print(f"Loaded model:\n{loaded_modelV2}")
    print(f"Model on device:\n{next(loaded_modelV2.parameters()).device}")

    # Evaluate loaded model
    loaded_modelV2.eval()
    with torch.inference_mode():
        loaded_modelV2_preds = loaded_modelV2(test_dataloader)
    print(loaded_modelV2_preds)



