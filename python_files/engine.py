"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


#Train Step for training the specifed model
def train_stp(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              optimizer: torch.optim.Optimizer, 
              device: torch.device) -> Tuple[float, float]:
    
    """
    Trains a Pytorch Model. Algorithm based on documentation supplied by instructor.    
    
    """

    model.train()

    train_loss = 0
    train_acc = 0

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_predictions = model(X)
        
        loss = loss_fn(y_predictions, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()


        y_predictions_class = torch.argmax(torch.softmax(y_predictions, dim=1), dim=1)
        train_acc += (y_predictions_class == y).sum().item()/len(y_predictions)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return  train_loss, train_acc

#Train Step for training the specifed model
def test_stp(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module, 
             device: torch.device) -> Tuple[float, float]:

    model.eval()
    test_loss = 0 
    test_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            X,y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

#Full train function that trains and tests data against a specifed model and prints/returns the results 
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          epochs: int, 
          device: torch.device) -> Dict[str, list]:

    results = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}

    for epoch in tqdm(range(epochs), "Epochs: "):
        train_loss, train_acc = train_stp(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)

        test_loss, test_acc = test_stp(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} \n"
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")        
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


