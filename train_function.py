import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from LSTMNN import *

def train(model, train_loader, val_loader, lr = 0.001, epochs = 100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, dt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.permute(1, 0, 2)
            dt = dt.permute(1, 0, 2)
            optimizer.zero_grad()
            lambda_end, lambda_integrate = model(x, dt)

            loss += -torch.sum((lambda_end - lambda_integrate))
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")
            val_loss = validate(model, val_loader)
            print(f"Validation Loss: {val_loss / len(val_loader)}")

def validate(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, dt in val_loader:
            x = x.permute(1, 0, 2)
            dt = dt.permute(1, 0, 2)
            lambda_end, lambda_integrate = model(x, dt)

            loss = -torch.sum((lambda_end - lambda_integrate))
            total_loss += loss.item()

    return total_loss
