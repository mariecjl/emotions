from torch.utils.data import DataLoader
from dataset import FER2013Dataset
from cnnmodel import EmotionCNN
import torch.nn as nn
import torch
from tqdm import tqdm

#datasets & split and dataloaders
train_ds = FER2013Dataset("fer2013.csv", split="Training")
val_ds   = FER2013Dataset("fer2013.csv", split="PublicTest")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)

#config -- device, model, loss criterion and optimizer used
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#number of epochs to train for
epochs = 20

for epoch in range(epochs):
    model.train()
    train_loss = 0

    #progress bar display
    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs} [Train]",
        leave=False
    )

    #training loop
    for x, y in train_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        #forward pass
        preds = model(x)
        loss = criterion(preds, y)
        #backprop
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #progress bar updates
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0

    #progress bar for validation loop
    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1}/{epochs} [Val]",
        leave=False
    )
    #validation loop
    with torch.no_grad():
        for x, y in val_bar:
            x, y = x.to(device), y.to(device)
            #forward pass
            preds = model(x)
            #correct predictions
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
            #update progress bar
            val_bar.set_postfix(
                acc=f"{100 * correct / total:.2f}%"
            )
    #progress update
    acc = correct / total * 100
    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train Loss {train_loss/len(train_loader):.4f} "
        f"Val Acc {acc:.2f}%"
    )

torch.save(model.state_dict(), "emotion_cnn_fer2013.pth")
print(" emotion_cnn_fer2013.pth done")
