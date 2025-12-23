import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import random

from torchvision.models import ResNet18_Weights

classes = ['Esox_lucius', 'Cyprinus_carpio', 'Carcharodon_carcharias',
           'Silurus_glanis', 'Delphinapterus_leucas']

class FishDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.classes = list(set(self.df['label']))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label_str = self.df.iloc[idx]['label']
        label = self.class_to_idx.get(label_str)

        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df = pd.read_csv('tree_ds.csv')
random.shuffle(df.values.tolist())
train_size = int(len(df) * 0.9)
train_df = df[:train_size]
val_df = df[train_size:]

train_dataset = FishDataset(train_df, transform=transform)
val_dataset = FishDataset(val_df, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000136, weight_decay=1e-7 * 6.4)


def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        running_loss = 0.0
        corrects = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects / total
        print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_acc = val_corrects / val_total
        print(f'Validation Accuracy: {val_acc:.4f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_fish_classifier.pth')


train_model(model, criterion, optimizer, num_epochs=9)