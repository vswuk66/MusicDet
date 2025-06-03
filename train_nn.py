import os
import glob
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import extract_features
from model.nn_model import GenreClassifier
import json

# Mapping genres to labels
genre_map = {
    0: 'Blues', 1: 'Classical', 2: 'Country',
    3: 'Disco', 4: 'Hiphop', 5: 'Jazz', 6: 'Metal',
    7: 'Pop', 8: 'Reggae', 9: 'Rock'
}
label_map = {v: k for k, v in genre_map.items()}

data_dir = 'data'
X, y = [], []

print("Extracting features...")
for genre, label in label_map.items():
    genre_dir = os.path.join(data_dir, genre)
    if not os.path.isdir(genre_dir):
        print(f"Warning: directory {genre_dir} not found, skipping...")
        continue
    for ext in ('*.wav', '*.mp3'):
        for file_path in glob.glob(os.path.join(genre_dir, ext)):
            try:
                feats = extract_features(file_path)
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"Error on {file_path}: {e}")

X = np.array(X)
y = np.array(y)
print(f"Extracted {len(X)} samples, each of dimension {X.shape[1]}.")

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs('model', exist_ok=True)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_ds = MusicDataset(X_train, y_train)
val_ds = MusicDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate model
dim_in = X_train.shape[1]
model = GenreClassifier(input_dim=dim_in, hidden_dims=[128, 64], num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 30
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct = 0.0, 0
    for feats_batch, labels_batch in train_loader:
        feats_batch, labels_batch = feats_batch.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(feats_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels_batch).sum().item()
    train_loss = total_loss / len(train_ds)
    train_acc = correct / len(train_ds)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for feats_batch, labels_batch in val_loader:
            feats_batch, labels_batch = feats_batch.to(device), labels_batch.to(device)
            outputs = model(feats_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item() * feats_batch.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels_batch).sum().item()
    val_loss /= len(val_ds)
    val_acc = val_correct / len(val_ds)

    print(f"Epoch {epoch}/{epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), 'model/nn_model.pth')
print("Training complete. Model saved to model/nn_model.pth")

# Save genre mapping used during training
with open('genre_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(genre_map, f, ensure_ascii=False, indent=2)
