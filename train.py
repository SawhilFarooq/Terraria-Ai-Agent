import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# ── LOAD ALL RECORDINGS ──
print("Loading recordings...")
all_frames = []
all_labels = []

files = [f for f in os.listdir('.') if f.startswith('recording_') and f.endswith('.pkl')]
files.sort()

if not files:
    print("No recording files found! Run record.py first.")
    exit()

for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loading {file}... ({len(data)} frames)")

    for entry in data:
        frame = cv2.cvtColor(entry['frame'], cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        all_frames.append(frame)

        # Handle both old recordings (keys dict) and new recordings (action int)
        if 'action' in entry:
            all_labels.append(entry['action'])
        else:
            # Convert old recording format to new action labels
            keys = entry['keys']
            clicking = entry['clicking']
            if 'h' in keys:
                label = 4
            elif 'Key.space' in keys:
                label = 3
            elif 'a' in keys:
                label = 1
            elif 'd' in keys:
                label = 2
            else:
                label = 0
            all_labels.append(label)

print(f"\nTotal frames: {len(all_frames)}")

# Show action distribution
action_names = ['shoot', 'move left', 'move right', 'jump', 'heal', 'fly', 'dash left', 'dash right']
print("Label distribution:")
for i, name in enumerate(action_names):
    count = all_labels.count(i)
    print(f"  {name}: {count} ({count/len(all_labels):.0%})")

# ── DATASET ──
class TerrariaDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = torch.tensor(np.array(frames)).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]

dataset = TerrariaDataset(all_frames, all_labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ── MODEL — 8 actions now ──
class TerrariaAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 33 * 60, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 8)  # 8 actions
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = TerrariaAgent()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining on {len(dataset)} frames with 8 actions...")

# ── TRAINING ──
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels in loader:
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.0%}")

torch.save(model.state_dict(), 'terraria_model.pth')
print("\nModel saved to terraria_model.pth!")
print("Training complete!")