import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
from collections import Counter
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import LSTMModel

# =====================
# DATASET
# =====================
dataset = load_dataset("mteb/turkish_movie_sentiment", split="train")

def tokenize(text):
    return text.lower().split()

# =====================
# VOCAB
# =====================
counter = Counter()
for x in dataset:
    counter.update(tokenize(x["text"]))

vocab = {w: i+1 for i, (w, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0

def encode(text, max_len=30):
    ids = [vocab.get(w, 0) for w in tokenize(text)]
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))

X = torch.tensor([encode(x["text"]) for x in dataset])
y = torch.tensor([x["label"] for x in dataset]).float().unsqueeze(1)

# =====================
# TRAIN / VAL / TEST SPLIT
# =====================
full_ds = TensorDataset(X, y)

train_size = int(0.7 * len(full_ds))
val_size   = int(0.15 * len(full_ds))
test_size  = len(full_ds) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_ds, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)
test_loader  = DataLoader(test_ds, batch_size=64)

# =====================
# MODEL
# =====================
model = LSTMModel(len(vocab))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =====================
# TRAIN + VALIDATION
# =====================
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    # VALIDATION
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model(xb)
            val_preds += (p > 0.5).int().tolist()
            val_trues += yb.int().tolist()

    acc = accuracy_score(val_trues, val_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        val_trues, val_preds, average="binary"
    )

    print(f"\nEpoch {epoch+1}")
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Validation Precision: {prec:.4f}")
    print(f"Validation Recall   : {rec:.4f}")
    print(f"Validation F1-score : {f1:.4f}")

# =====================
# TEST (FINAL)
# =====================
model.eval()
test_preds, test_trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        p = model(xb)
        test_preds += (p > 0.5).int().tolist()
        test_trues += yb.int().tolist()

test_acc = accuracy_score(test_trues, test_preds)
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
    test_trues, test_preds, average="binary"
)

print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"F1-score : {test_f1:.4f}")

# =====================
# CONFUSION MATRIX (TEST - GRAPH)
# =====================
cm = confusion_matrix(test_trues, test_preds)

plt.figure(figsize=(4,4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Negatif", "Pozitif"])
plt.yticks([0,1], ["Negatif", "Pozitif"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()

# =====================
# SAVE
# =====================
torch.save(model.state_dict(), "sentiment_lstm.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\nModel ve vocab kaydedildi.")
