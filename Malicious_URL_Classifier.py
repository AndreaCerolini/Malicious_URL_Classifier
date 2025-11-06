import os, pickle, numpy as np, torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import random

# ---------------- CONFIGURAZIONE ----------------

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

# Cartella di lavoro attuale

BASE_DIR = os.getcwd()

# Directory dei dati preprocessati

DATA_DIR = os.path.join(BASE_DIR, "preprocessed_data")

# Cartella per salvare i modelli e i report

REPORTS_DIR = os.path.join(BASE_DIR, "models", "reports")

MODEL_SAVE = os.path.join(BASE_DIR, "models", "best_model_embed.pth")

# Creo la cartella se non esiste

os.makedirs(REPORTS_DIR, exist_ok=True)

BATCH_SIZE = 64

EPOCHS = 100

ES_PATIENCE = 25

ES_MIN_DELTA = 0.0001

LR = 0.001

WD = 0.0001

os.makedirs(REPORTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device,"\n")

# ---------------- CARICAMENTO DATI ----------------

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

with open(os.path.join(DATA_DIR, "string_to_index.pkl"), "rb") as f:

    meta = pickle.load(f)

vocab_size = meta["vocab_size"]

PAD_IDX = meta["PAD"]

UNK = meta["UNK"]

labels = meta.get("labels", ["benign","defacement","malware","phishing"])

print(f"Dataset: X={X.shape}, y={y.shape}, vocab_size={vocab_size}","\n")

# ---------------- SPLIT ----------------

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

# ---------------- TORCH ----------------

X_train = torch.from_numpy(X_train).long()
X_val   = torch.from_numpy(X_val).long()
X_test  = torch.from_numpy(X_test).long()

y_train = torch.from_numpy(y_train).long()
y_val   = torch.from_numpy(y_val).long()
y_test  = torch.from_numpy(y_test).long()

# ---------------- STATISTICHE ----------------

def distr(Set_Name, y):

    c = Counter(y.tolist())

    tot = len(y)

    result = {}

    for key, value in sorted(c.items()):

        percentage = 100 * value / tot

        result[key] = f"{value} ({percentage:.1f}%)"

    print(Set_Name,result,"\n")

distr("Dataset: ", torch.from_numpy(y))

distr("Training Set: ", y_train)

distr("Validation Set: ", y_val)

distr("Test Set: ", y_test)

# ---------------- DATASET & SAMPLER ----------------

train_ds = TensorDataset(X_train, y_train)

val_ds   = TensorDataset(X_val, y_val)

test_ds  = TensorDataset(X_test, y_test)

num_classes = len(labels)

c = Counter(y_train.tolist())

class_counts = torch.tensor([c[i] for i in range(num_classes)], dtype=torch.float)

weights_per_class = 1.0 / class_counts

sample_weights = weights_per_class[y_train].to(torch.float)

sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights), replacement=True)
pin_mem = (device == "cuda")

train_loader = DataLoader(train_ds,
                          batch_size = BATCH_SIZE,
                          sampler = sampler,
                          num_workers = 0,
                          pin_memory = pin_mem)

val_loader = DataLoader(val_ds,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = 0,
                        pin_memory = pin_mem)

test_loader = DataLoader(test_ds,
                         batch_size = BATCH_SIZE,
                         shuffle = False,
                         num_workers = 0,
                         pin_memory = pin_mem)

# ---------------- MODELLO ----------------

class MultiClassModel(nn.Module):

    def __init__(self, vocab_size, emb_dim = 64, num_classes = 4, pad_idx = 0):

        super().__init__()

        self.pad_idx = pad_idx

        self.emb = nn.Embedding(vocab_size,
                                emb_dim,
                                padding_idx = pad_idx)

        self.layer_stack = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        E = self.emb(x)

        mask = (x != self.pad_idx).float()

        denom = mask.sum(dim = 1, keepdim = True).clamp_min(1.0)

        pooled = (E * mask.unsqueeze(-1)).sum(dim = 1) / denom

        return self.layer_stack(pooled)

model = MultiClassModel(vocab_size = vocab_size,
                        emb_dim = 128,
                        num_classes = num_classes,
                        pad_idx = PAD_IDX).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = WD)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

loss_fn = nn.CrossEntropyLoss()

# ---------------- TRAINING LOOP ----------------

best_val = float("inf")

es_wait = 0

train_losses, val_losses = [], []

def evaluate(loader):

    model.eval()

    loss_sum = 0.0

    n = 0

    y, all_preds = [], []

    with torch.inference_mode():

        for xb,yb in loader:

            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)

            loss = loss_fn(logits, yb)

            preds = logits.argmax(dim=1)

            m = yb.size(0)

            loss_sum += loss.item() * m

            n += m

            y.append(yb.cpu())

            all_preds.append(preds.cpu())

    y = torch.cat(y).numpy()

    all_preds = torch.cat(all_preds).numpy()

    return (loss_sum / n), y, all_preds

for epoch in range(EPOCHS):

    model.train()

    loss_sum = 0.0

    n = 0

    for xb, yb in train_loader:

        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)

        loss = loss_fn(logits, yb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        m = yb.size(0)

        loss_sum += loss.item() * m

        n += m

    train_loss = loss_sum / n

    train_losses.append(train_loss)

    val_loss, y_val_np, p_val_np = evaluate(val_loader)

    val_losses.append(val_loss)

    scheduler.step(val_loss)

    improved = val_loss < ( best_val - ES_MIN_DELTA )

    if improved:

        best_val = val_loss

        es_wait = 0

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "val_loss": best_val}, MODEL_SAVE)
    else:

        es_wait += 1

    lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
          f"lr={lr:.2e} | {'IMPROVED' if improved else f'no-improve({es_wait}/{ES_PATIENCE})'}")

    if epoch % 10 == 0:
        print("\nValidation report:")
        print(classification_report(y_val_np, p_val_np, target_names=labels, digits=4))

    if es_wait >= ES_PATIENCE:
        print(f"Early stopping a epoch {epoch+1}. Best val_loss={best_val:.6f}")
        break

# ---------------- TEST ----------------

test_loss, y_test, preds = evaluate(test_loader)

# Confusion matrix
cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.show()

print("Numero di esempi analizzati:", len(y_test))

print("Somma elementi matrice:", cm.sum())

# ---------------- PLOT ----------------

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")

plt.plot(val_losses, label="Val Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()