import csv
import math
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#------------------------
# Tokenization + Features
#------------------------


# Basic Unicode-aware word tokenizer; matches letters/digits/underscore.
WORD_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text):
    return [w.lower() for w in WORD_RE.findall(text)]



def fourth_root_length(tokens):
    return len(tokens) ** 0.25

def ovix(tokens):

    """
    Compute an OVIX-like lexical diversity measure with numeric safeguards.

    Parameters =>tokens : list[str]

    Returns => OVIX value; 0.0 if n<=1 or if logs/denominator are unsafe.

    OVIX is defined (in one common form) based on:
        n : total tokens
        k : number of unique tokens (types)
    This code with clamps to avoid NaN/inf, which is practical for noisy text.
    """
    
    n = len(tokens)
    if n <= 1:
        return 0.0
    k = len(set(tokens))

    try:
        ln_n = math.log(n)
        ln_k = math.log(k) if k > 0 else 0
        inner = 2 - (ln_k / ln_n)
        if inner <= 0:
            return 0.0
        val = ln_n / math.log(inner)
        # clamp extreme values
        if math.isinf(val) or math.isnan(val) or val > 50:
            return 50.0
        return val
    except:
        return 0.0

        

def feature_vector(text):
    """
    Build a dense 2D feature vector for a single essay.

    Parameters
    ----------
    text : str

    Returns
    -------
    np.ndarray shape (2,)
        [ length^0.25 , OVIX ]
    """
    
    toks = tokenize(text)
    return np.array([fourth_root_length(toks), ovix(toks)], dtype=float)
    


# -----------------------
# Load TSV manually
# -----------------------

def load_asap(path):
    data = {1: [], 2: []}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        for row in reader:
            if len(row) < 7:
                continue

            try:
                set_id = int(row[1])
            except:
                continue

            if set_id not in (1, 2):
                continue

            text = row[2].strip()
            try:
                score = float(row[6])
            except:
                continue

            data[set_id].append((text, score))

    return data


# -----------------------
# Helpers
# -----------------------

def zscore(y):
    mu = np.mean(y)
    sd = np.std(y)
    if sd == 0:
        sd = 1
    return (y - mu)/sd


def normalize_features(X):
    """Normalize each feature column → avoids exploding gradients."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - mu) / sd, mu, sd


# -----------------------
# PyTorch model
# -----------------------

def train_torch(Xtr, ytr, Xdv, ydv, lr=0.005, epochs=200, batch=32):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xdv_t = torch.tensor(Xdv, dtype=torch.float32)
    ydv_t = torch.tensor(ydv, dtype=torch.float32)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xb, yb in dl:
            pred = model(xb).flatten()
            loss = ((pred - yb)**2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        tr_pred = model(Xtr_t).flatten().numpy()
        dv_pred = model(Xdv_t).flatten().numpy()

    w = model.weight.detach().numpy().flatten()
    b = float(model.bias.detach().numpy())
    tr_mse = np.mean((tr_pred - ytr)**2)
    dv_mse = np.mean((dv_pred - ydv)**2)

    return w, b, tr_mse, dv_mse


# -----------------------
# MAIN
# -----------------------

def main():

    data_file = "/home/dsv/robe/lis050/lab3/data/asap-train.tsv"
    data = load_asap(data_file)

    train = data[1]
    valid = data[2]

    # Build features
    Xtr = np.vstack([feature_vector(t) for t,_ in train])
    ytr = np.array([s for _,s in train], float)

    Xdv = np.vstack([feature_vector(t) for t,_ in valid])
    ydv = np.array([s for _,s in valid], float)

    # Standardize scores
    ytr_s = zscore(ytr)
    ydv_s = zscore(ydv)

    # Normalize feature matrix (critical!)
    Xtr_n, muX, sdX = normalize_features(Xtr)
    Xdv_n = (Xdv - muX) / sdX

    # sklearn baseline
    print("\n=== sklearn LinearRegression ===")
    sk = LinearRegression()
    sk.fit(Xtr_n, ytr_s)

    print("Weights:", sk.coef_)
    print("Bias:", sk.intercept_)
    print("Train MSE:", mean_squared_error(ytr_s, sk.predict(Xtr_n)))
    print("Dev   MSE:", mean_squared_error(ydv_s, sk.predict(Xdv_n)))

    # PyTorch
    print("\n=== PyTorch SGD Model ===")
    try:
        w, b, trm, dvm = train_torch(
            Xtr_n, ytr_s, Xdv_n, ydv_s,
            lr=0.005,       # LOWER LR = stable
            epochs=200,
            batch=32
        )

        print("Weights:", w)
        print("Bias:", b)
        print("Train MSE:", trm)
        print("Dev   MSE:", dvm)

    except ImportError:
        print("PyTorch not available.")


if __name__ == "__main__":
    main()
