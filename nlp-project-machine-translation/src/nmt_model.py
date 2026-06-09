"""
NMT model
----------
This model uses:
 - Simple subword splitting (no external packages)
 - Lowercasing + punctuation spacing
 - Truncated sentences (max 40 tokens)
 - min_freq=1 vocab (no rare-word removal)
 - LSTM encoder-decoder
 - Dot-product attention
 - Mini-batch training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import math
import re

# ================================================
# TEXT PREPROCESSING
# ================================================

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_subwords(tokens, chunk=5):
    out = []
    for w in tokens:
        if len(w) <= chunk:
            out.append(w)
        else:
            for i in range(0, len(w), chunk):
                out.append(w[i:i+chunk])
    return out

# ================================================
# VOCAB
# ================================================

class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.pad = "<pad>"
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.unk = "<unk>"
        self.word2idx = {}
        self.idx2word = []

    def build(self, sentences):
        counter = Counter()
        for s in sentences:
            counter.update(s)

        vocab = [self.pad, self.sos, self.eos, self.unk] + \
                [w for w, f in counter.items() if f >= self.min_freq]

        self.idx2word = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}

    def __len__(self):
        return len(self.idx2word)

    def encode(self, tokens):
        return [self.word2idx.get(t, self.word2idx[self.unk]) for t in tokens]

    def decode(self, ids):
        return [self.idx2word[i] for i in ids]

# ================================================
# ENCODER
# ================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        return out, h, c

# ================================================
# ATTENTION
# ================================================

class Attention(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.scale = 1 / math.sqrt(hid)

    def forward(self, h, enc_out):
        h = h.permute(1, 2, 0)   # (batch, H, 1)
        scores = torch.bmm(enc_out, h).squeeze(2)
        attn = torch.softmax(scores * self.scale, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out)
        return ctx

# ================================================
# DECODER
# ================================================

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.attn = Attention(hid)
        self.lstm = nn.LSTM(emb + hid, hid, batch_first=True)
        self.fc = nn.Linear(hid*2 + emb, vocab_size)

    def forward(self, tok, h, c, enc_out):
        tok = tok.unsqueeze(1)
        emb = self.embedding(tok)
        ctx = self.attn(h, enc_out)
        lstm_in = torch.cat((emb, ctx), 2)
        out, (h, c) = self.lstm(lstm_in, (h, c))
        pred = self.fc(torch.cat((out, ctx, emb), 2)).squeeze(1)
        return pred, h, c

# ================================================
# SEQ2SEQ
# ================================================

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, src, trg, tf=0.7):   # teacher forcing increased to 0.7
        batch, T = trg.shape
        vocab = self.dec.embedding.num_embeddings
        out = torch.zeros(batch, T, vocab)

        enc_out, h, c = self.enc(src)
        tok = trg[:, 0]

        for t in range(1, T):
            pred, h, c = self.dec(tok, h, c, enc_out)
            out[:, t] = pred
            teacher = (torch.rand(1).item() < tf)
            tok = trg[:, t] if teacher else pred.argmax(1)

        return out

# ================================================
# DATA LOADING
# ================================================

def load_data(path, limit=None):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            text = simple_preprocess(line)
            toks = text.split()
            toks = split_into_subwords(toks)
            toks = toks[:40]        # truncate long sentences
            lines.append(toks)
    return lines

def pad_batch(seqs, pad_idx):
    m = max(len(s) for s in seqs)
    return [s + [pad_idx]*(m - len(s)) for s in seqs]

def batches(X, Y, size=32):
    for i in range(0, len(X), size):
        yield X[i:i+size], Y[i:i+size]

# ================================================
# TRAINING
# ================================================

def train_nmt():
    print("Loading data...")
    en = load_data("data/train_10000.en")
    sv = load_data("data/train_10000.sv")

    print("Building vocab...")
    en_vocab = Vocab(min_freq=1); en_vocab.build(en)
    sv_vocab = Vocab(min_freq=1); sv_vocab.build(sv)

    print("Encoding...")
    X = [en_vocab.encode(s) for s in en]
    Y = [[sv_vocab.sos] + s + [sv_vocab.eos] for s in sv]
    Y = [sv_vocab.encode(s) for s in Y]

    print("Building model...")
    enc = Encoder(len(en_vocab))
    dec = Decoder(len(sv_vocab))
    model = Seq2Seq(enc, dec)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=sv_vocab.word2idx[sv_vocab.pad])

    print("Training...")
    EPOCHS = 12

    for ep in range(EPOCHS):
        total = 0
        for xb, yb in batches(X, Y, 32):
            xb = pad_batch(xb, en_vocab.word2idx[en_vocab.pad])
            yb = pad_batch(yb, sv_vocab.word2idx[sv_vocab.pad])

            xb = torch.tensor(xb).long()
            yb = torch.tensor(yb).long()

            optimizer.zero_grad()
            out = model(xb, yb, tf=0.7)

            vocab = out.shape[-1]
            loss = loss_fn(out[:, 1:].reshape(-1, vocab),
                           yb[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()

        print(f"Epoch {ep+1}/{EPOCHS}  loss={total:.3f}")

    torch.save(model.state_dict(), "models/nmt_bpe.pt")
    print("Saved → models/nmt_bpe.pt")

    return model, en_vocab, sv_vocab

# ================================================
# TRANSLATION
# ================================================

def translate_sentence(model, line, en_vocab, sv_vocab):
    text = simple_preprocess(line)
    toks = text.split()
    toks = split_into_subwords(toks)[:40]
    src = torch.tensor([en_vocab.encode(toks)])

    enc_out, h, c = model.enc(src)

    tok = torch.tensor([sv_vocab.word2idx["<sos>"]])
    out_words = []

    for _ in range(40):
        pred, h, c = model.dec(tok, h, c, enc_out)
        top = pred.argmax(1).item()

        if top == sv_vocab.word2idx["<eos>"]:
            break

        w = sv_vocab.idx2word[top]
        if w not in ("<pad>", "<sos>", "<eos>", "<unk>"):
            out_words.append(w)

        tok = torch.tensor([top])

    return out_words

# ================================================
# MAIN
# ================================================

if __name__ == "__main__":
    print("STARTING IMPROVED NMT TRAINING...")
    model, en_vocab, sv_vocab = train_nmt()

    print("Translating test set...")
    with open("data/test.en") as fin, open("eval/nmt_output.sv", "w") as fout:
        for line in fin:
            sv = translate_sentence(model, line, en_vocab, sv_vocab)
            fout.write(" ".join(sv) + "\n")

    print("Done → eval/nmt_output.sv")
