#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mia_attack.py

End-to-end Membership Inference Attack (MIA) script for TEM fastText models.
.ft.txt splits live under `data/`
.Privatized .bin models live under `data/imdb/`

Usage:
  python mia_attack.py \
    --data-dir data \
    --model-pattern ft_imdb_priv_e{eps}.bin \
    --epsilons 2.0 2.5 3.0 3.5 4.0 \
    --shadow-epochs 20 --shadow-dim 300 --shadow-ngrams 1 \
    --attack-epochs 20 --batch-size 64 --attack-lr 1e-3
"""
import os
import argparse
import numpy as np
import fasttext
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------------------------------
# Attack model: a 2-layer MLP (64 hidden units, ReLU)
# -----------------------------------------------------------------------------
class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------------------------------------------------------
# Load fastText-format file and extract top-2 probabilities
# -----------------------------------------------------------------------------
def gather_fasttext_probs(model, ft_path):
    texts = []
    with open(ft_path, 'r', encoding='utf-8') as f:
        for line in f:
            _, text = line.strip().split(' ', 1)
            texts.append(text)
    labels, probs = model.predict(texts, k=2)
    arr = np.zeros((len(texts), 2), dtype=float)
    for i, p_list in enumerate(probs):
        arr[i, :] = p_list[:2]
    return arr

# -----------------------------------------------------------------------------
# Train the attack MLP on shadow model outputs
# -----------------------------------------------------------------------------
def train_attack_model(shadow, te1_ft, te2_ft, device,
                       epochs, batch_size, lr):
    out_feats = gather_fasttext_probs(shadow, te1_ft)
    in_feats  = gather_fasttext_probs(shadow, te2_ft)
    X = np.vstack([in_feats, out_feats])
    y = np.hstack([np.ones(len(in_feats)), np.zeros(len(out_feats))]).astype(int)

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    attack = AttackModel(X.shape[1]).to(device)
    opt = optim.Adam(attack.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    attack.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = attack(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return attack

# -----------------------------------------------------------------------------
# Evaluate attack on target model outputs
# -----------------------------------------------------------------------------
def evaluate_attack(attack, target, tr1_ft, tr2_ft, device):
    in_feats  = gather_fasttext_probs(target, tr1_ft)
    out_feats = gather_fasttext_probs(target, tr2_ft)
    X = np.vstack([in_feats, out_feats])
    y = np.hstack([np.ones(len(in_feats)), np.zeros(len(out_feats))]).astype(int)

    X_t = torch.from_numpy(X).float().to(device)
    attack.eval()
    with torch.no_grad():
        logits = attack(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return roc_auc_score(y, probs)

# -----------------------------------------------------------------------------
# Main: parse args, train shadow, attack, evaluate
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',       type=str,   required=True,
                   help='Folder with TE1.tsv.ft.txt, TE2.tsv.ft.txt, TR1.tsv.ft.txt, TR2.tsv.ft.txt')
    p.add_argument('--model-pattern',  type=str,   default='ft_imdb_priv_e{eps}.bin',
                   help='Pattern for model filenames under data/imdb; use {eps}')
    p.add_argument('--epsilons',       type=float, nargs='+', required=True,
                   help='List of ε values, e.g. 2.0 2.5 3.0')
    p.add_argument('--shadow-epochs',  type=int,   default=20,
                   help='Epochs to train shadow model on TE2')
    p.add_argument('--shadow-dim',     type=int,   default=300,
                   help='Embedding dimension for shadow model')
    p.add_argument('--shadow-ngrams',  type=int,   default=1,
                   help='wordNgrams for shadow model (must be 1)')
    p.add_argument('--attack-epochs',  type=int,   default=20,
                   help='Epochs to train attack MLP')
    p.add_argument('--batch-size',     type=int,   default=64)
    p.add_argument('--attack-lr',      type=float, default=1e-3,
                   help='Learning rate for attack MLP')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    te1_ft = os.path.join(args.data_dir, 'TE1.tsv.ft.txt')
    te2_ft = os.path.join(args.data_dir, 'TE2.tsv.ft.txt')
    tr1_ft = os.path.join(args.data_dir, 'TR1.tsv.ft.txt')
    tr2_ft = os.path.join(args.data_dir, 'TR2.tsv.ft.txt')

    print(f"Training shadow model: epochs={args.shadow_epochs}, dim={args.shadow_dim}, ngrams={args.shadow_ngrams}")
    shadow = fasttext.train_supervised(
        input=te2_ft,
        epoch=args.shadow_epochs,
        dim=args.shadow_dim,
        wordNgrams=args.shadow_ngrams
    )

    print("Training attack model...")
    attack = train_attack_model(
        shadow, te1_ft, te2_ft, device,
        epochs=args.attack_epochs,
        batch_size=args.batch_size,
        lr=args.attack_lr
    )

    print("ε\tAUROC")
    for eps in args.epsilons:
        bin_name = args.model_pattern.format(eps=eps)
        model_path = os.path.join(args.data_dir, 'imdb', bin_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Target model not found: {model_path}")
        target = fasttext.load_model(model_path)
        auc = evaluate_attack(attack, target, tr1_ft, tr2_ft, device)
        print(f"{eps}\t{auc:.4f}")
