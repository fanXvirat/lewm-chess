#!/usr/bin/env python3
"""
diagnostics.py — World-model probes for LeWM-Chess.

Three sanity checks that separate a *world model* from a curve-fitter:

  1. Action sensitivity   — predict z_{t+1} given the true move, a shuffled
                            legal move, pure noise, and a no-change null.
                            A causal model must do far better with the truth.
  2. Imagination drift    — closed-loop rollouts: feed the predictor its own
                            outputs and measure latent MSE vs. reality.
  3. Embedding health     — dead dimensions, effective rank, and pairwise
                            correlation of the 256-d latent space.

Usage:
    python diagnostics.py --ckpt outputs/lewm_chess_best.pt
"""
from __future__ import annotations

import argparse
import random

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train as T


def load(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = T.Config()
    for k, v in ckpt["cfg"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = T.ChessLeWM(cfg).to(T.DEVICE).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}) on {T.DEVICE}")
    return model, cfg


# ------------------------------------------------------------------ probe 1
@torch.no_grad()
def action_sensitivity(model, cfg, batches: int = 50):
    vl = DataLoader(T.CachedChessDS(cfg, "val"), batch_size=cfg.batch_size,
                    shuffle=False, num_workers=4)
    H = cfg.history_size
    errs = {"true": [], "shuffled": [], "random": [], "null": []}
    zs = []
    for bi, (frames, moves, result, progress) in enumerate(tqdm(vl, total=batches,
                                                                desc="action probe")):
        if bi >= batches:
            break
        frames = T.frames_to_gpu(frames)
        moves = moves.to(T.DEVICE)
        progress = progress.to(T.DEVICE)
        emb = model.encode(frames)
        zs.append(emb.float().cpu().numpy())
        tgt = emb[:, H]

        a_true = model.action_enc(moves, progress)[:, :H]
        perm = torch.randperm(emb.shape[0], device=emb.device)
        a_shuf = model.action_enc(moves[perm], progress[perm])[:, :H]
        a_rand = torch.randn_like(a_true)

        for name, a in (("true", a_true), ("shuffled", a_shuf), ("random", a_rand)):
            p = model.predict(emb[:, :H], a)[:, -1]
            errs[name].append(((p - tgt) ** 2).mean(-1).cpu().numpy())
        errs["null"].append(((emb[:, H - 1] - tgt) ** 2).mean(-1).cpu().numpy())

    mse = {k: float(np.concatenate(v).mean()) for k, v in errs.items()}
    print("\n--- Latent Prediction Error (MSE) ---")
    print(f"1. No-Change Baseline: {mse['null']:.5f}  (assume pieces never move)")
    print(f"2. True Action:        {mse['true']:.5f}  (real move played)")
    print(f"3. Shuffled Action:    {mse['shuffled']:.5f}  (wrong legal move)")
    print(f"4. Random Noise:       {mse['random']:.5f}  (complete garbage)")
    gap = mse["shuffled"] - mse["true"]
    print(f"\nCausal planning gap: +{gap:.5f}")
    print("PASS: model relies on actions to predict the future."
          if gap > 0 else "FAIL: model ignores action conditioning.")

    labels = ["True Move", "Shuffled Move", "Random Noise", "No-Change Null"]
    vals = [mse["true"], mse["shuffled"], mse["random"], mse["null"]]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, vals,
                   color=["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"],
                   edgecolor="black")
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * .02,
                 f"{b.get_height():.4f}", ha="center", fontweight="bold")
    plt.title("LeWM Action Sensitivity: Does the model understand moves?")
    plt.ylabel("Prediction Error (MSE)")
    plt.ylim(0, max(vals) * 1.2)
    plt.grid(axis="y", ls="--", alpha=.6)
    plt.tight_layout()
    plt.savefig("assets/action_sensitivity.png", dpi=150)
    print("saved assets/action_sensitivity.png")
    return np.concatenate(zs, axis=0)


# ------------------------------------------------------------------ probe 2
@torch.no_grad()
def imagination_drift(model, steps: int = 5):
    board = chess.Board()
    z = model.encode_board(board).unsqueeze(0)
    cur, drift, b = z, [0.0], board.copy()
    for _ in range(steps):
        legal = list(b.legal_moves)
        if not legal:
            break
        move = legal[0]
        idx = torch.tensor([[T.move_to_idx(move)]], device=T.DEVICE)
        a = model.action_enc(idx, torch.tensor([[0.0]], device=T.DEVICE))
        cur = model.predict(cur, a)
        b.push(move)
        real = model.encode_board(b).unsqueeze(0)
        drift.append(float(((cur - real) ** 2).mean()))

    print("\nImagination drift (closed-loop latent MSE):")
    for i, d in enumerate(drift):
        print(f"  Step {i}: MSE {d:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(drift)), drift, "o-", label="Imagination Error")
    plt.title("Model Reality Stability (Lower is Better)")
    plt.xlabel("Imagination Step")
    plt.ylabel("Latent MSE")
    plt.grid(alpha=.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/imagination_drift.png", dpi=150)
    print("saved assets/imagination_drift.png")


# ------------------------------------------------------------------ probe 3
def embedding_health(z: np.ndarray):
    z_flat = z.reshape(-1, z.shape[-1])
    std = z_flat.std(axis=0)
    dead = int((std < 1e-3).sum())
    _, S, _ = np.linalg.svd(z_flat - z_flat.mean(0), full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    eff_rank = int((var > 0.01).sum())
    corr = np.corrcoef(z_flat, rowvar=False)

    print("\n--- Embedding Health Report ---")
    print(f"Total Latent Dimensions: {z.shape[-1]}")
    print(f"Dead Dimensions (Std < 0.001): {dead}")
    print(f"Effective Rank (Dimensionality Usage): {eff_rank}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(std, bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Latent Stdevs")
    plt.xlabel("Std Dev"); plt.ylabel("Count of Dimensions")
    plt.subplot(1, 2, 2)
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Latent Dimension Correlation")
    plt.tight_layout()
    plt.savefig("assets/latent_health.png", dpi=150)
    print("saved assets/latent_health.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/lewm_chess_best.pt")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    model, cfg = load(args.ckpt)
    z = action_sensitivity(model, cfg, args.batches)
    imagination_drift(model, args.steps)
    embedding_health(z)


if __name__ == "__main__":
    main()
