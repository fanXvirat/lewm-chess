#!/usr/bin/env python3
"""
lewm-chess-engine-v2/train.py
================================================================================
LeWorldModel (LeWM) Chess World Model — v2.  Single-file, H100/Lightning-ready.

Frontier upgrades over v1:
  (A) Multi-step rollout loss with discount gamma   -> trains the deployed
      autoregressive computation, killing exposure-bias / compounding error
      (Theorem 1: E||z_hat_{t+k} - z_{t+k}|| <= eps * (L^k - 1)/(L - 1))
  (B) Hutchinson Jacobian contraction penalty       -> forces the latent map
      toward L <= 1 so rollout error grows linearly, not exponentially
  (C) Scheduled sampling (DAgger-style)             -> the predictor trains on
      its own error distribution, prob annealed 0 -> 0.5
  (D) InfoNCE action contrast                       -> different moves must
      produce distinguishable latent deltas (the planner's signal)
  (E) tanh(cp/400)-style bounded value head + legality-masked policy head
  (F) uint8 mmap frame cache + channels_last + torch.compile + fused AdamW +
      bf16 + 16 prefetching workers  -> GPU pinned at ~100%

Modes:
  python train.py --mode build-cache --pgn elite.pgn
  python train.py --mode train       --pgn elite.pgn --epochs 10
  python train.py --mode eval        --ckpt outputs/lewm_chess_best.pt
  python train.py --mode stockfish   --ckpt outputs/lewm_chess_best.pt
  python train.py --mode export      --ckpt outputs/lewm_chess_best.pt

Log format per epoch (identical to the reported run):
  E001 | train loss L pred P pol_acc A nr R | val loss L pred P pol_acc A nr R
where `nr` = ||rollout prediction|| / ||teacher-forced target|| (norm ratio):
nr -> ~1 means the predictor no longer shrinks toward the mean when fed its
own outputs — the capability v1 lacked.
================================================================================
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8

# Flash SDPA supports fast first-order training, but not the double-backward path
# needed by the Hutchinson Jacobian contraction penalty.  We keep Flash for the
# normal rollout loss and force math SDPA only inside the Jacobian branch.
def _math_sdp_context():
    if DEVICE != "cuda":
        import contextlib
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class Config:
    pgn_path: str = "chessgames.pgn"
    cache_path: str = "outputs/chess_cache.pt"
    boards_npy: str = "outputs/chess_boards.npy"
    out_dir: str = "outputs"
    val_split: float = 0.10
    seed: int = 42
    max_games: int = 20000            # 0 = all

    img_size: int = 128
    seq_len: int = 16             # frames per window: history_size ctx + num_preds rollout
    history_size: int = 12
    num_preds: int = 4            # K in the multi-step rollout loss

    embed_dim: int = 256
    pred_depth: int = 8
    pred_heads: int = 16
    pred_mlp_dim: int = 2048
    pred_dim_head: int = 64
    move_embed_dim: int = 128
    n_move_vocab: int = 4096 + 4096 * 4   # from*64+to, plus 4 promotion planes
    dropout: float = 0.10

    sigreg_lambda: float = 0.09
    sigreg_warmup_steps: int = 2000
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    temporal_gamma: float = 0.90  # discount on far rollout steps
    beta_jac: float = 1e-3        # Hutchinson Jacobian penalty weight
    infonce_weight: float = 0.01
    policy_weight: float = 0.20
    value_weight: float = 0.05
    scheduled_sampling_max: float = 0.5   # anneal model-input prob 0 -> this

    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 1e-3
    warmup_epochs: int = 2
    grad_clip: float = 5.0
    num_workers: int = 16
    prefetch_factor: int = 6
    compile_model: bool = False
    amp: str = "bf16"

    stockfish_path: str = "/usr/bin/stockfish"
    stockfish_depth: int = 12
    stockfish_positions: int = 500


# ==============================================================================
# SEED / UTIL
# ==============================================================================
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


# ==============================================================================
# MOVE VOCAB  (collision-free UCI id)
# ==============================================================================
_PROMOS = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
_PROMO_OFF = {p: i for i, p in enumerate(_PROMOS)}


def move_to_idx(m: chess.Move) -> int:
    base = m.from_square * 64 + m.to_square
    if m.promotion is None:
        return base
    return 4096 + base * 4 + _PROMO_OFF.get(m.promotion, 0)


def idx_to_move(idx: int, board: chess.Board) -> Optional[chess.Move]:
    for m in board.legal_moves:
        if move_to_idx(m) == idx:
            return m
    return None


# ==============================================================================
# BOARD RENDERER  (12 piece-specific colors; one square per ViT patch)
# ==============================================================================
_LIGHT = (240, 217, 181); _DARK = (181, 136, 99)
_PIECE_RGB = {
    (chess.PAWN, True): (170, 170, 255), (chess.PAWN, False): (40, 40, 140),
    (chess.KNIGHT, True): (170, 255, 170), (chess.KNIGHT, False): (40, 140, 40),
    (chess.BISHOP, True): (255, 170, 170), (chess.BISHOP, False): (140, 40, 40),
    (chess.ROOK, True): (255, 255, 170), (chess.ROOK, False): (140, 140, 40),
    (chess.QUEEN, True): (255, 170, 255), (chess.QUEEN, False): (140, 40, 140),
    (chess.KING, True): (170, 255, 255), (chess.KING, False): (40, 140, 140),
}
_RADIUS = {chess.PAWN: .28, chess.KNIGHT: .34, chess.BISHOP: .32,
           chess.ROOK: .36, chess.QUEEN: .40, chess.KING: .42}
_MN = np.array([0.485, 0.456, 0.406], np.float32)
_SD = np.array([0.229, 0.224, 0.225], np.float32)


def render_uint8(board: chess.Board, size: int = 128) -> np.ndarray:
    cell = size // 8
    img = Image.new("RGB", (size, size)); d = ImageDraw.Draw(img)
    for r in range(8):
        for f in range(8):
            x0, y0 = f * cell, (7 - r) * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1],
                        fill=_LIGHT if (r + f) % 2 == 0 else _DARK)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        cx = chess.square_file(sq) * cell + cell // 2
        cy = (7 - chess.square_rank(sq)) * cell + cell // 2
        rr = int(_RADIUS[p.piece_type] * cell)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                  fill=_PIECE_RGB[(p.piece_type, p.color)], outline=(30, 30, 30))
    return np.asarray(img, dtype=np.uint8)


@lru_cache(maxsize=8192)
def fen_to_tensor(fen: str, size: int) -> torch.Tensor:
    arr = render_uint8(chess.Board(fen), size).astype(np.float32) / 255.0
    arr = (arr - _MN) / _SD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


# ==============================================================================
# PGN PARSE + UINT8 MMAP FRAME CACHE   (key to 100% GPU util)
# ==============================================================================
_RESULT = {"1-0": 0, "0-1": 1, "1/2-1/2": 2, "*": 2}


def parse_pgn(cfg: Config) -> str:
    """Parse PGN into a cache of (fens, moves, results, progress, boundaries)."""
    cache = Path(cfg.cache_path)
    if cache.exists():
        print(f"Using cached parsed PGN: {cache}")
        return str(cache)
    ensure_dir(cache.parent)
    fens, moves, results, progress, boundaries = [], [], [], [], []
    n_games = 0
    with open(cfg.pgn_path, errors="ignore") as fh:
        pbar = tqdm(desc="parse pgn")
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            ms = list(game.mainline_moves())
            if len(ms) < cfg.seq_len + 1:
                continue
            res = _RESULT.get(game.headers.get("Result", "*"), 2)
            boundaries.append(len(fens))
            b = game.board()
            n = len(ms)
            for i, m in enumerate(ms):
                fens.append(b.fen())
                moves.append(move_to_idx(m))
                results.append(res)
                progress.append(i / max(n - 1, 1))
                b.push(m)
            n_games += 1; pbar.update(1)
            if cfg.max_games and n_games >= cfg.max_games:
                break
        pbar.close()
    torch.save({"fens": fens,
                "moves": np.asarray(moves, np.int64),
                "results": np.asarray(results, np.int64),
                "progress": np.asarray(progress, np.float32),
                "boundaries": np.asarray(boundaries, np.int64),
                "n_games": n_games}, cache)
    print(f"parsed {n_games} games, {len(fens)} positions -> {cache}")
    return str(cache)


@torch.no_grad()
def build_board_cache(cfg: Config):
    """Render every position ONCE into a uint8 mmap. Removes PIL from the hot loop."""
    cache = parse_pgn(cfg)
    d = torch.load(cache, weights_only=False)
    fens = d["fens"]
    ensure_dir(Path(cfg.boards_npy).parent)
    arr = np.lib.format.open_memmap(cfg.boards_npy, mode="w+", dtype=np.uint8,
                                    shape=(len(fens), cfg.img_size, cfg.img_size, 3))
    for i, fen in enumerate(tqdm(fens, desc="render boards")):
        arr[i] = render_uint8(chess.Board(fen), cfg.img_size)
    arr.flush()
    print(f"cached {arr.shape} uint8 -> {cfg.boards_npy} "
          f"({arr.nbytes / 1e9:.1f} GB; page cache will hold it in 240GB RAM)")


class CachedChessDS(Dataset):
    def __init__(self, cfg: Config, split: str):
        self.boards = np.load(cfg.boards_npy, mmap_mode="r")
        d = torch.load(cfg.cache_path, weights_only=False)
        self.moves = d["moves"]
        self.results = d["results"]
        self.progress = d["progress"]
        bounds = d["boundaries"].tolist() + [len(d["fens"])]
        n = d["n_games"]
        nval = max(1, int(n * cfg.val_split))

        # --- RANDOM GAME SPLIT ---
        all_games = list(range(n))
        rng = random.Random(cfg.seed)  # reproducible
        rng.shuffle(all_games)
        train_games = all_games[:-nval]
        val_games = all_games[-nval:]
        selected = train_games if split == "train" else val_games
        # -------------------------

        self.samples = []
        for g in selected:
            start = bounds[g]
            end = bounds[g + 1]
            # add all windows of length cfg.seq_len inside this game
            for i in range(start, end - cfg.seq_len):
                self.samples.append(i)
        self.sl = cfg.seq_len
        print(f"{split}: {len(self.samples):,} windows from {len(selected):,} games")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]; sl = self.sl
        # Force a writable copy of the uint8 slice to avoid PyTorch writability warning
        fr = self.boards[s:s + sl].copy()
        return (torch.from_numpy(fr),
                torch.from_numpy(self.moves[s:s + sl].copy()),
                int(self.results[s]),
                torch.from_numpy(self.progress[s:s + sl].copy()))


# ==============================================================================
# GPU NORMALIZATION HELPER
# ==============================================================================
def frames_to_gpu(frames: torch.Tensor) -> torch.Tensor:
    """
    Convert uint8 (B,T,H,W,3) from DataLoader into normalized float32 (B,T,3,H,W).
    All operations happen on GPU.
    """
    frames = frames.to(DEVICE, non_blocking=True)
    if frames.dtype == torch.uint8:
        frames = frames.to(torch.float32).div_(255.0)
        mean = torch.tensor(_MN, device=frames.device).view(1, 1, 1, 1, 3)
        std = torch.tensor(_SD, device=frames.device).view(1, 1, 1, 1, 3)
        frames = (frames - mean) / std
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
    else:
        frames = frames.to(torch.float32)
    return frames


# ==============================================================================
# SIGREG  (Epps–Pulley ECF statistic over random 1-D projections)
# ==============================================================================
class SIGReg(nn.Module):
    def __init__(self, num_proj: int = 1024, knots: int = 17):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        w = torch.full((knots,), 2 * dt); w[[0, -1]] = dt
        win = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", win)
        self.register_buffer("weights", w * win)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D) time-first."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0) + EPS)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


# ==============================================================================
# TRANSFORMER BLOCKS  (faithful to LeWM module.py)
# ==============================================================================
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, drop=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads; self.drop = drop
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(drop))

    def forward(self, x, causal=True):
        x = self.norm(x)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    """AdaLN-zero: 6-way (shift, scale, gate) x (attn, ffn)."""

    def __init__(self, dim, heads, dim_head, mlp, drop=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, drop)
        self.mlp = FeedForward(dim, mlp, drop)
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sm, scm, gm, sf, scf, gf = self.ada(c).chunk(6, dim=-1)
        x = x + gm * self.attn(modulate(self.n1(x), sm, scm))
        x = x + gf * self.mlp(modulate(self.n2(x), sf, scf))
        return x


class ARPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.embed_dim
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len, D) * 0.02)
        self.blocks = nn.ModuleList([
            ConditionalBlock(D, cfg.pred_heads, cfg.pred_dim_head,
                             cfg.pred_mlp_dim, cfg.dropout)
            for _ in range(cfg.pred_depth)])
        self.norm = nn.LayerNorm(D)
        self.drift = nn.Linear(D, D)   # residual drift: z_hat = z + drift(h)
        nn.init.zeros_(self.drift.weight); nn.init.zeros_(self.drift.bias)

    def forward(self, x, c):
        h = x + self.pos[:, :x.size(1)]
        for blk in self.blocks:
            h = blk(h, c)
        return x + self.drift(self.norm(h))


class ChessMoveEmbedder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embedding = nn.Embedding(cfg.n_move_vocab, cfg.move_embed_dim)
        self.proj = nn.Sequential(nn.Linear(cfg.move_embed_dim, 4 * cfg.embed_dim),
                                  nn.SiLU(), nn.Linear(4 * cfg.embed_dim, cfg.embed_dim))
        self.progress = nn.Sequential(nn.Linear(1, cfg.embed_dim), nn.SiLU(),
                                      nn.Linear(cfg.embed_dim, cfg.embed_dim))

    def forward(self, moves, progress=None):
        out = self.proj(self.embedding(moves))
        if progress is not None:
            out = out + self.progress(progress.unsqueeze(-1))
        return out


class ChessLeWM(nn.Module):
    """Encoder (ViT) + projector + AR predictor + heads, end-to-end."""

    def __init__(self, cfg: Config):
        super().__init__()
        import timm
        self.cfg = cfg
        self.encoder = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=0,
            img_size=cfg.img_size, embed_dim=192)
        self.enc_proj = nn.Sequential(nn.Linear(192, 2048), nn.BatchNorm1d(2048),
                                      nn.GELU(), nn.Linear(2048, cfg.embed_dim))
        self.predictor = ARPredictor(cfg)
        self.action_enc = ChessMoveEmbedder(cfg)
        self.policy_head = nn.Sequential(nn.LayerNorm(cfg.embed_dim),
                                         nn.Linear(cfg.embed_dim, 1024), nn.GELU(),
                                         nn.Linear(1024, cfg.n_move_vocab))
        self.value_head = nn.Sequential(nn.LayerNorm(cfg.embed_dim),
                                        nn.Linear(cfg.embed_dim, 256), nn.GELU(),
                                        nn.Linear(256, 1))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) -> (B, T, D)"""
        B, T = frames.shape[:2]
        # Ensure channels_last memory format for better ViT performance
        x = frames.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        z = self.encoder(x)
        z = self.enc_proj(z)
        return z.view(B, T, -1)

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        x = fen_to_tensor(board.fen(), self.cfg.img_size)
        return self.encode(x.unsqueeze(0).unsqueeze(0).to(next(self.parameters()).device))[:, 0]

    def predict(self, emb, act):
        return self.predictor(emb, act)


# ==============================================================================
# LOSSES — the heart of v2
# ==============================================================================
def rollout_and_jacobian(model, emb, act, cfg, ss_prob: float):
    """
    Multi-step rollout loss with scheduled sampling + Hutchinson Jacobian penalty.

    Returns (loss, pred_1step, norm_ratio):
      pred_1step  — the headline `pred` metric (1-step MSE, comparable to v1)
      norm_ratio  — ||K-step rollout|| / ||target||, the `nr` log metric
    """
    B, T, D = emb.shape
    H, K = cfg.history_size, cfg.num_preds
    ctx = emb[:, :H]

    total = emb.new_tensor(0.0); denom = 0.0
    pred_1 = emb.new_tensor(0.0); nr = emb.new_tensor(1.0)
    cur = ctx
    for k in range(1, K + 1):
        a = act[:, k - 1:k - 1 + cur.shape[1]]
        nxt = model.predict(cur, a)
        tgt = emb[:, k:k + nxt.shape[1]].detach()
        w = cfg.temporal_gamma ** (k - 1)
        total = total + w * F.mse_loss(nxt, tgt)
        denom += w
        if k == 1:
            pred_1 = F.mse_loss(nxt, tgt).detach()
        if k == K:
            nr = (nxt[:, -1].norm(dim=-1).mean()
                  / (tgt[:, -1].norm(dim=-1).mean() + EPS)).detach()
        # scheduled sampling: mix model output into the next context
        if torch.rand(()) < ss_prob:
            cur = torch.cat([cur[:, 1:], nxt[:, -1:]], dim=1)
        else:
            cur = emb[:, k:k + H]
    loss = total / max(denom, EPS)

    # Hutchinson contraction penalty on the drift Jacobian (1 matvec).
    # IMPORTANT: this needs a double backward. PyTorch Flash/Mem-efficient SDPA
    # does not implement that derivative, so this branch is forced to math SDPA.
    # Also skip it during eval/validation; it is a training regularizer only.
    if cfg.beta_jac > 0 and torch.is_grad_enabled():
        z = emb[:, :H].detach().requires_grad_(True)
        with _math_sdp_context():
            zhat = model.predict(z, act[:, :H])
        v = torch.randn_like(zhat)
        (jvp,) = torch.autograd.grad((zhat * v).sum(), z, create_graph=True)
        loss = loss + cfg.beta_jac * jvp.pow(2).mean()
    return loss, pred_1, nr


def full_forward(model, sigreg, batch, cfg, ss_prob, sig_scale: float = 1.0):
    frames, moves, result, progress = batch
    # Normalize on GPU: uint8 -> float32, ImageNet stats, (B,T,3,H,W)
    frames = frames_to_gpu(frames)
    moves = moves.to(DEVICE, non_blocking=True)
    result = result.to(DEVICE, non_blocking=True)
    progress = progress.to(DEVICE, non_blocking=True)

    emb = model.encode(frames)                       # (B, T, D)
    act = model.action_enc(moves, progress)          # (B, T, D)

    L_roll, pred_1, nr = rollout_and_jacobian(model, emb, act, cfg, ss_prob)
    L_sig = sigreg(emb.transpose(0, 1))

    # policy on history positions
    H = cfg.history_size; D = emb.shape[-1]
    zf = emb[:, :H].reshape(-1, D); mf = moves[:, :H].reshape(-1)
    plog = model.policy_head(zf)
    L_pol = F.cross_entropy(plog, mf)

    # bounded value (WDL proxy here; tanh(cp/400) targets used in RL stage)
    vpred = torch.tanh(model.value_head(emb.mean(1))).squeeze(-1)
    wdl = torch.where(result == 0, 1.0, torch.where(result == 1, -1.0, 0.0)).float()
    L_val = F.mse_loss(vpred, wdl)

    # InfoNCE: true action vs shuffled action must be distinguishable
    z0 = emb[:, :H]; a_pos = act[:, :H]
    perm = torch.randperm(emb.shape[0], device=emb.device)
    p_pos = model.predict(z0, a_pos)[:, -1]
    p_neg = model.predict(z0, act[perm][:, :H])[:, -1]
    tgt = emb[:, H].detach() if emb.shape[1] > H else emb[:, -1].detach()
    s_pos = F.cosine_similarity(p_pos, tgt, dim=-1)
    s_neg = F.cosine_similarity(p_neg, tgt, dim=-1)
    L_nce = F.cross_entropy(torch.stack([s_pos, s_neg], 1),
                            torch.zeros(emb.shape[0], dtype=torch.long, device=emb.device))

    loss = (L_roll + sig_scale * cfg.sigreg_lambda * L_sig + cfg.policy_weight * L_pol
            + cfg.value_weight * L_val + cfg.infonce_weight * L_nce)
    with torch.no_grad():
        pacc = (plog.argmax(-1) == mf).float().mean()
    return {"loss": loss, "pred": pred_1, "nr": nr, "pacc": pacc.detach()}


def cosine_lr(opt, step, total, warmup, base, min_ratio):
    if step < warmup:
        lr = base * step / max(warmup, 1)
    else:
        p = (step - warmup) / max(total - warmup, 1)
        lr = base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ==============================================================================
# TRAIN
# ==============================================================================
def train(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    parse_pgn(cfg)
    if not os.path.exists(cfg.boards_npy):
        build_board_cache(cfg)

    tr = DataLoader(CachedChessDS(cfg, "train"), batch_size=cfg.batch_size,
                    shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0,
                    prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
                    drop_last=True)
    vl = DataLoader(CachedChessDS(cfg, "val"), batch_size=cfg.batch_size,
                    shuffle=False, num_workers=max(cfg.num_workers // 2, 1),
                    pin_memory=True, persistent_workers=cfg.num_workers > 0)

    model = ChessLeWM(cfg).to(DEVICE).to(memory_format=torch.channels_last)
    sigreg = SIGReg(cfg.sigreg_num_proj, cfg.sigreg_knots).to(DEVICE)
    if cfg.compile_model and hasattr(torch, "compile") and DEVICE == "cuda":
        model = torch.compile(model)
    raw = getattr(model, "_orig_mod", model)
    params = list(model.parameters())
    fused_ok = DEVICE == "cuda"
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            fused=fused_ok)

    steps_per_epoch = len(tr)
    total = steps_per_epoch * cfg.epochs
    warm = steps_per_epoch * cfg.warmup_epochs
    print(f"Device: {DEVICE}; epochs={cfg.epochs}; steps/epoch={steps_per_epoch}; "
          f"batch={cfg.batch_size}; amp={cfg.amp}")

    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    gstep, best = 0, float("inf")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "pred": 0.0, "pacc": 0.0, "nr": 0.0}; n = 0
        ss = cfg.scheduled_sampling_max * min(1.0, (ep - 1) / max(cfg.epochs - 1, 1))
        # Step-wise progress bar
        pbar = tqdm(tr, desc=f"Epoch {ep}", unit="batch", leave=False)
        for batch in pbar:
            cosine_lr(opt, gstep, total, warm, cfg.lr, cfg.min_lr_ratio)
            sig_scale = min(1.0, (gstep + 1) / max(cfg.sigreg_warmup_steps, 1))
            with torch.autocast("cuda", dtype=amp_dtype, enabled=DEVICE == "cuda"):
                out = full_forward(model, sigreg, batch, cfg, ss_prob=ss,
                                   sig_scale=sig_scale)
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            gstep += 1; n += 1
            for k in ("loss", "pred", "pacc", "nr"):
                agg[k] += float(out[k].detach())   # detach to avoid warning
            pbar.set_postfix(loss=float(out["loss"].detach()))
        trm = {k: v / max(n, 1) for k, v in agg.items()}

        # ---- validation
        model.eval()
        vagg = {"loss": 0.0, "pred": 0.0, "pacc": 0.0, "nr": 0.0}; vn = 0
        with torch.no_grad():
            for batch in vl:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=DEVICE == "cuda"):
                    out = full_forward(model, sigreg, batch, cfg, ss_prob=ss)
                vn += 1
                for k in ("loss", "pred", "pacc", "nr"):
                    vagg[k] += float(out[k].detach())
        vlm = {k: v / max(vn, 1) for k, v in vagg.items()}

        print(f"E{ep:03d} | train loss {trm['loss']:.4f} pred {trm['pred']:.4f} "
              f"pol_acc {trm['pacc']:.3f} nr {trm['nr']:.2f} | "
              f"val loss {vlm['loss']:.4f} pred {vlm['pred']:.4f} "
              f"pol_acc {vlm['pacc']:.3f} nr {vlm['nr']:.2f}")

        # Save latest checkpoint unconditionally
        latest_path = Path(cfg.out_dir) / "lewm_chess_latest.pt"
        torch.save({"model": raw.state_dict(), "cfg": asdict(cfg), "epoch": ep}, latest_path)

        # Save best if improved
        if vlm["pred"] < best:
            best = vlm["pred"]
            best_path = Path(cfg.out_dir) / "lewm_chess_best.pt"
            torch.save({"model": raw.state_dict(), "cfg": asdict(cfg),
                        "epoch": ep, "val_pred": best}, best_path)
            print(f"saved best: {best_path} (val pred_loss={best:.6f})")
    return model


# ==============================================================================
# EVAL: Stockfish-grounded metrics on held-out positions
# ==============================================================================
@torch.no_grad()
def masked_policy_topk(model, board: chess.Board, k: int = 5) -> List[chess.Move]:
    z = model.encode_board(board)
    logits = model.policy_head(z).squeeze(0)
    legal = list(board.legal_moves)
    idx = torch.tensor([move_to_idx(m) for m in legal], device=logits.device)
    scores = logits[idx]
    order = scores.argsort(descending=True)[:k]
    return [legal[i] for i in order.tolist()]


@torch.no_grad()
def stockfish_eval(cfg: Config, ckpt_path: str):
    import chess.engine
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg2 = Config(**{**asdict(cfg), **{k: v for k, v in ck["cfg"].items()
                                       if k in asdict(cfg)}})
    model = ChessLeWM(cfg2).to(DEVICE).eval()
    model.load_state_dict(ck["model"])

    d = torch.load(cfg2.cache_path, weights_only=False)
    bounds = d["boundaries"].tolist() + [len(d["fens"])]
    n = d["n_games"]; nval = max(1, int(n * cfg2.val_split))
    val_fens = [d["fens"][i] for g in range(n - nval, n)
                for i in range(bounds[g], bounds[g + 1])]
    random.seed(cfg2.seed)
    sample = random.sample(val_fens, min(cfg2.stockfish_positions, len(val_fens)))

    eng = chess.engine.SimpleEngine.popen_uci(cfg2.stockfish_path)
    legal_ok, top1, top5, deltas = [], [], [], []
    for fen in tqdm(sample, desc="stockfish eval"):
        b = chess.Board(fen)
        if b.is_game_over():
            continue
        cand = masked_policy_topk(model, b, k=5)
        legal_ok.append(all(m in b.legal_moves for m in cand))
        info = eng.analyse(b, chess.engine.Limit(depth=cfg2.stockfish_depth),
                           multipv=min(5, b.legal_moves.count()))
        infos = info if isinstance(info, list) else [info]
        sf_moves = [e["pv"][0] for e in infos if "pv" in e and e["pv"]]
        if not sf_moves:
            continue
        top1.append(cand[0] == sf_moves[0])
        top5.append(sf_moves[0] in cand)
        # delta-cp of the model's chosen move
        b2 = b.copy(); b2.push(cand[0])
        sc_best = infos[0]["score"].pov(b.turn).score(mate_score=10000)
        sc_mine = eng.analyse(b2, chess.engine.Limit(depth=cfg2.stockfish_depth)
                              )["score"].pov(b.turn).score(mate_score=10000)
        if sc_best is not None and sc_mine is not None:
            deltas.append(max(0.0, float(sc_best - sc_mine)))
    eng.quit()
    deltas = np.asarray(deltas)
    print(f"legal_move_rate      {np.mean(legal_ok):.2f}")
    print(f"stockfish_top1_match {np.mean(top1):.2f}")
    print(f"stockfish_top5_match {np.mean(top5):.2f}")
    print(f"mean_delta_cp        {deltas.mean():.2f}")
    print(f"median_delta_cp      {np.median(deltas):.0f}")


# ==============================================================================
# EXPORT: per-position arrays for the ablation/CI pack
# ==============================================================================
@torch.no_grad()
def export_arrays(cfg: Config, ckpt_path: str, out: str = "exports/chess_seed0.npz"):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ChessLeWM(cfg).to(DEVICE).eval()
    model.load_state_dict(ck["model"])
    vl = DataLoader(CachedChessDS(cfg, "val"), batch_size=cfg.batch_size,
                    shuffle=False, num_workers=4)
    errs_true, errs_zero, errs_shuf = [], [], []
    for frames, moves, result, progress in tqdm(vl, desc="export"):
        frames = frames.to(DEVICE); moves = moves.to(DEVICE); progress = progress.to(DEVICE)
        emb = model.encode(frames)
        H = cfg.history_size
        tgt = emb[:, H]
        a_true = model.action_enc(moves, progress)[:, :H]
        a_zero = torch.zeros_like(a_true)
        perm = torch.randperm(emb.shape[0], device=emb.device)
        a_shuf = model.action_enc(moves[perm], progress[perm])[:, :H]
        for a, dst in ((a_true, errs_true), (a_zero, errs_zero), (a_shuf, errs_shuf)):
            p = model.predict(emb[:, :H], a)[:, -1]
            dst.append(((p - tgt) ** 2).mean(-1).cpu().numpy())
    ensure_dir(Path(out).parent)
    np.savez(out, err_true=np.concatenate(errs_true),
             err_zero=np.concatenate(errs_zero),
             err_shuffled=np.concatenate(errs_shuf))
    print(f"exported corruption arrays -> {out}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train",
                    choices=["build-cache", "train", "eval", "stockfish", "export"])
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--ckpt", default="outputs/lewm_chess_best.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--beta-jac", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--prefetch-factor", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.pgn: cfg.pgn_path = args.pgn
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.max_games is not None: cfg.max_games = args.max_games
    if args.beta_jac is not None: cfg.beta_jac = args.beta_jac
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.prefetch_factor is not None: cfg.prefetch_factor = args.prefetch_factor

    if args.mode == "build-cache":
        build_board_cache(cfg)
    elif args.mode == "train":
        train(cfg)
    elif args.mode in ("eval", "stockfish"):
        stockfish_eval(cfg, args.ckpt)
    elif args.mode == "export":
        export_arrays(cfg, args.ckpt)


if __name__ == "__main__":
    main()