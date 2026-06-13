#!/usr/bin/env python3
"""
play.py — Pit LeWM-Chess against Stockfish (or a human) over UCI.

The model's masked policy head proposes moves; Stockfish replies at a chosen
depth. The full game is exported as PGN.

Usage:
    python play.py --ckpt outputs/lewm_chess_best.pt --color white --depth 18
    python play.py --ckpt outputs/lewm_chess_best.pt --color black --top-k 3
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import chess
import chess.engine
import chess.pgn
import torch

import train as T


def load_model(ckpt_path: str) -> tuple[T.ChessLeWM, T.Config]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = T.Config()
    for k, v in ckpt["cfg"].items():
        if hasattr(cfg, k) and k != "stockfish_path":
            setattr(cfg, k, v)
    model = T.ChessLeWM(cfg).to(T.DEVICE).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}) on {T.DEVICE}")
    return model, cfg


def lewm_move(model: T.ChessLeWM, board: chess.Board, top_k: int) -> chess.Move:
    candidates = T.masked_policy_topk(model, board, k=top_k)
    if not candidates:
        return random.choice(list(board.legal_moves))
    return candidates[0] if top_k == 1 else random.choice(candidates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/lewm_chess_best.pt")
    ap.add_argument("--color", choices=["white", "black"], default="white",
                    help="side LeWM plays")
    ap.add_argument("--depth", type=int, default=12, help="Stockfish depth")
    ap.add_argument("--top-k", type=int, default=1,
                    help="sample uniformly among the model's top-k moves")
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH",
                                                          "/usr/bin/stockfish"))
    ap.add_argument("--out", default="lewm_vs_stockfish.pgn")
    args = ap.parse_args()

    model, _ = load_model(args.ckpt)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    print(f"Stockfish loaded from {args.stockfish} (depth={args.depth})")

    lewm_is_white = args.color == "white"
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"LeWM vs Stockfish (top-k={args.top_k}, depth={args.depth})"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "LeWM-Chess" if lewm_is_white else f"Stockfish d{args.depth}"
    game.headers["Black"] = f"Stockfish d{args.depth}" if lewm_is_white else "LeWM-Chess"
    node = game

    while not board.is_game_over() and board.fullmove_number <= args.max_moves:
        lewm_to_move = board.turn == chess.WHITE if lewm_is_white else board.turn == chess.BLACK
        if lewm_to_move:
            move = lewm_move(model, board, args.top_k)
            who = "LeWM"
        else:
            move = engine.play(board, chess.engine.Limit(depth=args.depth)).move
            who = "Stockfish"
        print(f"{who} ({'White' if board.turn else 'Black'}): {board.san(move)}")
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result()
    engine.quit()
    print(f"\nGame over: {board.result()}")
    with open(args.out, "w") as fh:
        fh.write(str(game) + "\n")
    print(f"PGN saved to {args.out}")


if __name__ == "__main__":
    main()
