"""Hot-wire path simulator.

    python simulate.py                     # runs examples/demo.svg
    python simulate.py my_shapes.svg       # or a .dxf
    python simulate.py file.svg --clearance 1.0

Controls:  space = pause / resume,  up / down = faster / slower,  enter = restart
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import shapes_io
from planner import Planner

COLORS = {"travel": "#f39c12", "cut": "#27ae60", "bridge": "#e74c3c"}
LABELS = {"travel": "travel (through free foam)", "cut": "cut (shape outline)",
          "bridge": "bridge (cut-in to a hole)"}


def densify(moves, n_target=4000):
    """Resample all moves into evenly spaced points so the pointer moves at constant speed."""
    total = sum(np.sum(np.hypot(*np.diff(np.array(p), axis=0).T)) for _, p in moves)
    step = max(total / n_target, 1e-9)
    X, Y, K = [], [], []
    for kind, pts in moves:
        pts = np.array(pts)
        X.append(pts[0, 0]); Y.append(pts[0, 1]); K.append(kind)
        for p, q in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(np.hypot(*(q - p)) / step)))
            for t in np.linspace(0, 1, n + 1)[1:]:
                X.append(p[0] + (q[0] - p[0]) * t)
                Y.append(p[1] + (q[1] - p[1]) * t)
                K.append(kind)
    return np.array(X), np.array(Y), np.array(K)


def path_lengths(moves):
    out = {k: 0.0 for k in COLORS}
    for kind, pts in moves:
        out[kind] += float(np.sum(np.hypot(*np.diff(np.array(pts), axis=0).T)))
    return out


def run(contours, clearance=None):
    planner = Planner(contours, origin=(0.0, 0.0), clearance=clearance)
    moves = planner.plan()

    n_parts = sum(1 for c in planner.contours if not c.is_hole)
    n_holes = len(planner.contours) - n_parts
    lengths = path_lengths(moves)
    print(f"{n_parts} parts, {n_holes} holes")
    for k, v in lengths.items():
        print(f"  {k:7s}: {v:9.1f}")
    print(f"  total  : {sum(lengths.values()):9.1f}")

    X, Y, K = densify(moves)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.canvas.manager.set_window_title("Hot-wire path simulator")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    for c in planner.contours:  # the shapes themselves
        x, y = c.poly.exterior.xy
        ax.fill(x, y, color="white" if c.is_hole else "#d6e4f0", zorder=1)
        ax.plot(x, y, color="#5d6d7e", lw=0.8, zorder=2)

    for kind, pts in moves:  # planned path, faint
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color="#999999", lw=0.8, ls="--", alpha=0.6, zorder=3)

    done = {}
    for kind, col in COLORS.items():  # traversed path, coloured by move type
        done[kind], = ax.plot([], [], color=col, lw=2.2, zorder=4, label=LABELS[kind])
    pointer, = ax.plot([], [], "o", color="black", mfc="red", ms=9, zorder=6)
    ax.plot(0, 0, marker="*", color="purple", ms=16, zorder=5, label="origin (0, 0)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    status = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", family="monospace",
                     fontsize=9, bbox=dict(fc="white", ec="#cccccc", alpha=0.9), zorder=7)

    pad = 0.05 * max(np.ptp(X), np.ptp(Y))
    ax.set_xlim(X.min() - pad, X.max() + pad)
    ax.set_ylim(Y.min() - pad, Y.max() + pad)

    masked = {k: (np.where(K == k, X, np.nan), np.where(K == k, Y, np.nan)) for k in COLORS}
    seglen = np.concatenate([[0], np.hypot(np.diff(X), np.diff(Y))])
    dist = np.cumsum(seglen)
    state = {"i": 0, "speed": 6, "paused": False}

    def draw(i):
        for k, line in done.items():
            line.set_data(masked[k][0][:i + 1], masked[k][1][:i + 1])
        pointer.set_data([X[i]], [Y[i]])
        status.set_text(f"pos   X {X[i]:8.2f}  Y {Y[i]:8.2f}\n"
                        f"move  {K[i]}\n"
                        f"dist  {dist[i]:8.1f} / {dist[-1]:.1f}  ({100 * i / (len(X) - 1):5.1f}%)\n"
                        f"speed x{state['speed']}" + ("   [PAUSED]" if state["paused"] else ""))

    def update(_):
        if not state["paused"]:
            state["i"] = min(state["i"] + state["speed"], len(X) - 1)
        draw(state["i"])
        return []

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "up":
            state["speed"] = min(state["speed"] * 2, 256)
        elif event.key == "down":
            state["speed"] = max(state["speed"] // 2, 1)
        elif event.key == "enter":
            state["i"] = 0

    fig.canvas.mpl_connect("key_press_event", on_key)
    ax.set_title("space: pause   up/down: speed   enter: restart", fontsize=9)
    anim = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return anim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("file", nargs="?", default=os.path.join(here, "examples", "demo.svg"))
    ap.add_argument("--clearance", type=float, default=None,
                    help="gap kept between travel moves and shapes (default: 0.5%% of drawing size)")
    args = ap.parse_args()
    run(shapes_io.load(args.file), args.clearance)


if __name__ == "__main__":
    main()
