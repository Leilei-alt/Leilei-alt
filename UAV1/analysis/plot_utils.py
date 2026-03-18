# analysis/plot_utils.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_line_with_errorbars(
    x_vals,
    y_means,
    y_stds,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 5))
    plt.errorbar(x_vals, y_means, yerr=y_stds, marker="o", capsize=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multi_line(
    x_vals,
    series: list[tuple[str, list[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 5))
    for label, y_vals in series:
        plt.plot(x_vals, y_vals, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
