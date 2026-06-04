#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = [
    "Полный спектр LR",
    "MI top-12 + LR",
    "SPA top-12 + LR",
    "ADMM direct LS",
    "ADMM top-12 + LR",
]

TASK_LABELS = {"health": "здоровые / больные", "stages": "временные стадии"}


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def fmt_seconds(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.2f} с"
    if sec < 3600:
        return f"{sec/60:.2f} мин"
    return f"{sec/3600:.2f} ч"


def load_runs(run_dirs):
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        metrics = rd / "method_metrics.csv"
        summary = rd / "run_summary.json"
        if not metrics.exists():
            print("skip, no method_metrics.csv:", rd)
            continue
        df = pd.read_csv(metrics)
        if summary.exists():
            with open(summary, "r", encoding="utf-8") as f:
                s = json.load(f)
            experiment = f"{s.get('dataset', '')} / {TASK_LABELS.get(s.get('task', ''), s.get('task', ''))}"
            df["experiment"] = experiment
            df["dataset"] = s.get("dataset", df.get("dataset", ""))
            df["task"] = s.get("task", df.get("task", ""))
            df["plant_leakage"] = s.get("plant_leakage", np.nan)
            df["n_train"] = s.get("n_train", np.nan)
            df["n_test"] = s.get("n_test", np.nan)
        else:
            df["experiment"] = rd.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError("Не найдено ни одного method_metrics.csv")
    out = pd.concat(rows, ignore_index=True)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["experiment", "method"])


def plot_heatmap(pivot, title, cbar_label, path, fmt=".3f", log_values=False):
    data = pivot.to_numpy(dtype=float)
    plot_data = np.log10(np.maximum(data, 1e-3)) if log_values else data
    fig_w = max(9, 1.5 * pivot.shape[1] + 4)
    fig_h = max(5, 0.55 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_data, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontsize=14, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                text = fmt_seconds(val) if log_values else format(val, fmt)
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_grouped_ba(df, path):
    experiments = list(df["experiment"].drop_duplicates())
    methods = [m for m in METHOD_ORDER if m in set(df["method"].astype(str))]
    x = np.arange(len(experiments))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(11, 2.2 * len(experiments)), 6))
    for r, method in enumerate(methods):
        vals = []
        for exp in experiments:
            sub = df[(df["experiment"] == exp) & (df["method"].astype(str) == method)]
            vals.append(float(sub["balanced_accuracy"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x - 0.4 + width / 2 + r * width, vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Сравнение качества по экспериментам")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_grouped_runtime(df, path):
    experiments = list(df["experiment"].drop_duplicates())
    methods = [m for m in METHOD_ORDER if m in set(df["method"].astype(str))]
    x = np.arange(len(experiments))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(11, 2.2 * len(experiments)), 6))
    for r, method in enumerate(methods):
        vals = []
        for exp in experiments:
            sub = df[(df["experiment"] == exp) & (df["method"].astype(str) == method)]
            vals.append(float(sub["time_sec"].iloc[0]) if len(sub) else np.nan)
        ax.bar(x - 0.4 + width / 2 + r * width, np.maximum(vals, 1e-3), width, label=method)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=20, ha="right")
    ax.set_ylabel("Время, секунды (логарифмическая шкала)")
    ax.set_title("Сравнение времени работы по экспериментам")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_table(df, path):
    d = df[["experiment", "method", "accuracy", "balanced_accuracy", "f1_macro", "n_features", "time_sec"]].copy()
    d = d.rename(columns={
        "experiment": "Эксперимент", "method": "Метод", "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy", "f1_macro": "F1 macro",
        "n_features": "Признаки", "time_sec": "Время"
    })
    for c in ["Accuracy", "Balanced accuracy", "F1 macro"]:
        d[c] = d[c].map(lambda x: f"{x:.3f}")
    d["Время"] = d["Время"].map(fmt_seconds)
    fig, ax = plt.subplots(figsize=(15, max(6, 0.33 * len(d) + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=d.values, colLabels=d.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    ax.set_title("Итоговая таблица метрик и времени", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    ensure_dir(args.outdir)
    df = load_runs(args.run_dirs)
    df.to_csv(Path(args.outdir) / "all_method_metrics.csv", index=False)
    ba = df.pivot_table(index="method", columns="experiment", values="balanced_accuracy", aggfunc="first")
    rt = df.pivot_table(index="method", columns="experiment", values="time_sec", aggfunc="first")
    ba = ba.reindex([m for m in METHOD_ORDER if m in ba.index])
    rt = rt.reindex([m for m in METHOD_ORDER if m in rt.index])
    ba.to_csv(Path(args.outdir) / "balanced_accuracy_matrix.csv")
    rt.to_csv(Path(args.outdir) / "runtime_matrix_sec.csv")
    plot_heatmap(ba, "Balanced accuracy по методам и экспериментам", "Balanced accuracy", Path(args.outdir) / "01_balanced_accuracy_heatmap_ru.png")
    plot_heatmap(rt, "Время работы по методам и экспериментам", "log10(seconds)", Path(args.outdir) / "02_runtime_heatmap_ru.png", log_values=True)
    plot_grouped_ba(df, Path(args.outdir) / "03_balanced_accuracy_grouped_ru.png")
    plot_grouped_runtime(df, Path(args.outdir) / "04_runtime_grouped_log_ru.png")
    plot_table(df, Path(args.outdir) / "00_all_metrics_table_ru.png")
    print("Saved:", args.outdir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
