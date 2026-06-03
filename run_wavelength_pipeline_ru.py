#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None


METHOD_ORDER = [
    "Полный спектр LR",
    "MI top-12 + LR",
    "PCA, 12 компонент + LR",
    "SPA top-12 + LR",
    "ADMM direct LS",
    "ADMM top-12 + LR",
    "Автоэнкодер latent + LR",
]

METHOD_KEY_TO_NAME = {
    "full": "Полный спектр LR",
    "mi": "MI top-12 + LR",
    "pca": "PCA, 12 компонент + LR",
    "spa": "SPA top-12 + LR",
    "admm_direct": "ADMM direct LS",
    "admm_top12": "ADMM top-12 + LR",
    "autoencoder": "Автоэнкодер latent + LR",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=conv)


def parse_stage_map(s):
    out = {}
    if not s:
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        name, days = part.split(":", 1)
        name = name.strip()
        for d in days.split(","):
            d = d.strip()
            if d:
                out[int(d)] = name
    return out


def load_plant_day(input_dir):
    input_dir = Path(input_dir)
    x_path = input_dir / "X_plant_day.npy"
    meta_path = input_dir / "meta_plant_day.csv"
    wl_path = input_dir / "wavelengths.npy"
    if x_path.exists() and meta_path.exists() and wl_path.exists():
        X = np.load(x_path)
        meta = pd.read_csv(meta_path)
        wavelengths = np.load(wl_path)
        return X.astype(float), meta, wavelengths.astype(float)

    pq = input_dir / "plant_day_spectra.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
        vector_cols = [c for c in df.columns if c.lower() in ["spectrum", "spec", "spec_cal", "mean_spectrum", "x"]]
        if not vector_cols:
            for c in df.columns:
                v = df[c].iloc[0]
                if isinstance(v, (list, tuple, np.ndarray)):
                    vector_cols.append(c)
                    break
        if not vector_cols:
            raise FileNotFoundError("Не нашла спектральную колонку в plant_day_spectra.parquet")
        spec_col = vector_cols[0]
        X = np.vstack(df[spec_col].apply(np.asarray).to_numpy()).astype(float)
        if "wavelengths" in df.columns:
            wavelengths = np.asarray(df["wavelengths"].iloc[0], dtype=float)
        elif wl_path.exists():
            wavelengths = np.load(wl_path).astype(float)
        else:
            wavelengths = np.arange(X.shape[1], dtype=float)
        meta = df.drop(columns=[spec_col], errors="ignore")
        return X, meta, wavelengths

    raise FileNotFoundError(
        f"В {input_dir} нужны X_plant_day.npy, meta_plant_day.csv, wavelengths.npy "
        "или plant_day_spectra.parquet"
    )


def infer_groups(meta):
    if "_group_key" in meta.columns:
        return meta["_group_key"].astype(str).to_numpy(), "_group_key"
    if "plant_label" in meta.columns:
        return meta["plant_label"].astype(str).to_numpy(), "plant_label"
    if "plant_prefix" in meta.columns and "plant_id" in meta.columns:
        return (meta["plant_prefix"].astype(str) + meta["plant_id"].astype(str)).to_numpy(), "plant_prefix+plant_id"
    if "health" in meta.columns and "plant_id" in meta.columns:
        return (meta["health"].astype(str) + "_" + meta["plant_id"].astype(str)).to_numpy(), "health+plant_id"
    if "plant_id" in meta.columns:
        return meta["plant_id"].astype(str).to_numpy(), "plant_id"
    raise ValueError("Не нашла колонку для групп растений")


def prepare_task(X, meta, task, drop_days, stage_map):
    meta = meta.copy().reset_index(drop=True)
    keep = np.ones(len(meta), dtype=bool)
    if drop_days and "day_num" in meta.columns:
        keep &= ~meta["day_num"].astype(int).isin(drop_days).to_numpy()

    if task == "health":
        if "health" not in meta.columns:
            raise ValueError("Для task=health нужна колонка health")
        y = meta["health"].astype(str).to_numpy()
        valid = np.isin(y, ["healthy", "diseased", "здоровые", "больные", "healthy_control"])
        keep &= valid
        y = y[keep]
        meta2 = meta.loc[keep].reset_index(drop=True)
        X2 = X[keep]
        return X2, meta2, y

    if task == "stages":
        if "day_num" not in meta.columns:
            raise ValueError("Для task=stages нужна колонка day_num")
        if "health" in meta.columns:
            keep &= meta["health"].astype(str).str.lower().isin(["diseased", "больные", "infected"]).to_numpy()
        days = meta["day_num"].astype(int).to_numpy()
        labels = np.array([stage_map.get(int(d), None) for d in days], dtype=object)
        keep &= pd.notna(labels)
        y = labels[keep].astype(str)
        meta2 = meta.loc[keep].reset_index(drop=True)
        X2 = X[keep]
        return X2, meta2, y

    raise ValueError(task)


def make_split(X, y, groups, n_splits=5, fold_index=0, seed=42):
    n_groups = len(np.unique(groups))
    n_splits_actual = min(int(n_splits), n_groups)
    n_splits_actual = max(2, n_splits_actual)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
        tr, te = splits[int(fold_index) % len(splits)]
        method = "StratifiedGroupKFold"
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=1 / n_splits_actual, random_state=seed)
        tr, te = next(splitter.split(X, y, groups))
        method = "GroupShuffleSplit"
    leak = len(set(groups[tr]).intersection(set(groups[te])))
    return tr, te, {"split_method": method, "n_splits_actual": n_splits_actual, "plant_leakage": leak}


def fit_lr_eval(Xtr, Xte, ytr, yte, selected_idx=None, pca_components=None, seed=42):
    t0 = time.perf_counter()
    if selected_idx is not None:
        Xtr0 = Xtr[:, selected_idx]
        Xte0 = Xte[:, selected_idx]
        scaler = StandardScaler().fit(Xtr0)
        Xtr1 = scaler.transform(Xtr0)
        Xte1 = scaler.transform(Xte0)
    elif pca_components is not None:
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        ncomp = min(int(pca_components), Xtr_s.shape[1], max(1, Xtr_s.shape[0] - 1))
        pca = PCA(n_components=ncomp, random_state=seed).fit(Xtr_s)
        Xtr1 = pca.transform(Xtr_s)
        Xte1 = pca.transform(Xte_s)
    else:
        scaler = StandardScaler().fit(Xtr)
        Xtr1 = scaler.transform(Xtr)
        Xte1 = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(Xtr1, ytr)
    pred = clf.predict(Xte1)
    elapsed = time.perf_counter() - t0
    return pred, elapsed


def metrics_dict(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def topk_from_scores(scores, k):
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return np.argsort(scores)[::-1][: int(k)]


def mi_select(Xtr, ytr, k, seed=42):
    scores = mutual_info_classif(Xtr, ytr, random_state=seed, discrete_features=False)
    return topk_from_scores(scores, k), scores


def spa_select(Xtr, ytr, wavelengths, k, min_gap_nm=0.0):
    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    rel, _ = f_classif(Xs, ytr)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    if rel.max() > rel.min():
        reln = (rel - rel.min()) / (rel.max() - rel.min())
    else:
        reln = np.ones_like(rel)
    selected = [int(np.argmax(reln))]
    p = Xs.shape[1]
    while len(selected) < min(k, p):
        S = Xs[:, selected]
        best_j, best_score = None, -np.inf
        residuals = []
        candidates = []
        for j in range(p):
            if j in selected:
                continue
            if min_gap_nm > 0 and len(wavelengths) == p:
                if any(abs(float(wavelengths[j]) - float(wavelengths[s])) < min_gap_nm for s in selected):
                    continue
            v = Xs[:, j]
            coef, *_ = np.linalg.lstsq(S, v, rcond=None)
            r = v - S @ coef
            residuals.append(float(np.linalg.norm(r)))
            candidates.append(j)
        if not candidates:
            break
        max_res = max(residuals) if max(residuals) > 0 else 1.0
        for j, rnorm in zip(candidates, residuals):
            score = float(reln[j]) * (0.25 + 0.75 * float(rnorm) / max_res)
            if score > best_score:
                best_score = score
                best_j = j
        selected.append(int(best_j))
    scores = np.zeros(p)
    for rank, j in enumerate(selected):
        scores[j] = len(selected) - rank
    return np.array(selected, dtype=int), scores


def soft_threshold_torch(x, tau):
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def run_admm_ls(Xtr, Xte, ytr, yte, args, classes):
    if torch is None:
        raise RuntimeError("PyTorch не установлен, ADMM на GPU недоступен")
    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested)
    dtype = torch.float32
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype("float32")
    Xte_s = scaler.transform(Xte).astype("float32")
    le = LabelEncoder().fit(classes)
    ytr_i = le.transform(ytr)
    yte_i = le.transform(yte)
    n, p = Xtr_s.shape
    c = len(le.classes_)
    Y = np.zeros((n, c), dtype="float32")
    Y[np.arange(n), ytr_i] = 1.0
    Xg = torch.tensor(Xtr_s, device=device, dtype=dtype)
    Xteg = torch.tensor(Xte_s, device=device, dtype=dtype)
    Yg = torch.tensor(Y, device=device, dtype=dtype)
    D = torch.zeros((p - 1, p), device=device, dtype=dtype)
    idx = torch.arange(p - 1, device=device)
    D[idx, idx] = -1.0
    D[idx, idx + 1] = 1.0
    I = torch.eye(p, device=device, dtype=dtype)
    XtX = (Xg.T @ Xg) / float(n)
    XtY = (Xg.T @ Yg) / float(n)
    DtD = D.T @ D
    A = XtX + args.lambda_l2 * I + args.rho_z * I + args.rho_v * DtD
    B = torch.zeros((p, c), device=device, dtype=dtype)
    Z = torch.zeros_like(B)
    U = torch.zeros_like(B)
    V = torch.zeros((p - 1, c), device=device, dtype=dtype)
    T = torch.zeros_like(V)
    history = []
    converged = False
    t0 = time.perf_counter()
    for it in range(1, int(args.admm_max_iter) + 1):
        Z_old = Z.clone()
        V_old = V.clone()
        rhs = XtY + args.rho_z * (Z - U) + args.rho_v * (D.T @ (V - T))
        B = torch.linalg.solve(A, rhs)
        Z = soft_threshold_torch(B + U, args.lambda_l1 / args.rho_z)
        DB = D @ B
        V = soft_threshold_torch(DB + T, args.lambda_tv / args.rho_v)
        U = U + B - Z
        T = T + DB - V
        primal = torch.sqrt(torch.sum((B - Z) ** 2) + torch.sum((DB - V) ** 2)).item()
        dual = torch.sqrt(torch.sum((args.rho_z * (Z - Z_old)) ** 2) + torch.sum((args.rho_v * (D.T @ (V - V_old))) ** 2)).item()
        if it == 1 or it % args.admm_log_every == 0 or it == args.admm_max_iter:
            obj = (0.5 / float(n)) * torch.sum((Xg @ B - Yg) ** 2)
            obj = obj + args.lambda_l1 * torch.sum(torch.abs(B))
            obj = obj + args.lambda_tv * torch.sum(torch.abs(D @ B))
            obj = obj + 0.5 * args.lambda_l2 * torch.sum(B ** 2)
            history.append({"iteration": it, "objective": float(obj.item()), "primal_residual": primal, "dual_residual": dual})
        if primal < args.admm_tol and dual < args.admm_tol:
            converged = True
            break
        if args.max_seconds and (time.perf_counter() - t0) > args.max_seconds:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        scores_te = Xteg @ B
        pred_i = torch.argmax(scores_te, dim=1).detach().cpu().numpy()
        pred = le.inverse_transform(pred_i)
        B_np = B.detach().cpu().numpy()
    row_scores = np.linalg.norm(B_np, axis=1)
    admm_info = {
        "device_used": str(device),
        "dtype_used": "float32",
        "converged": bool(converged),
        "iterations_done": int(it),
        "elapsed_sec": float(elapsed),
        "nonzero_wavelengths": int(np.sum(row_scores > 1e-8)),
    }
    return pred, row_scores, pd.DataFrame(history), admm_info, elapsed


class DenseAE(nn.Module):
    def __init__(self, p, latent_dim):
        super().__init__()
        h1 = min(128, max(32, p // 2))
        h2 = min(64, max(16, p // 4))
        self.encoder = nn.Sequential(
            nn.Linear(p, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, p)
        )
    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z


def run_autoencoder(Xtr, Xte, ytr, yte, args):
    if torch is None:
        raise RuntimeError("PyTorch не установлен, автоэнкодер недоступен")
    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype("float32")
    Xte_s = scaler.transform(Xte).astype("float32")
    model = DenseAE(Xtr_s.shape[1], int(args.ae_latent_dim)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.ae_weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=min(args.ae_batch_size, len(ds)), shuffle=True, drop_last=False)
    losses = []
    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, int(args.ae_epochs) + 1):
        vals = []
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            rec, _ = model(xb)
            loss = loss_fn(rec, xb)
            loss.backward()
            opt.step()
            vals.append(float(loss.item()))
        losses.append({"epoch": epoch, "loss": float(np.mean(vals))})
    if device.type == "cuda":
        torch.cuda.synchronize()
    ae_time = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        Ztr = model.encoder(torch.tensor(Xtr_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
        Zte = model.encoder(torch.tensor(Xte_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    t1 = time.perf_counter()
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=args.seed)
    clf.fit(Ztr, ytr)
    pred = clf.predict(Zte)
    total_time = ae_time + (time.perf_counter() - t1)
    return pred, total_time, pd.DataFrame(losses), {"device_used": str(device), "latent_dim": int(args.ae_latent_dim), "epochs": int(args.ae_epochs)}


def fmt_seconds(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.2f} с"
    if sec < 3600:
        return f"{sec/60:.2f} мин"
    return f"{sec/3600:.2f} ч"


def plot_results_table(df, path):
    cols = ["method", "accuracy", "balanced_accuracy", "f1_macro", "n_features", "time_sec"]
    d = df[cols].copy()
    d = d.rename(columns={
        "method": "Метод", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
        "f1_macro": "F1 macro", "n_features": "Признаки", "time_sec": "Время"
    })
    for c in ["Accuracy", "Balanced accuracy", "F1 macro"]:
        d[c] = d[c].map(lambda x: f"{x:.3f}")
    d["Время"] = d["Время"].map(fmt_seconds)
    fig, ax = plt.subplots(figsize=(13, max(3, 0.45 * len(d) + 1.2)))
    ax.axis("off")
    tbl = ax.table(cellText=d.values, colLabels=d.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    ax.set_title("Сравнение методов", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ba(df, path):
    d = df.sort_values("balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(d))))
    ax.barh(d["method"], d["balanced_accuracy"])
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Качество классификации по методам")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(d["balanced_accuracy"]):
        ax.text(min(v + 0.015, 1.02), i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_runtime(df, path):
    d = df.sort_values("time_sec", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(d))))
    vals = np.maximum(d["time_sec"].to_numpy(float), 1e-3)
    ax.barh(d["method"], vals)
    ax.set_xscale("log")
    ax.set_xlabel("Время работы, секунды (логарифмическая шкала)")
    ax.set_title("Вычислительная стоимость методов")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, fmt_seconds(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_acc_f1(df, path):
    d = df.copy()
    x = np.arange(len(d))
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(d)), 5))
    ax.bar(x - width, d["accuracy"], width, label="Accuracy")
    ax.bar(x, d["balanced_accuracy"], width, label="Balanced accuracy")
    ax.bar(x + width, d["f1_macro"], width, label="F1 macro")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(d["method"], rotation=35, ha="right")
    ax.set_title("Метрики качества")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_selected_waves(X, y, wavelengths, selected, path):
    if not selected:
        return
    methods = list(selected.keys())
    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    classes = list(pd.Series(y).astype(str).unique())
    for ax, method in zip(axes, methods):
        for cls in classes:
            m = y.astype(str) == cls
            ax.plot(wavelengths, X[m].mean(axis=0), label=str(cls), linewidth=1.5)
        idx = selected[method]
        for j in idx:
            ax.axvline(wavelengths[j], linewidth=0.8, alpha=0.45)
        wl_text = ", ".join([f"{wavelengths[j]:.1f}" for j in idx[:12]])
        ax.set_title(f"{method}: выбранные длины волн ({wl_text} нм)", fontsize=10)
        ax.set_ylabel("Отражение")
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Длина волны, нм")
    fig.suptitle("Выбранные волны на средних спектрах", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_admm_history(hist, outdir):
    if hist is None or hist.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["iteration"], hist["primal_residual"], label="primal residual")
    ax.plot(hist["iteration"], hist["dual_residual"], label="dual residual")
    ax.set_yscale("log")
    ax.set_xlabel("Итерация ADMM")
    ax.set_ylabel("Остаток, log")
    ax.set_title("Сходимость ADMM по остаткам")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "06_admm_residuals_ru.png", dpi=220)
    plt.close(fig)

    if "objective" in hist.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist["iteration"], hist["objective"])
        ax.set_xlabel("Итерация ADMM")
        ax.set_ylabel("Значение целевой функции")
        ax.set_title("Траектория целевой функции ADMM")
        fig.tight_layout()
        fig.savefig(Path(outdir) / "07_admm_objective_ru.png", dpi=220)
        plt.close(fig)


def plot_ae_loss(losses, outdir):
    if losses is None or losses.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses["epoch"], losses["loss"])
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE реконструкции")
    ax.set_title("Обучение автоэнкодера")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "08_autoencoder_loss_ru.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--task", choices=["health", "stages"], required=True)
    ap.add_argument("--drop-days", type=int, nargs="*", default=[])
    ap.add_argument("--stage-map", default="")
    ap.add_argument("--k-waves", type=int, default=12)
    ap.add_argument("--methods", nargs="+", default=["full", "mi", "pca", "spa", "admm_direct", "admm_top12", "autoencoder"],
                    choices=list(METHOD_KEY_TO_NAME.keys()))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--fold-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--lambda-l1", type=float, default=1e-3)
    ap.add_argument("--lambda-tv", type=float, default=5e-3)
    ap.add_argument("--lambda-l2", type=float, default=1e-3)
    ap.add_argument("--rho-z", type=float, default=1.0)
    ap.add_argument("--rho-v", type=float, default=1.0)
    ap.add_argument("--admm-max-iter", type=int, default=2000)
    ap.add_argument("--admm-tol", type=float, default=1e-4)
    ap.add_argument("--admm-log-every", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--ae-latent-dim", type=int, default=8)
    ap.add_argument("--ae-epochs", type=int, default=300)
    ap.add_argument("--ae-batch-size", type=int, default=64)
    ap.add_argument("--ae-lr", type=float, default=1e-3)
    ap.add_argument("--ae-weight-decay", type=float, default=1e-5)
    ap.add_argument("--spa-min-gap-nm", type=float, default=0.0)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    stage_map = parse_stage_map(args.stage_map)
    X, meta, wavelengths = load_plant_day(args.input_dir)
    X, meta, y = prepare_task(X, meta, args.task, args.drop_days, stage_map)
    groups, group_source = infer_groups(meta)
    tr, te, split_info = make_split(X, y, groups, args.n_splits, args.fold_index, args.seed)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    classes = list(LabelEncoder().fit(y).classes_)

    rows = []
    predictions = {}
    selected = {}
    selected_rows = []
    admm_hist = None
    ae_losses = None

    def add_method(name, pred, time_sec, n_features, extra=None):
        m = metrics_dict(yte, pred)
        row = {
            "dataset": args.dataset_name,
            "task": args.task,
            "method": name,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "f1_macro": m["f1_macro"],
            "n_features": int(n_features),
            "time_sec": float(time_sec),
        }
        if extra:
            row.update(extra)
        rows.append(row)
        predictions[name] = pred

    if "full" in args.methods:
        pred, sec = fit_lr_eval(Xtr, Xte, ytr, yte, seed=args.seed)
        add_method(METHOD_KEY_TO_NAME["full"], pred, sec, X.shape[1])

    if "mi" in args.methods:
        t0 = time.perf_counter()
        idx, scores = mi_select(Xtr, ytr, args.k_waves, args.seed)
        pred, lr_sec = fit_lr_eval(Xtr, Xte, ytr, yte, selected_idx=idx, seed=args.seed)
        sec = time.perf_counter() - t0
        name = METHOD_KEY_TO_NAME["mi"]
        selected[name] = idx
        add_method(name, pred, sec, len(idx), {"selection_time_sec": sec - lr_sec, "classifier_time_sec": lr_sec})
        for rank, j in enumerate(idx, 1):
            selected_rows.append({"method": name, "rank": rank, "index": int(j), "wavelength_nm": float(wavelengths[j]), "score": float(scores[j])})

    if "pca" in args.methods:
        pred, sec = fit_lr_eval(Xtr, Xte, ytr, yte, pca_components=args.k_waves, seed=args.seed)
        add_method(METHOD_KEY_TO_NAME["pca"], pred, sec, min(args.k_waves, X.shape[1]))

    if "spa" in args.methods:
        t0 = time.perf_counter()
        idx, scores = spa_select(Xtr, ytr, wavelengths, args.k_waves, args.spa_min_gap_nm)
        pred, lr_sec = fit_lr_eval(Xtr, Xte, ytr, yte, selected_idx=idx, seed=args.seed)
        sec = time.perf_counter() - t0
        name = METHOD_KEY_TO_NAME["spa"]
        selected[name] = idx
        add_method(name, pred, sec, len(idx), {"selection_time_sec": sec - lr_sec, "classifier_time_sec": lr_sec})
        for rank, j in enumerate(idx, 1):
            selected_rows.append({"method": name, "rank": rank, "index": int(j), "wavelength_nm": float(wavelengths[j]), "score": float(scores[j])})

    admm_scores = None
    admm_elapsed = None
    if ("admm_direct" in args.methods) or ("admm_top12" in args.methods):
        pred, admm_scores, admm_hist, info, admm_elapsed = run_admm_ls(Xtr, Xte, ytr, yte, args, classes)
        if "admm_direct" in args.methods:
            add_method(METHOD_KEY_TO_NAME["admm_direct"], pred, admm_elapsed, info["nonzero_wavelengths"], {
                "admm_converged": info["converged"],
                "admm_iterations": info["iterations_done"],
                "admm_device": info["device_used"],
            })
        if "admm_top12" in args.methods:
            idx = topk_from_scores(admm_scores, args.k_waves)
            t0 = time.perf_counter()
            pred2, lr_sec = fit_lr_eval(Xtr, Xte, ytr, yte, selected_idx=idx, seed=args.seed)
            total_sec = admm_elapsed + (time.perf_counter() - t0)
            name = METHOD_KEY_TO_NAME["admm_top12"]
            selected[name] = idx
            add_method(name, pred2, total_sec, len(idx), {
                "selection_time_sec": admm_elapsed,
                "classifier_time_sec": lr_sec,
                "admm_converged": info["converged"],
                "admm_iterations": info["iterations_done"],
                "admm_device": info["device_used"],
            })
            for rank, j in enumerate(idx, 1):
                selected_rows.append({"method": name, "rank": rank, "index": int(j), "wavelength_nm": float(wavelengths[j]), "score": float(admm_scores[j])})
        if admm_hist is not None:
            admm_hist.to_csv(Path(args.outdir) / "admm_history.csv", index=False)

    if "autoencoder" in args.methods:
        pred, sec, ae_losses, ae_info = run_autoencoder(Xtr, Xte, ytr, yte, args)
        add_method(METHOD_KEY_TO_NAME["autoencoder"], pred, sec, int(args.ae_latent_dim), ae_info)
        if ae_losses is not None:
            ae_losses.to_csv(Path(args.outdir) / "autoencoder_loss.csv", index=False)

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)
    df.to_csv(Path(args.outdir) / "method_metrics.csv", index=False)
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(Path(args.outdir) / "selected_wavelengths.csv", index=False)

    best = df.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_method = str(best["method"])
    best_pred = predictions[best_method]
    labels = list(LabelEncoder().fit(np.concatenate([yte, best_pred])).classes_)
    cm = confusion_matrix(yte, best_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(Path(args.outdir) / "confusion_matrix_best.csv")
    pd.DataFrame(classification_report(yte, best_pred, labels=labels, output_dict=True, zero_division=0)).T.to_csv(Path(args.outdir) / "classification_report_best.csv")
    pd.DataFrame({"y_true": yte, "y_pred": best_pred, "method": best_method}).to_csv(Path(args.outdir) / "predictions_best.csv", index=False)

    summary = {
        "dataset": args.dataset_name,
        "task": args.task,
        "input_dir": args.input_dir,
        "outdir": args.outdir,
        "n_objects": int(len(X)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "n_groups_total": int(len(np.unique(groups))),
        "n_groups_train": int(len(np.unique(groups[tr]))),
        "n_groups_test": int(len(np.unique(groups[te]))),
        "group_source": group_source,
        "classes": classes,
        "drop_days": args.drop_days,
        "stage_map": args.stage_map,
        "best_method": best_method,
        "best_balanced_accuracy": float(best["balanced_accuracy"]),
    }
    summary.update(split_info)
    save_json(summary, Path(args.outdir) / "run_summary.json")

    plot_results_table(df, Path(args.outdir) / "00_results_table_ru.png")
    plot_ba(df, Path(args.outdir) / "01_balanced_accuracy_ru.png")
    plot_runtime(df, Path(args.outdir) / "02_runtime_log_ru.png")
    plot_acc_f1(df, Path(args.outdir) / "03_accuracy_ba_f1_ru.png")
    plot_selected_waves(X, y.astype(str), wavelengths, selected, Path(args.outdir) / "04_selected_wavelengths_ru.png")
    plot_confusion(cm, labels, f"Матрица ошибок: лучший метод — {best_method}", Path(args.outdir) / "05_confusion_best_ru.png")
    plot_admm_history(admm_hist, args.outdir)
    plot_ae_loss(ae_losses, args.outdir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(df.to_string(index=False))
    print("Saved:", args.outdir)


if __name__ == "__main__":
    main()
