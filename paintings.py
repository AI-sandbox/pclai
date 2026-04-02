#!/usr/bin/env python3

import os
import io
import re
import gzip
import glob
import json
import pickle
import pathlib
import argparse
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm

from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

mpl.rcParams["path.simplify"] = False

def load_inference_outputs(outdir: str | pathlib.Path, verbose: bool = True):
    outdir = pathlib.Path(outdir)

    results_path = outdir / "results.pkl.gz"
    results_cp_path = outdir / "results_cp.pkl.gz"
    stats_path = outdir / "stats.tsv"
    metadata_path = outdir / "metadata.json"

    if verbose:
        print(f"[load] Reading outputs from: {outdir}", flush=True)

    with gzip.open(results_path, "rb") as f:
        results = pickle.load(f)

    with gzip.open(results_cp_path, "rb") as f:
        results_cp = pickle.load(f)

    stats_df = pd.read_csv(stats_path, sep="\t") if stats_path.exists() else pd.DataFrame()

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    if verbose:
        print(f"[load] Loaded results for {len(results)} samples", flush=True)
        print(f"[load] stats_df shape = {stats_df.shape}", flush=True)

    return results, results_cp, stats_df, metadata


def ensure_hap_structure(results_for_one_sample: dict):
    """
    Expected format:
      {'chr14': {'h1': (W,2), 'h2': (W,2)}, ...}
    If older layout is flat, wrap it.
    """
    out = {}
    for chrk, val in results_for_one_sample.items():
        if isinstance(val, dict) and (("h1" in val) or ("h2" in val)):
            out[chrk] = val
        else:
            out[chrk] = {"h1": val}
    return out


def ensure_cp_structure(results_cp_for_one_sample: dict | None):
    """
    Expected format:
      {'chr14': {'h1': (W,), 'h2': (W,)}, ...}
    """
    if results_cp_for_one_sample is None:
        return None
    out = {}
    for chrk, val in results_cp_for_one_sample.items():
        if isinstance(val, dict) and (("h1" in val) or ("h2" in val)):
            out[chrk] = val
        else:
            out[chrk] = {"h1": val}
    return out

def _has_bcftools() -> bool:
    try:
        subprocess.run(
            ["bcftools", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _read_pos_with_bcftools(vcf_path: str) -> np.ndarray:
    res = subprocess.run(
        ["bcftools", "query", "-f", "%POS\n", vcf_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.fromiter((int(x) for x in res.stdout.splitlines() if x), dtype=np.int64)


def _read_pos_with_snputils(vcf_path: str) -> np.ndarray:
    from snputils.snp.io.read.vcf import VCFReaderPolars

    rd = VCFReaderPolars(vcf_path)
    df = rd.read()

    for col in ("POS", "pos", "Position", "position"):
        try:
            s = df[col]
            try:
                return s.to_numpy()
            except Exception:
                return s.values.astype(np.int64)
        except Exception:
            continue

    raise KeyError(f"Could not find POS column in {vcf_path}")


def resolve_chrom_vcf_path(vcf_dir: str, chrom: int) -> str | None:
    ch = f"chr{chrom}"

    candidate_patterns = [
        os.path.join(vcf_dir, f"{ch}.vcf.gz"),
        os.path.join(vcf_dir, f"{ch}.vcf"),
        os.path.join(vcf_dir, ch, f"{ch}.vcf.gz"),
        os.path.join(vcf_dir, ch, f"{ch}.vcf"),
        os.path.join(vcf_dir, ch, f"final.for_model.{ch}.vcf.gz"),
        os.path.join(vcf_dir, ch, f"final.for_model.{ch}.vcf"),
        os.path.join(vcf_dir, ch, "*.vcf.gz"),
        os.path.join(vcf_dir, ch, "*.vcf"),
    ]

    for pattern in candidate_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_pos_by_chr(vcf_dir: str, chroms=range(1, 23), prefix="chr") -> dict:
    """
    Returns:
      {'chr1': np.array([...]), ..., 'chr22': ...}
    """
    pos_by_chr = {}
    use_bcftools = _has_bcftools()

    print(f"[pos] Using bcftools={use_bcftools}", flush=True)

    for ch in chroms:
        ck = f"{prefix}{ch}"
        vcf = resolve_chrom_vcf_path(vcf_dir, ch)
        if vcf is None:
            print(f"[warn] VCF missing for {ck} under {vcf_dir} — skipping", flush=True)
            continue

        try:
            if use_bcftools:
                pos = _read_pos_with_bcftools(vcf)
            else:
                pos = _read_pos_with_snputils(vcf)

            if pos.size == 0:
                print(f"[warn] No variants in {vcf}", flush=True)
                continue

            pos = np.asarray(pos, dtype=np.int64)
            if (np.diff(pos) < 0).any():
                pos = np.sort(pos)

            pos_by_chr[ck] = pos
            print(f"[ok] {ck}: {pos.size:,} POS loaded from {vcf}", flush=True)

        except Exception as e:
            print(f"[err] {ck}: failed to load POS from {vcf}: {e}", flush=True)

    return pos_by_chr

HG38_CENTROMERES = {
    "chr1":  (121700000, 125100000),
    "chr2":  (91800000, 96000000),
    "chr3":  (87800000, 94000000),
    "chr4":  (48200000, 51800000),
    "chr5":  (46100000, 51400000),
    "chr6":  (58500000, 62600000),
    "chr7":  (58100000, 62100000),
    "chr8":  (43200000, 47200000),
    "chr9":  (42200000, 45500000),
    "chr10": (38000000, 41600000),
    "chr11": (51000000, 55800000),
    "chr12": (33200000, 37800000),
    "chr13": (16500000, 18900000),
    "chr14": (16100000, 18200000),
    "chr15": (17500000, 20500000),
    "chr16": (35300000, 38400000),
    "chr17": (22700000, 27400000),
    "chr18": (15400000, 21500000),
    "chr19": (24200000, 28100000),
    "chr20": (25700000, 30400000),
    "chr21": (10900000, 13000000),
    "chr22": (13700000, 17400000),
}

HG38_CHR_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
}

def chr_to_int(k):
    try:
        return int(str(k).lower().replace("chr", ""))
    except Exception:
        return 10**9


def pca_to_rgb_setup(founders_tsv, pca_constructor_path, scale_by_pca=True, margin=0.1):
    founders = pd.read_csv(founders_tsv, sep="\t")
    x1_f = founders["x1"].to_numpy(float)
    x2_f = founders["x2"].to_numpy(float)

    if scale_by_pca:
        with open(pca_constructor_path, "rb") as f:
            pca = pickle.load(f)
        s1 = float(np.sqrt(pca.explained_variance_[0]))
        s2 = float(np.sqrt(pca.explained_variance_[1]))
        x1_f, x2_f = x1_f / s1, x2_f / s2

    x1_min, x1_max = x1_f.min() - margin, x1_f.max() + margin
    x2_min, x2_max = x2_f.min() - margin, x2_f.max() + margin

    try:
        from skimage.color import lab2rgb

        def pca_to_rgb(x1, x2):
            L_fixed, ab_span = 80.0, 120.0
            a = ((x1 - x1_min) / (x1_max - x1_min + 1e-12) - 0.5) * 2.0 * ab_span
            b = ((x2 - x2_min) / (x2_max - x2_min + 1e-12) - 0.5) * 2.0 * ab_span
            L = np.full_like(a, L_fixed, float)
            Lab = np.stack([L, a, b], -1)
            return np.clip(lab2rgb(Lab[np.newaxis, ...])[0], 0, 1)

    except Exception:
        import matplotlib.colors as mcolors

        def pca_to_rgb(x1, x2):
            nx = (x1 - x1_min) / (x1_max - x1_min + 1e-12)
            ny = (x2 - x2_min) / (x2_max - x2_min + 1e-12)
            ang = np.arctan2(ny - 0.5, nx - 0.5)
            hue = (ang / (2 * np.pi)) % 1.0
            sat = np.clip(np.hypot(nx - 0.5, ny - 0.5) * np.sqrt(2), 0, 1)
            val = 0.92
            return mcolors.hsv_to_rgb(np.stack([hue, sat, np.full_like(hue, val)], -1))

    return pca_to_rgb


def _load_founders_and_bounds(founders_tsv, pca_constructor_path, scale_by_pca=True, margin=0.1):
    founders = pd.read_csv(founders_tsv, sep="\t")
    x1_f = founders["x1"].to_numpy(float)
    x2_f = founders["x2"].to_numpy(float)

    if scale_by_pca:
        with open(pca_constructor_path, "rb") as f:
            pca = pickle.load(f)
        s1 = float(np.sqrt(pca.explained_variance_[0]))
        s2 = float(np.sqrt(pca.explained_variance_[1]))
        x1_f, x2_f = x1_f / s1, x2_f / s2

    x1_min, x1_max = x1_f.min() - margin, x1_f.max() + margin
    x2_min, x2_max = x2_f.min() - margin, x2_f.max() + margin

    try:
        from skimage.color import lab2rgb

        def pca_to_rgb(x1, x2, L_fixed=80.0, ab_span=120.0):
            a = ((x1 - x1_min) / (x1_max - x1_min + 1e-12) - 0.5) * 2.0 * ab_span
            b = ((x2 - x2_min) / (x2_max - x2_min + 1e-12) - 0.5) * 2.0 * ab_span
            L = np.full_like(a, L_fixed, dtype=float)
            Lab = np.stack([L, a, b], axis=-1)
            return np.clip(lab2rgb(Lab[np.newaxis, ...])[0], 0, 1)

    except Exception:
        import matplotlib.colors as mcolors

        def pca_to_rgb(x1, x2, L_fixed=80.0, ab_span=90.0):
            nx = (x1 - x1_min) / (x1_max - x1_min + 1e-12)
            ny = (x2 - x2_min) / (x2_max - x2_min + 1e-12)
            ang = np.arctan2(ny - 0.5, nx - 0.5)
            hue = (ang / (2 * np.pi)) % 1.0
            r = np.hypot(nx - 0.5, ny - 0.5) * np.sqrt(2)
            sat = np.clip(r, 0, 1)
            val = 0.9
            return mcolors.hsv_to_rgb(np.stack([hue, sat, np.full_like(hue, val)], axis=-1))

    return x1_f, x2_f, (x1_min, x1_max, x2_min, x2_max), pca_to_rgb

def _cp_to_prob(cp_vec):
    if cp_vec is None:
        return None
    p = np.asarray(cp_vec).reshape(-1)
    if (p.min() < 0) or (p.max() > 1):
        p = 1.0 / (1.0 + np.exp(-p))
    return np.clip(p, 0.0, 1.0)


def _collect_points_weights_and_cp(
    results_for_one_sample: dict,
    results_cp_for_one_sample: dict | None,
    weights_mode: str = "most_confident",
    gamma: float = 2.5,
):
    xs_all, ys_all, ws_all, cp_all = [], [], [], []

    chr_keys = sorted(results_for_one_sample.keys(), key=chr_to_int)

    for chr_key in chr_keys:
        entry = results_for_one_sample.get(chr_key, {})
        cp_entry = results_cp_for_one_sample.get(chr_key, {}) if results_cp_for_one_sample is not None else {}

        for hap in ("h1", "h2"):
            coords = entry.get(hap, None)
            if coords is None:
                continue

            coords = np.asarray(coords)
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(
                    f"Expected coords[chr={chr_key}][{hap}] shape (n_win,>=2), got {coords.shape}"
                )

            x = coords[:, 0]
            y = coords[:, 1]

            cp_raw = None
            if cp_entry and hap in cp_entry:
                cp_raw = np.asarray(cp_entry[hap]).reshape(-1)
                cp = np.clip(cp_raw, 0.0, 1.0)
            else:
                cp = np.zeros_like(x, dtype=float)

            if weights_mode == "uniform" or results_cp_for_one_sample is None:
                w = np.ones_like(x, dtype=float)
            elif weights_mode == "most_confident":
                w = (1.0 - cp) ** gamma
            else:
                raise ValueError(f"Unknown weights_mode={weights_mode}")

            xs_all.append(x)
            ys_all.append(y)
            ws_all.append(w)
            cp_all.append(cp)

    if not xs_all:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    ws = np.concatenate(ws_all)
    cp_probs = np.concatenate(cp_all)
    return xs, ys, ws, cp_probs

def _capsule_path_v_rxry(xc, y0, y1, rx, ry, n_arc=512):
    if y1 <= y0 or rx <= 0 or ry <= 0:
        xL, xR = xc - max(rx, 1e-9), xc + max(rx, 1e-9)
        verts = [(xL, y0), (xR, y0), (xR, y1), (xL, y1), (xL, y0)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        return Path(verts, codes)

    h = float(y1 - y0)
    ry = min(ry, h / 2.0)

    th_bot = np.linspace(np.pi, 2 * np.pi, n_arc)
    th_top = np.linspace(0, np.pi, n_arc)
    bot = np.c_[xc + rx * np.cos(th_bot), (y0 + ry) + ry * np.sin(th_bot)]
    top = np.c_[xc + rx * np.cos(th_top), (y1 - ry) + ry * np.sin(th_top)]

    xL = xc - rx
    verts = np.vstack([[xL, y0 + ry], bot, [xc + rx, y1 - ry], top, [xL, y0 + ry]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def paint_centromere_waist_bp(
    ax,
    xc,
    L_bp,
    R_bp,
    *,
    bar_width,
    shrink=0.72,
    color="#9E9E9E",
    alpha=0.88,
    z=4,
    outline_lw=0.0,
    tick_color="#616161",
    tick_lw=0.9,
    tick_frac=0.45,
):
    if not (R_bp > L_bp):
        return

    ww = bar_width * float(shrink)
    r = min(ww / 2.0, (R_bp - L_bp) / 2.0)

    patch = mpatches.FancyBboxPatch(
        (xc - ww / 2.0, L_bp),
        ww,
        (R_bp - L_bp),
        boxstyle=mpatches.BoxStyle("Round", pad=0.0, rounding_size=r),
        facecolor=color,
        edgecolor=(color if outline_lw > 0 else "none"),
        linewidth=outline_lw,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)

    xL = xc - bar_width * tick_frac
    xR = xc + bar_width * tick_frac
    ax.plot([xL, xR], [L_bp, L_bp], color=tick_color, lw=tick_lw, zorder=z + 0.2, clip_on=True)
    ax.plot([xL, xR], [R_bp, R_bp], color=tick_color, lw=tick_lw, zorder=z + 0.2, clip_on=True)


def draw_chr_bp_colors_split_capsules(
    ax,
    *,
    xc: float,
    chrom: str,
    chrom_len_bp: int,
    pos_bp: np.ndarray,
    window_size_snps: int,
    window_rgb: np.ndarray,
    bar_width: float = 1.8,
    cap_px: int = 20,
    tip_aspect: float = 1.8,
    edgecolor="k",
    edge_lw=0.9,
    grey="#616161",
    n_arc: int = 512,
    z_base: int = 2,
    centromeres: dict | None = HG38_CENTROMERES,
    draw_waist: bool = True,
    window_alpha: float = 0.38,
):
    ax.figure.canvas.draw_idle()
    ax.figure.canvas.flush_events()
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((1, 1))
    yppu = (y1 - y0)

    rx = bar_width / 2.0
    ry = (cap_px * tip_aspect) / max(yppu, 1e-9)

    if centromeres and (chrom in centromeres):
        L_bp, R_bp = centromeres[chrom]
    else:
        L_bp, R_bp = 0, 0

    paths = []
    if L_bp > 0:
        paths.append(_capsule_path_v_rxry(xc, 0, L_bp, rx, ry, n_arc=n_arc))
    if R_bp < chrom_len_bp:
        paths.append(_capsule_path_v_rxry(xc, R_bp, chrom_len_bp, rx, ry, n_arc=n_arc))
    if not paths:
        paths = [_capsule_path_v_rxry(xc, 0, chrom_len_bp, rx, ry, n_arc=n_arc)]

    for p in paths:
        bg = mpatches.Rectangle(
            (xc - rx, 0),
            bar_width,
            chrom_len_bp,
            facecolor=grey,
            edgecolor="none",
            zorder=z_base,
            alpha=float(window_alpha),
        )
        bg.set_clip_path(p, transform=ax.transData)
        ax.add_artist(bg)

    pos0 = (np.asarray(pos_bp, dtype=np.int64) - 1) if pos_bp is not None else np.array([], dtype=np.int64)
    n = len(pos0)

    if n > 0:
        n_win_by_snps = (n + window_size_snps - 1) // window_size_snps
        n_win = min(len(window_rgb), n_win_by_snps)

        rects_low, colors_low = [], []
        rects_high, colors_high = [], []

        for w in range(n_win):
            i0 = w * window_size_snps
            if i0 >= n:
                break
            i1 = min((w + 1) * window_size_snps, n)

            y0b = int(pos0[i0])
            y1b = int(pos0[i1 - 1]) + 1
            if y1b <= y0b:
                continue
            col = window_rgb[w]

            if L_bp > 0:
                yy0 = y0b
                yy1 = min(y1b, L_bp)
                if yy1 > yy0:
                    rects_low.append(mpatches.Rectangle((xc - rx, yy0), bar_width, (yy1 - yy0), edgecolor="none"))
                    colors_low.append(col)

            if R_bp < chrom_len_bp:
                yy0 = max(y0b, R_bp)
                yy1 = y1b
                if yy1 > yy0:
                    rects_high.append(mpatches.Rectangle((xc - rx, yy0), bar_width, (yy1 - yy0), edgecolor="none"))
                    colors_high.append(col)

            if L_bp == 0 and R_bp == 0:
                rects_low.append(mpatches.Rectangle((xc - rx, y0b), bar_width, (y1b - y0b), edgecolor="none"))
                colors_low.append(col)

        if rects_low:
            white_low = PatchCollection(rects_low, facecolor="white", edgecolor="none", zorder=z_base + 1.0, match_original=True, alpha=1.0)
            white_low.set_clip_path(paths[0], transform=ax.transData)
            ax.add_collection(white_low)

        if rects_high and (len(paths) > 1):
            white_high = PatchCollection(rects_high, facecolor="white", edgecolor="none", zorder=z_base + 1.0, match_original=True, alpha=1.0)
            white_high.set_clip_path(paths[1], transform=ax.transData)
            ax.add_collection(white_high)

        if rects_low:
            pc_low = PatchCollection(rects_low, facecolor=colors_low, edgecolor="none", zorder=z_base + 1.1, match_original=True, alpha=float(window_alpha))
            pc_low.set_clip_path(paths[0], transform=ax.transData)
            ax.add_collection(pc_low)

        if rects_high and (len(paths) > 1):
            pc_high = PatchCollection(rects_high, facecolor=colors_high, edgecolor="none", zorder=z_base + 1.1, match_original=True, alpha=float(window_alpha))
            pc_high.set_clip_path(paths[1], transform=ax.transData)
            ax.add_collection(pc_high)

    if draw_waist and (L_bp < R_bp) and (0 < L_bp < chrom_len_bp):
        paint_centromere_waist_bp(
            ax, xc, L_bp, R_bp,
            bar_width=bar_width, shrink=0.72,
            color="#9E9E9E", alpha=0.88, z=z_base + 1.5,
        )

    for p in paths:
        ax.add_patch(mpatches.PathPatch(p, facecolor="none", edgecolor=edgecolor, lw=edge_lw, zorder=z_base + 2))


def plot_chromosome_painting_full_bp_with_legend(
    founders_tsv: str,
    pca_constructor_path: str,
    results_for_one_sample: dict,
    pos_by_chr: dict,
    window_size_snps: int = 1000,
    *,
    bar_width=1.8,
    intra_gap=0.35,
    inter_gap=1.55,
    figsize=(18, 8),
    dpi=200,
    cap_px=20,
    tip_aspect=1.8,
    edge_lw=0.9,
    grey="#616161",
    window_alpha=0.38,
    ax=None,
    fig=None,
    scale_by_pca=True,
    margin=0.1,
    legend_mb: float = 10.0,
    cM_per_Mb: float = 1.0,
    show_scale_bar: bool = True,
):
    pca_to_rgb = pca_to_rgb_setup(
        founders_tsv, pca_constructor_path, scale_by_pca=scale_by_pca, margin=margin
    )

    chrom_keys = sorted(
        [ck for ck in results_for_one_sample.keys() if ck in HG38_CHR_SIZES],
        key=chr_to_int
    )
    if not chrom_keys:
        raise ValueError("No chromosomes found in results_for_one_sample that match HG38_CHR_SIZES.")

    x_centers, x = {}, 0.0
    for ck in chrom_keys:
        x += bar_width / 2
        x_centers[(ck, "h1")] = x
        x += bar_width + intra_gap
        x_centers[(ck, "h2")] = x
        x += bar_width / 2 + inter_gap
    x_max = x

    max_len = max(HG38_CHR_SIZES[ck] for ck in chrom_keys)
    Y_PAD = int(0.01 * max_len)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=False)
        created_fig = True
    else:
        if fig is None:
            fig = ax.figure

    extra_x = bar_width * 1.5 if show_scale_bar else 0.0
    ax.set_xlim(-bar_width, x_max + extra_x)
    ax.set_ylim(-Y_PAD, max_len + Y_PAD)

    for ck in tqdm.tqdm(chrom_keys):
        chrom_len_bp = HG38_CHR_SIZES[ck]
        pos_bp = np.asarray(pos_by_chr.get(ck))
        for hap in ("h1", "h2"):
            arr = results_for_one_sample.get(ck, {}).get(hap)
            if arr is None or (np.asarray(arr).size == 0):
                window_rgb = np.zeros((0, 3))
            else:
                arr = np.asarray(arr)
                window_rgb = pca_to_rgb(arr[:, 0], arr[:, 1])

            xc = x_centers[(ck, hap)]
            draw_chr_bp_colors_split_capsules(
                ax,
                xc=xc,
                chrom=ck,
                chrom_len_bp=chrom_len_bp,
                pos_bp=pos_bp,
                window_size_snps=window_size_snps,
                window_rgb=window_rgb,
                bar_width=bar_width,
                cap_px=cap_px,
                tip_aspect=tip_aspect,
                edge_lw=edge_lw,
                grey=grey,
                centromeres=HG38_CENTROMERES,
                draw_waist=True,
                window_alpha=window_alpha,
            )

    xticks, xlabels = [], []
    for ck in chrom_keys:
        mid = 0.5 * (x_centers[(ck, "h1")] + x_centers[(ck, "h2")])
        xticks.append(mid)
        xlabels.append(ck)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("bp")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="x", length=0)

    if show_scale_bar and (legend_mb is not None) and (legend_mb > 0):
        legend_bp = legend_mb * 1e6
        max_bar_bp = 0.30 * max_len
        if legend_bp > max_bar_bp:
            scale = max_bar_bp / legend_bp
            legend_bp *= scale
            legend_mb *= scale

        if legend_bp > 0:
            L_cM = legend_mb * cM_per_Mb
            n_gen = 50.0 / L_cM if L_cM > 0 else np.nan

            x_leg = x_max + 0.6 * bar_width
            y0 = 0.08 * max_len
            y1 = y0 + legend_bp

            ax.plot([x_leg, x_leg], [y0, y1], color="black", lw=2.0)
            ax.plot([x_leg - bar_width * 0.15, x_leg + bar_width * 0.15], [y0, y0], color="black", lw=1.5)
            ax.plot([x_leg - bar_width * 0.15, x_leg + bar_width * 0.15], [y1, y1], color="black", lw=1.5)

            label_lines = [f"{legend_mb:.1f} Mb"]
            if L_cM > 0:
                label_lines.append(f"≈ {L_cM:.1f} cM")
                label_lines.append(f"≈ {n_gen:.1f} generations")
            label = "\n".join(label_lines)

            ax.text(
                x_leg + bar_width * 0.35,
                0.5 * (y0 + y1),
                label,
                va="center",
                ha="left",
                fontsize=9,
            )

    if created_fig:
        fig.tight_layout()
    return fig, ax

def plot_pca_windows_contour(
    founders_tsv: str,
    pca_constructor_path: str,
    results_for_one_sample: dict,
    results_cp_for_one_sample: dict | None = None,
    *,
    breakpoint_alpha: float | None = None,
    weights_mode: str = "most_confident",
    weight_gamma: float = 2.5,
    scale_by_pca: bool = True,
    margin: float = 0.1,
    bg_res: int = 400,
    hist_bins: int = 300,
    kde_sigma: float = 1.2,
    use_log: bool = True,
    contour_levels: int | list = 12,
    line_alpha_min: float = 0.18,
    line_alpha_max: float = 0.92,
    line_width: float = 0.8,
    label_mode: str = "log1p",
    label_fontsize: float = 8.5,
    founders_size: float = 18,
    founders_edge_lw: float = 0.15,
    founders_edge_color: str = "k",
    figsize=(7.5, 6.5),
    dpi=160,
    title: str = "Window density (confidence-weighted) in PCA space",
    ax=None,
    fig=None,
):
    x1_f, x2_f, (x1_min, x1_max, x2_min, x2_max), pca_to_rgb = _load_founders_and_bounds(
        founders_tsv,
        pca_constructor_path,
        scale_by_pca=scale_by_pca,
        margin=margin,
    )

    Xbg = np.linspace(x1_min, x1_max, bg_res)
    Ybg = np.linspace(x2_min, x2_max, bg_res)
    XXbg, YYbg = np.meshgrid(Xbg, Ybg)
    bg_rgb = pca_to_rgb(XXbg, YYbg)

    xs, ys, ws, cp_probs = _collect_points_weights_and_cp(
        results_for_one_sample,
        results_cp_for_one_sample,
        weights_mode=weights_mode,
        gamma=weight_gamma,
    )

    if breakpoint_alpha is not None and cp_probs.size:
        mask = cp_probs <= float(breakpoint_alpha)
        xs = xs[mask]
        ys = ys[mask]
        ws = ws[mask]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=False)
        created_fig = True
    else:
        if fig is None:
            fig = ax.figure

    ax.imshow(
        bg_rgb,
        origin="lower",
        extent=[x1_min, x1_max, x2_min, x2_max],
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
        alpha=0.38,
    )

    cl = None
    if xs.size:
        x_edges = np.linspace(x1_min, x1_max, hist_bins + 1)
        y_edges = np.linspace(x2_min, x2_max, hist_bins + 1)
        H, xe, ye = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=ws)

        try:
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(H, sigma=kde_sigma, mode="nearest")
        except Exception:
            Z = H

        if use_log:
            Z = np.log1p(Z)

        Xc = 0.5 * (xe[:-1] + xe[1:])
        Yc = 0.5 * (ye[:-1] + ye[1:])

        if isinstance(contour_levels, int):
            zmin = np.nanmin(Z)
            zmax = np.nanmax(Z)
            if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
                levels = np.linspace(zmin + 1e-12, zmax, contour_levels)
            else:
                levels = []
        else:
            levels = np.asarray(contour_levels, float)

        if len(levels) > 0:
            cl = ax.contour(Xc, Yc, Z.T, levels=levels, colors="k", linewidths=line_width, zorder=3.0)

            cols = getattr(cl, "collections", None)
            if cols is None:
                cols = cl.get_children()

            alphas = np.linspace(line_alpha_min, line_alpha_max, len(levels))
            for coll, a in zip(cols, alphas):
                try:
                    coll.set_alpha(float(a))
                except Exception:
                    pass

            if label_mode == "raw" and use_log:
                fmt = {lev: f"{np.round(np.expm1(lev), 2)}" for lev in cl.levels}
            else:
                fmt = {lev: f"{np.round(lev, 2)}" for lev in cl.levels}

            try:
                ax.clabel(cl, inline=True, fontsize=label_fontsize, fmt=fmt, inline_spacing=6)
            except Exception:
                ax.clabel(cl, inline=True, fontsize=label_fontsize, fmt="%g", inline_spacing=6)

    f_rgb = pca_to_rgb(x1_f, x2_f)
    ax.scatter(
        x1_f,
        x2_f,
        c=f_rgb,
        s=founders_size,
        edgecolor=founders_edge_color,
        linewidths=founders_edge_lw,
        alpha=0.9,
        zorder=4,
        label="Founders",
    )

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=1, alpha=0.8, zorder=5)
    ax.axvline(0, color="k", lw=1, alpha=0.8, zorder=5)

    for s in ax.spines.values():
        s.set_visible(False)

    if created_fig:
        fig.tight_layout()

    return fig, ax, cl

def get_sample_ids(results: dict, sample_id: str | None = None):
    if sample_id is not None:
        if sample_id not in results:
            raise KeyError(f"Sample {sample_id} not found in results")
        return [sample_id]
    return sorted(results.keys())

def cli_main():
    ap = argparse.ArgumentParser(
        description="Generate chromosome paintings and PCA contour plots from saved PCLAI inference outputs."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # --------------------------------------------------------
    # paint-chromosomes
    # --------------------------------------------------------
    ap_chr = sub.add_parser("paint-chromosomes", help="Generate chromosome painting(s)")
    ap_chr.add_argument("--results-dir", required=True, help="Directory containing results.pkl.gz and results_cp.pkl.gz")
    ap_chr.add_argument("--vcf-dir", required=True, help="Directory containing harmonized per-chromosome VCFs")
    ap_chr.add_argument("--founders-tsv", required=True)
    ap_chr.add_argument("--pca-constructor", required=True)
    ap_chr.add_argument("--sample-id", default=None, help="One sample ID to plot; if omitted, plot all samples")
    ap_chr.add_argument("--outdir", required=True)
    ap_chr.add_argument("--window-size-snps", type=int, default=1000)
    ap_chr.add_argument("--dpi", type=int, default=200)
    ap_chr.add_argument("--figwidth", type=float, default=18)
    ap_chr.add_argument("--figheight", type=float, default=8)
    ap_chr.add_argument("--legend-mb", type=float, default=10.0)
    ap_chr.add_argument("--cM-per-Mb", type=float, default=1.0)

    # --------------------------------------------------------
    # paint-pca
    # --------------------------------------------------------
    ap_pca = sub.add_parser("paint-pca", help="Generate PCA contour plot(s)")
    ap_pca.add_argument("--results-dir", required=True, help="Directory containing results.pkl.gz and results_cp.pkl.gz")
    ap_pca.add_argument("--founders-tsv", required=True)
    ap_pca.add_argument("--pca-constructor", required=True)
    ap_pca.add_argument("--sample-id", default=None, help="One sample ID to plot; if omitted, plot all samples")
    ap_pca.add_argument("--outdir", required=True)
    ap_pca.add_argument("--dpi", type=int, default=160)
    ap_pca.add_argument("--figwidth", type=float, default=7.5)
    ap_pca.add_argument("--figheight", type=float, default=6.5)
    ap_pca.add_argument("--breakpoint-alpha", type=float, default=None, help="Keep only windows with breakpoint prob <= this threshold")
    ap_pca.add_argument("--weight-gamma", type=float, default=2.5)

    args = ap.parse_args()

    results, results_cp, stats_df, metadata = load_inference_outputs(args.results_dir, verbose=True)
    os.makedirs(args.outdir, exist_ok=True)

    if args.command == "paint-chromosomes":
        pos_by_chr = build_pos_by_chr(args.vcf_dir, chroms=range(1, 23), prefix="chr")
        sample_ids = get_sample_ids(results, args.sample_id)

        print(f"[paint-chromosomes] plotting {len(sample_ids)} sample(s)", flush=True)

        for sample_id in sample_ids:
            print(f"[paint-chromosomes] sample={sample_id}", flush=True)

            pred_by_chr = ensure_hap_structure(results[sample_id])
            cp_by_chr = ensure_cp_structure(results_cp.get(sample_id, None))

            fig, ax = plot_chromosome_painting_full_bp_with_legend(
                founders_tsv=args.founders_tsv,
                pca_constructor_path=args.pca_constructor,
                results_for_one_sample=pred_by_chr,
                pos_by_chr=pos_by_chr,
                window_size_snps=args.window_size_snps,
                figsize=(args.figwidth, args.figheight),
                dpi=args.dpi,
                legend_mb=args.legend_mb,
                cM_per_Mb=args.cM_per_Mb,
            )

            out_png = os.path.join(args.outdir, f"{sample_id}.chromosome_painting.png")
            out_pdf = os.path.join(args.outdir, f"{sample_id}.chromosome_painting.pdf")

            fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
            fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

            print(f"[paint-chromosomes] wrote {out_png}", flush=True)
            print(f"[paint-chromosomes] wrote {out_pdf}", flush=True)

    elif args.command == "paint-pca":
        sample_ids = get_sample_ids(results, args.sample_id)
        print(f"[paint-pca] plotting {len(sample_ids)} sample(s)", flush=True)

        for sample_id in sample_ids:
            print(f"[paint-pca] sample={sample_id}", flush=True)

            pred_by_chr = ensure_hap_structure(results[sample_id])
            cp_by_chr = ensure_cp_structure(results_cp.get(sample_id, None))

            fig, ax, cl = plot_pca_windows_contour(
                founders_tsv=args.founders_tsv,
                pca_constructor_path=args.pca_constructor,
                results_for_one_sample=pred_by_chr,
                results_cp_for_one_sample=cp_by_chr,
                breakpoint_alpha=args.breakpoint_alpha,
                weight_gamma=args.weight_gamma,
                figsize=(args.figwidth, args.figheight),
                dpi=args.dpi,
                title=f"PCLAI PCA contour: {sample_id}",
            )

            out_png = os.path.join(args.outdir, f"{sample_id}.pca_contour.png")
            out_pdf = os.path.join(args.outdir, f"{sample_id}.pca_contour.pdf")

            fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
            fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

            print(f"[paint-pca] wrote {out_png}", flush=True)
            print(f"[paint-pca] wrote {out_pdf}", flush=True)


if __name__ == "__main__":
    cli_main()