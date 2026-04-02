import os
import io
import json
import glob
import time
import shutil
import subprocess
from collections import defaultdict
import argparse
import gzip
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from snputils.snp.io.read.vcf import VCFReaderPolars


def _log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, flush=True)

def _section(title: str, verbose: bool = True):
    if verbose:
        print("\n" + "=" * 80, flush=True)
        print(title, flush=True)
        print("=" * 80, flush=True)

def _subsection(title: str, verbose: bool = True):
    if verbose:
        print("\n" + "-" * 80, flush=True)
        print(title, flush=True)
        print("-" * 80, flush=True)

# Environment / bundle helpers
def _require_bcftools(verbose: bool = True):
    path = shutil.which("bcftools")
    if path is None:
        raise RuntimeError("bcftools not found in PATH")
    _log(f"[env] bcftools found at: {path}", verbose=verbose)


def load_bundle_manifest(bundle_dir: str, verbose: bool = True) -> dict:
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")

    _log(f"[bundle] Loading manifest: {manifest_path}", verbose=verbose)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    _log(f"[bundle] Manifest format: {manifest.get('format', 'UNKNOWN')}", verbose=verbose)
    _log(f"[bundle] Number of exported model entries: {len(manifest.get('models', []))}", verbose=verbose)
    if "window_size" in manifest:
        _log(f"[bundle] window_size = {manifest['window_size']}", verbose=verbose)
    if "fixed_chunk_snps" in manifest:
        _log(f"[bundle] fixed_chunk_snps = {manifest['fixed_chunk_snps']}", verbose=verbose)

    return manifest


def load_exported_model(bundle_dir: str, rel_model_path: str, device: str | None = None, verbose: bool = True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full_model_path = os.path.join(bundle_dir, rel_model_path)
    if not os.path.isfile(full_model_path):
        raise FileNotFoundError(f"Exported model file not found: {full_model_path}")

    _log(f"[model] Loading exported model: {full_model_path}", verbose=verbose)
    _log(f"[model] Requested device: {device}", verbose=verbose)

    ep = torch.export.load(full_model_path)
    model = ep.module().to(device)   # no .eval() for exported program

    _log(f"[model] Exported model loaded successfully on {device}", verbose=verbose)
    return model

# Flexible VCF path resolution
def resolve_chrom_vcf_path(vcf_dir: str, chrom: int, verbose: bool = True) -> str | None:
    """
    Try several collaborator-friendly file layouts.
    Returns the first matching VCF path, or None if nothing found.
    """

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

    _log(f"[vcf] Resolving VCF path for {ch} inside: {vcf_dir}", verbose=verbose)

    for pattern in candidate_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            if len(matches) > 1:
                _log(f"[vcf] Multiple matches for pattern {pattern}; using first: {matches[0]}", verbose=verbose)
            else:
                _log(f"[vcf] Matched pattern {pattern}: {matches[0]}", verbose=verbose)
            return matches[0]

    _log(f"[vcf] No VCF found for {ch}", verbose=verbose)
    return None

# VCF readers
def extract_variant_table(vcf_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Ordered CHROM/POS/ID/REF/ALT from the collaborator VCF.
    """
    _require_bcftools(verbose=verbose)
    _log(f"[vcf-meta] Extracting CHROM/POS/ID/REF/ALT from: {vcf_path}", verbose=verbose)

    cmd = ["bcftools", "query", "-f", "%CHROM\t%POS\t%ID\t%REF\t%ALT\n", vcf_path]
    t0 = time.time()
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    dt = time.time() - t0

    df = pd.read_csv(
        io.StringIO(res.stdout),
        sep="\t",
        header=None,
        names=["chrom", "pos", "rsid", "ref", "alt"],
        dtype={"chrom": str, "pos": np.int64, "rsid": str, "ref": str, "alt": str},
    )
    df["rsid"] = df["rsid"].fillna(".").astype(str)
    df["vcf_index"] = np.arange(len(df), dtype=np.int64)

    _log(f"[vcf-meta] Loaded {len(df):,} variant rows in {dt:.2f}s", verbose=verbose)
    if len(df) > 0:
        _log(
            f"[vcf-meta] First variant: chrom={df.iloc[0]['chrom']} pos={df.iloc[0]['pos']} rsid={df.iloc[0]['rsid']}",
            verbose=verbose,
        )

    return df

def read_vcf_gt(vcf_path: str, verbose: bool = True):
    """
    Returns:
      gt: (n_snps, n_samples, 2)
      samples: list[str]
    """
    _log(f"[vcf-gt] Reading genotypes with VCFReaderPolars: {vcf_path}", verbose=verbose)
    t0 = time.time()

    reader = VCFReaderPolars(vcf_path)
    vcf_data = reader.read()
    gt = vcf_data["calldata_gt"][:, :, :]
    samples = [str(x) for x in list(vcf_data["samples"])]

    dt = time.time() - t0
    _log(f"[vcf-gt] GT shape = {gt.shape}", verbose=verbose)
    _log(f"[vcf-gt] Number of samples = {len(samples)}", verbose=verbose)
    _log(f"[vcf-gt] Read completed in {dt:.2f}s", verbose=verbose)

    if samples:
        _log(f"[vcf-gt] First few samples: {samples[:5]}", verbose=verbose)

    return gt, samples

# SNP matching / tensor assembly
def build_model_input_from_vcf(
    vcf_path: str,
    required_snp_manifest_path: str,
    fill_missing_value: int = 0,
    validate_ref_alt: bool = False,
    verbose: bool = True,
):
    """
    Reorders the collaborator VCF into the exact SNP order expected by one exported model.

    Returns
    -------
    x : torch.FloatTensor
        Shape (n_samples, 2, n_required_snps)
    samples : list[str]
    stats : dict
    """
    _subsection("Building model input tensor", verbose=verbose)
    _log(f"[input] Required SNP manifest: {required_snp_manifest_path}", verbose=verbose)
    _log(f"[input] Collaborator VCF: {vcf_path}", verbose=verbose)
    _log(f"[input] fill_missing_value = {fill_missing_value}", verbose=verbose)
    _log(f"[input] validate_ref_alt = {validate_ref_alt}", verbose=verbose)

    req = pd.read_csv(required_snp_manifest_path, sep="\t")
    _log(f"[input] Required SNP count = {len(req):,}", verbose=verbose)

    vcf_df = extract_variant_table(vcf_path, verbose=verbose)
    gt, samples = read_vcf_gt(vcf_path, verbose=verbose)

    if len(vcf_df) != gt.shape[0]:
        raise ValueError(
            f"Variant metadata length ({len(vcf_df)}) != GT rows ({gt.shape[0]}) for {vcf_path}"
        )

    rsid_to_idx = {}
    duplicate_rsid_count = 0
    missing_id_in_vcf_count = 0

    for row in vcf_df.itertuples(index=False):
        rid = str(row.rsid)
        if rid == ".":
            missing_id_in_vcf_count += 1
            continue
        if rid in rsid_to_idx:
            duplicate_rsid_count += 1
            continue
        rsid_to_idx[rid] = int(row.vcf_index)

    _log(f"[input] Unique usable rsIDs in VCF = {len(rsid_to_idx):,}", verbose=verbose)
    _log(f"[input] VCF records with missing rsID '.' = {missing_id_in_vcf_count:,}", verbose=verbose)
    _log(f"[input] Duplicate rsIDs skipped after first occurrence = {duplicate_rsid_count:,}", verbose=verbose)

    n_req = len(req)
    n_samples = gt.shape[1]
    out = np.full((n_req, n_samples, 2), fill_missing_value, dtype=gt.dtype)

    matched = 0
    missing_rsids = []
    refalt_mismatch = []

    for i, row in enumerate(req.itertuples(index=False)):
        rid = str(row.rsid)

        if rid == "." or rid not in rsid_to_idx:
            missing_rsids.append(rid)
            continue

        idx = rsid_to_idx[rid]

        if validate_ref_alt:
            vrow = vcf_df.iloc[idx]
            if (str(vrow.ref) != str(row.ref)) or (str(vrow.alt) != str(row.alt)):
                refalt_mismatch.append(
                    {
                        "rsid": rid,
                        "required_ref": str(row.ref),
                        "required_alt": str(row.alt),
                        "vcf_ref": str(vrow.ref),
                        "vcf_alt": str(vrow.alt),
                    }
                )
                continue

        out[i, :, :] = gt[idx, :, :]
        matched += 1

    # (n_req, n_samples, 2) -> (n_samples, 2, n_req)
    out = out.transpose(1, 2, 0)
    x = torch.tensor(out, dtype=torch.float32)

    missing_fraction = 0.0 if n_req == 0 else (len(missing_rsids) / n_req)
    match_fraction = 0.0 if n_req == 0 else (matched / n_req)

    stats = {
        "n_required": int(n_req),
        "n_matched": int(matched),
        "n_missing": int(len(missing_rsids)),
        "missing_fraction": float(missing_fraction),
        "match_fraction": float(match_fraction),
        "missing_rsids_first10": missing_rsids[:10],
        "n_refalt_mismatch": int(len(refalt_mismatch)),
        "refalt_mismatch_first5": refalt_mismatch[:5],
        "n_samples": int(n_samples),
        "tensor_shape": tuple(x.shape),
    }

    _log(f"[input] Output tensor shape = {tuple(x.shape)}", verbose=verbose)
    _log(f"[input] Matched SNPs = {matched:,} / {n_req:,} ({100.0 * match_fraction:.2f}%)", verbose=verbose)
    _log(f"[input] Missing SNPs = {len(missing_rsids):,} / {n_req:,} ({100.0 * missing_fraction:.2f}%)", verbose=verbose)
    _log(f"[input] REF/ALT mismatches = {len(refalt_mismatch):,}", verbose=verbose)

    if missing_rsids:
        _log(f"[input] First few missing rsIDs: {missing_rsids[:10]}", verbose=verbose)

    if refalt_mismatch:
        _log(f"[input] First few REF/ALT mismatches: {refalt_mismatch[:3]}", verbose=verbose)

    return x, samples, stats

# Main chromosome-level runner
def run_bundle_on_chrom_vcf(
    bundle_dir: str,
    vcf_path: str,
    chrom: int,
    device: str | None = None,
    fill_missing_value: int = 0,
    validate_ref_alt: bool = False,
    verbose: bool = True,
):
    """
    Runs all exported models for one chromosome on one chromosome VCF.

    Returns
    -------
    results : dict
    results_cp : dict
    stats_df : pd.DataFrame
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _section(f"Running bundle on chromosome {chrom}", verbose=verbose)
    _log(f"[run] bundle_dir = {bundle_dir}", verbose=verbose)
    _log(f"[run] vcf_path   = {vcf_path}", verbose=verbose)
    _log(f"[run] device     = {device}", verbose=verbose)

    manifest = load_bundle_manifest(bundle_dir, verbose=verbose)
    entries = [m for m in manifest["models"] if int(m["chrom"]) == int(chrom)]
    entries = sorted(entries, key=lambda x: (999 if x["subset_idx"] is None else x["subset_idx"]))

    if not entries:
        raise ValueError(f"No bundle entries found for chr{chrom}")

    _log(f"[run] Found {len(entries)} exported model entry(ies) for chr{chrom}", verbose=verbose)

    results = defaultdict(lambda: defaultdict(dict))
    results_cp = defaultdict(lambda: defaultdict(dict))
    all_stats = []

    per_sample_h1 = None
    per_sample_h2 = None
    per_sample_cp_h1 = None
    per_sample_cp_h2 = None
    samples_ref = None

    for entry_idx, entry in enumerate(entries, start=1):
        _subsection(
            f"Processing bundle entry {entry_idx}/{len(entries)} "
            f"(subset_idx={entry['subset_idx']}, model_file={entry['model_file']})",
            verbose=verbose,
        )

        snp_manifest_path = os.path.join(bundle_dir, entry["snp_manifest_file"])
        x, samples, stats = build_model_input_from_vcf(
            vcf_path=vcf_path,
            required_snp_manifest_path=snp_manifest_path,
            fill_missing_value=fill_missing_value,
            validate_ref_alt=validate_ref_alt,
            verbose=verbose,
        )

        stats["chrom"] = chrom
        stats["subset_idx"] = entry["subset_idx"]
        stats["model_file"] = entry["model_file"]
        all_stats.append(stats)

        if samples_ref is None:
            samples_ref = samples
            per_sample_h1 = {sid: [] for sid in samples}
            per_sample_h2 = {sid: [] for sid in samples}
            per_sample_cp_h1 = {sid: [] for sid in samples}
            per_sample_cp_h2 = {sid: [] for sid in samples}
            _log(f"[run] Initialized per-sample output buffers for {len(samples)} samples", verbose=verbose)

        model = load_exported_model(bundle_dir, entry["model_file"], device=device, verbose=verbose)

        _log(f"[infer] Beginning inference for {len(samples)} samples", verbose=verbose)
        t0 = time.time()

        with torch.no_grad():
            for s_idx, sample_id in enumerate(samples):
                xb = x[s_idx].to(device)  # (2, n_snps)

                if verbose and (s_idx < 3 or (s_idx + 1) % 25 == 0 or (s_idx + 1) == len(samples)):
                    _log(
                        f"[infer] Sample {s_idx + 1}/{len(samples)}: {sample_id} | input shape = {tuple(xb.shape)}",
                        verbose=verbose,
                    )

                main_coords, _, cp_logits, _, _ = model(xb)

                mc = main_coords.detach().cpu().numpy()
                cp_np = None if cp_logits is None else cp_logits.detach().cpu().numpy()

                per_sample_h1[sample_id].append(mc[0])
                per_sample_h2[sample_id].append(mc[1])

                if cp_np is not None:
                    per_sample_cp_h1[sample_id].append(cp_np[0])
                    per_sample_cp_h2[sample_id].append(cp_np[1])

                if verbose and s_idx == 0:
                    _log(f"[infer] Example output main_coords shape = {mc.shape}", verbose=verbose)
                    if cp_np is not None:
                        _log(f"[infer] Example output cp_logits shape = {cp_np.shape}", verbose=verbose)
                    else:
                        _log("[infer] cp_logits is None for this model", verbose=verbose)

        dt = time.time() - t0
        _log(f"[infer] Finished entry in {dt:.2f}s", verbose=verbose)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
            _log("[infer] Emptied CUDA cache after model", verbose=verbose)

    chr_key = f"chr{chrom}"
    _subsection("Assembling final chromosome outputs", verbose=verbose)

    for sample_id in samples_ref:
        results[sample_id][chr_key]["h1"] = np.concatenate(per_sample_h1[sample_id], axis=0)
        results[sample_id][chr_key]["h2"] = np.concatenate(per_sample_h2[sample_id], axis=0)

        if per_sample_cp_h1[sample_id]:
            results_cp[sample_id][chr_key]["h1"] = np.concatenate(per_sample_cp_h1[sample_id], axis=0)
            results_cp[sample_id][chr_key]["h2"] = np.concatenate(per_sample_cp_h2[sample_id], axis=0)

    first_sample = samples_ref[0]
    _log(
        f"[done] Example final output for {first_sample}: "
        f"h1 shape = {results[first_sample][chr_key]['h1'].shape}, "
        f"h2 shape = {results[first_sample][chr_key]['h2'].shape}",
        verbose=verbose,
    )

    if first_sample in results_cp and chr_key in results_cp[first_sample]:
        _log(
            f"[done] Example breakpoint output for {first_sample}: "
            f"h1 shape = {results_cp[first_sample][chr_key]['h1'].shape}, "
            f"h2 shape = {results_cp[first_sample][chr_key]['h2'].shape}",
            verbose=verbose,
        )

    stats_df = pd.DataFrame(all_stats)

    _log("[done] Chromosome run complete", verbose=verbose)
    _log(f"[done] stats_df shape = {stats_df.shape}", verbose=verbose)

    if not stats_df.empty:
        cols = [c for c in ["chrom", "subset_idx", "n_required", "n_matched", "n_missing", "match_fraction"] if c in stats_df.columns]
        _log("[done] Summary table:", verbose=verbose)
        print(stats_df[cols], flush=True)

    return results, results_cp, stats_df

# Directory-level runner
def run_bundle_on_vcf_dir(
    bundle_dir: str,
    vcf_dir: str,
    chroms=None,
    device: str | None = None,
    fill_missing_value: int = 0,
    validate_ref_alt: bool = False,
    verbose: bool = True,
):
    """
    Looks for chromosome VCFs in collaborator-friendly layouts.

    Recommended layout:
      vcf_dir/
        chr1.vcf.gz
        chr2.vcf.gz
        ...
        chr22.vcf.gz

    Also accepts:
      vcf_dir/chr1/chr1.vcf.gz
      vcf_dir/chr1/final.for_model.chr1.vcf.gz
      vcf_dir/chr1/*.vcf.gz
    """
    _section("Running bundle on VCF directory", verbose=verbose)
    _log(f"[run-dir] bundle_dir = {bundle_dir}", verbose=verbose)
    _log(f"[run-dir] vcf_dir    = {vcf_dir}", verbose=verbose)

    manifest = load_bundle_manifest(bundle_dir, verbose=verbose)

    if chroms is None:
        chroms = sorted({int(m["chrom"]) for m in manifest["models"]}, reverse=True)

    _log(f"[run-dir] Chromosomes requested: {chroms}", verbose=verbose)

    results = defaultdict(lambda: defaultdict(dict))
    results_cp = defaultdict(lambda: defaultdict(dict))
    stats_frames = []

    for ch in chroms:
        _subsection(f"Resolving VCF for chr{ch}", verbose=verbose)
        vcf_path = resolve_chrom_vcf_path(vcf_dir, ch, verbose=verbose)

        if vcf_path is None:
            _log(f"[run-dir] No VCF found for chr{ch}; skipping", verbose=verbose)
            continue

        chr_results, chr_results_cp, chr_stats = run_bundle_on_chrom_vcf(
            bundle_dir=bundle_dir,
            vcf_path=vcf_path,
            chrom=ch,
            device=device,
            fill_missing_value=fill_missing_value,
            validate_ref_alt=validate_ref_alt,
            verbose=verbose,
        )

        for sample_id, by_chr in chr_results.items():
            for chr_key, payload in by_chr.items():
                results[sample_id][chr_key] = payload

        for sample_id, by_chr in chr_results_cp.items():
            for chr_key, payload in by_chr.items():
                results_cp[sample_id][chr_key] = payload

        stats_frames.append(chr_stats)

    stats_df = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()

    _section("Finished directory-level run", verbose=verbose)
    _log(f"[run-dir] Number of samples with results = {len(results)}", verbose=verbose)
    _log(f"[run-dir] stats_df shape = {stats_df.shape}", verbose=verbose)

    return results, results_cp, stats_df


def to_plain_dict(obj):
    """
    Recursively convert defaultdict / nested dict-like structures into plain dicts,
    so they can be serialized safely with pickle/json.
    """
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_plain_dict(v) for v in obj)
    else:
        return obj


def save_inference_outputs(
    outdir: str | Path,
    *,
    results,
    results_cp,
    stats_df,
    metadata: dict | None = None,
    verbose: bool = True,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_path = outdir / "results.pkl.gz"
    results_cp_path = outdir / "results_cp.pkl.gz"
    stats_path = outdir / "stats.tsv"
    metadata_path = outdir / "metadata.json"

    if verbose:
        print(f"[save] Writing outputs to: {outdir}", flush=True)

    results_plain = to_plain_dict(results)
    results_cp_plain = to_plain_dict(results_cp)

    with gzip.open(results_path, "wb") as f:
        pickle.dump(results_plain, f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(results_cp_path, "wb") as f:
        pickle.dump(results_cp_plain, f, protocol=pickle.HIGHEST_PROTOCOL)

    if stats_df is not None:
        stats_df.to_csv(stats_path, sep="\t", index=False)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_file": str(results_path.name),
        "results_cp_file": str(results_cp_path.name),
        "stats_file": str(stats_path.name),
    }
    if metadata:
        meta.update(metadata)

    with metadata_path.open("w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"[save] results    -> {results_path}", flush=True)
        print(f"[save] results_cp -> {results_cp_path}", flush=True)
        print(f"[save] stats      -> {stats_path}", flush=True)
        print(f"[save] metadata   -> {metadata_path}", flush=True)


def load_inference_outputs(outdir: str | Path, verbose: bool = True):
    outdir = Path(outdir)

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


def cli_main():
    ap = argparse.ArgumentParser(
        description="Run exported PCLAI bundle on one chromosome VCF or a directory of chromosome VCFs."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    ap_chrom = sub.add_parser("run-chrom", help="Run one chromosome VCF against one chromosome bundle entry set")
    ap_chrom.add_argument("--bundle-dir", required=True)
    ap_chrom.add_argument("--vcf-path", required=True)
    ap_chrom.add_argument("--chrom", required=True, type=int)
    ap_chrom.add_argument("--device", default=None, help='e.g. "cuda" or "cpu"')
    ap_chrom.add_argument("--fill-missing-value", type=int, default=0)
    ap_chrom.add_argument("--validate-ref-alt", action="store_true")
    ap_chrom.add_argument("--outdir", required=True)
    ap_chrom.add_argument("--quiet", action="store_true")
    ap_dir = sub.add_parser("run-dir", help="Run a directory of chromosome VCFs against the bundle")
    ap_dir.add_argument("--bundle-dir", required=True)
    ap_dir.add_argument("--vcf-dir", required=True)
    ap_dir.add_argument("--device", default=None, help='e.g. "cuda" or "cpu"')
    ap_dir.add_argument("--fill-missing-value", type=int, default=0)
    ap_dir.add_argument("--validate-ref-alt", action="store_true")
    ap_dir.add_argument("--outdir", required=True)
    ap_dir.add_argument(
        "--chroms",
        default=None,
        help='Optional comma-separated list, e.g. "1,2,21,22"',
    )
    ap_dir.add_argument("--quiet", action="store_true")
    ap_load = sub.add_parser("load", help="Load previously saved outputs and print a short summary")
    ap_load.add_argument("--outdir", required=True)

    args = ap.parse_args()
    verbose = not args.quiet if hasattr(args, "quiet") else True

    if args.command == "run-chrom":
        results, results_cp, stats_df = run_bundle_on_chrom_vcf(
            bundle_dir=args.bundle_dir,
            vcf_path=args.vcf_path,
            chrom=args.chrom,
            device=args.device,
            fill_missing_value=args.fill_missing_value,
            validate_ref_alt=args.validate_ref_alt,
            verbose=verbose,
        )

        save_inference_outputs(
            args.outdir,
            results=results,
            results_cp=results_cp,
            stats_df=stats_df,
            metadata={
                "mode": "run-chrom",
                "bundle_dir": args.bundle_dir,
                "vcf_path": args.vcf_path,
                "chrom": args.chrom,
                "device": args.device,
                "fill_missing_value": args.fill_missing_value,
                "validate_ref_alt": args.validate_ref_alt,
            },
            verbose=verbose,
        )

    elif args.command == "run-dir":
        chroms = None
        if args.chroms:
            chroms = [int(x) for x in args.chroms.split(",") if x.strip()]

        results, results_cp, stats_df = run_bundle_on_vcf_dir(
            bundle_dir=args.bundle_dir,
            vcf_dir=args.vcf_dir,
            chroms=chroms,
            device=args.device,
            fill_missing_value=args.fill_missing_value,
            validate_ref_alt=args.validate_ref_alt,
            verbose=verbose,
        )

        save_inference_outputs(
            args.outdir,
            results=results,
            results_cp=results_cp,
            stats_df=stats_df,
            metadata={
                "mode": "run-dir",
                "bundle_dir": args.bundle_dir,
                "vcf_dir": args.vcf_dir,
                "chroms": chroms,
                "device": args.device,
                "fill_missing_value": args.fill_missing_value,
                "validate_ref_alt": args.validate_ref_alt,
            },
            verbose=verbose,
        )

    elif args.command == "load":
        results, results_cp, stats_df, metadata = load_inference_outputs(args.outdir, verbose=True)

        print("\n=== METADATA ===", flush=True)
        print(json.dumps(metadata, indent=2), flush=True)

        print("\n=== STATS HEAD ===", flush=True)
        print(stats_df.head(), flush=True)

        if results:
            first_sample = next(iter(results))
            first_chr = next(iter(results[first_sample]))
            print("\n=== EXAMPLE RESULTS ENTRY ===", flush=True)
            print(f"sample={first_sample} chrom={first_chr}", flush=True)
            print(f"h1 shape = {results[first_sample][first_chr]['h1'].shape}", flush=True)
            print(f"h2 shape = {results[first_sample][first_chr]['h2'].shape}", flush=True)

        if results_cp:
            first_sample = next(iter(results_cp))
            first_chr = next(iter(results_cp[first_sample]))
            print("\n=== EXAMPLE RESULTS_CP ENTRY ===", flush=True)
            print(f"sample={first_sample} chrom={first_chr}", flush=True)
            print(f"h1 shape = {results_cp[first_sample][first_chr]['h1'].shape}", flush=True)
            print(f"h2 shape = {results_cp[first_sample][first_chr]['h2'].shape}", flush=True)


if __name__ == "__main__":
    cli_main()