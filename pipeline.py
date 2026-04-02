#!/usr/bin/env python3
"""
pipeline.py

Harmonization pipeline.

Uses:
  - bundle_dir/manifest.json
  - bundle_dir/snp_manifests/*.snps.tsv

as the source of truth for:
  - final SNP order
  - allele QC against the expected model SNPs

Also uses:
  - --reference-split-template

as the Beagle reference panel input.

Example:
  python3 ./inference/pipeline.py \
    --input-vcf /path/to/input.vcf.gz \
    --workdir /path/to/workdir \
    --bundle-dir /path/to/pclai_1kg_bundle \
    --reference-split-template "/path/to/reference.chr{chrom}.vcf.gz" \
    --impute-engine beagle \
    --beagle-jar /path/to/beagle.jar \
    --threads 16 \
    --reference-fasta /path/to/GRCh38.fa \
    --auto-normalize-on-qc-fail \
    --log-level DEBUG
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


LOG = logging.getLogger("lai-pipeline")

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    LOG.info("Logging initialized at level=%s", level.upper())


def shjoin(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_stdout: bool = False,
    capture_stderr: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    LOG.info("RUN: %s", shjoin(cmd))
    if cwd:
        LOG.info("  cwd: %s", cwd)

    stdout = subprocess.PIPE if capture_stdout else None
    stderr = subprocess.PIPE if capture_stderr else None

    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
    )
    LOG.info("  exit_code=%s", p.returncode)

    if p.stdout:
        LOG.debug("  STDOUT (first 2000 chars):\n%s", p.stdout[:2000])
    if p.stderr:
        LOG.debug("  STDERR (first 2000 chars):\n%s", p.stderr[:2000])

    if check and p.returncode != 0:
        msg = f"Command failed ({p.returncode}): {shjoin(cmd)}"
        if p.stderr:
            msg += f"\nSTDERR:\n{p.stderr}"
        raise RuntimeError(msg)
    return p


def popen_lines(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.Popen:
    LOG.info("POPEN: %s", shjoin(cmd))
    if cwd:
        LOG.info("  cwd: %s", cwd)

    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


def count_stream_lines(proc: subprocess.Popen, *, label: str) -> int:
    assert proc.stdout is not None
    n = 0
    for _ in proc.stdout:
        n += 1
        if n % 1_000_000 == 0:
            LOG.info("%s: counted %d lines so far...", label, n)
    return n

@dataclass
class ToolConfig:
    bcftools: str
    java: str
    beagle_jar: Optional[Path]
    minimac4: str
    threads: int


@dataclass
class Templates:
    reference_split_template: Optional[str]
    genetic_map_template: Optional[str]


@dataclass
class AlleleConcordanceStats:
    shared_pos: int
    exact_match: int
    inverted_ref_alt: int
    other_mismatch: int
    missing_in_model: int
    exact_match_pct: float
    inverted_pct: float
    other_mismatch_pct: float
    examples: List[str]


@dataclass
class ChromStats:
    chrom: str
    input_contig: str
    total_manifest_records: int
    qc_passed: bool
    allele_exact_match_pct: float
    allele_inverted: int
    allele_other_mismatch: int
    target_is_phased: bool
    phased_or_imputed_vcf: Optional[str]
    final_model_order_vcf: str

def load_bundle_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    LOG.info("Loaded bundle manifest: %s", manifest_path)
    LOG.info("Bundle format=%s models=%d", manifest.get("format"), len(manifest.get("models", [])))
    return manifest


def bundle_entries_for_chrom(bundle_dir: Path, chrom: str) -> List[dict]:
    manifest = load_bundle_manifest(bundle_dir)
    entries = [
        m for m in manifest["models"]
        if str(m["chrom"]) == str(chrom)
    ]
    entries = sorted(entries, key=lambda x: (999 if x["subset_idx"] is None else x["subset_idx"]))
    if not entries:
        raise RuntimeError(f"No bundle entries found for chr{chrom} in {bundle_dir}")
    return entries


def available_bundle_chroms(bundle_dir: Path) -> List[str]:
    manifest = load_bundle_manifest(bundle_dir)
    chroms = sorted({str(m["chrom"]) for m in manifest["models"]}, key=lambda x: int(x))
    return chroms


def combined_snp_manifest_for_chrom(bundle_dir: Path, chrom: str, target_contig: Optional[str] = None) -> pd.DataFrame:
    """
    Concatenate all per-subset SNP manifests for a chromosome in bundle order.
    """
    entries = bundle_entries_for_chrom(bundle_dir, chrom)
    dfs = []
    for e in entries:
        p = bundle_dir / e["snp_manifest_file"]
        if not p.exists():
            raise FileNotFoundError(f"Missing SNP manifest file: {p}")
        df = pd.read_csv(p, sep="\t", dtype={"chrom": str, "pos": int, "rsid": str, "ref": str, "alt": str})
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if target_contig is not None:
        out = out.copy()
        out["chrom"] = str(target_contig)

    LOG.info(
        "Combined SNP manifest chr%s: %d rows from %d subset file(s)",
        chrom, len(out), len(dfs)
    )
    return out

def get_vcf_contigs(cfg: ToolConfig, vcf_gz: Path) -> List[str]:
    p = run([cfg.bcftools, "view", "-h", str(vcf_gz)], capture_stdout=True)
    hdr = p.stdout or ""
    contigs: List[str] = []
    for line in hdr.splitlines():
        if line.startswith("##contig=<"):
            m = re.search(r"ID=([^,>]+)", line)
            if m:
                contigs.append(m.group(1))
    return contigs


def detect_canonical_chrom_mapping(contigs: List[str]) -> Dict[str, str]:
    contig_set = set(contigs)

    def choose(canonical: str) -> Optional[str]:
        if canonical == "MT":
            candidates = ["chrM", "MT", "chrMT", "M"]
        else:
            candidates = [f"chr{canonical}", canonical]
        for c in candidates:
            if c in contig_set:
                return c
        return None

    mapping: Dict[str, str] = {}
    for c in [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]:
        picked = choose(c)
        if picked:
            mapping[c] = picked
    return mapping


def ensure_index(cfg: ToolConfig, vcf_gz: Path, *, prefer: str = "tbi", force: bool = False) -> None:
    v_mtime = vcf_gz.stat().st_mtime if vcf_gz.exists() else 0.0
    tbi = Path(str(vcf_gz) + ".tbi")
    csi = Path(str(vcf_gz) + ".csi")

    idx_exists = tbi.exists() or csi.exists()
    idx_stale = False
    if tbi.exists() and tbi.stat().st_mtime < v_mtime:
        idx_stale = True
    if csi.exists() and csi.stat().st_mtime < v_mtime:
        idx_stale = True

    if not force and idx_exists and not idx_stale:
        return

    if idx_stale and not force:
        LOG.warning("Index older than VCF for %s -> rebuilding", vcf_gz)
        force = True

    cmd = [cfg.bcftools, "index"]
    if force:
        cmd.append("-f")
    if prefer.lower() == "tbi":
        cmd.append("-t")
    else:
        cmd.append("--csi")
    cmd.append(str(vcf_gz))
    run(cmd)


def bcftools_count_records(cfg: ToolConfig, vcf_gz: Path) -> int:
    LOG.info("Counting records (streaming): %s", vcf_gz)
    proc = popen_lines([cfg.bcftools, "view", "-H", str(vcf_gz)])
    n = count_stream_lines(proc, label=f"count({vcf_gz.name})")
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bcftools count failed rc={rc}\nSTDERR:\n{stderr}")
    return n


def is_vcf_phased(cfg: ToolConfig, vcf_gz: Path, contig: str, max_lines: int = 2000) -> bool:
    hdr = run([cfg.bcftools, "view", "-h", str(vcf_gz)], capture_stdout=True).stdout or ""
    header_says_phased = False
    for line in hdr.splitlines():
        if line.startswith("##FORMAT=<ID=GT"):
            if re.search(r"phased", line, flags=re.IGNORECASE):
                header_says_phased = True
            break

    cmd = [cfg.bcftools, "query", "-r", contig, "-f", "[%GT\t]\n", str(vcf_gz)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    pipes = 0
    slashes = 0
    n_lines = 0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            n_lines += 1
            for gt in line.rstrip("\n").split("\t"):
                if not gt or gt == ".":
                    continue
                if "|" in gt:
                    pipes += 1
                if "/" in gt:
                    slashes += 1
            if n_lines >= max_lines:
                break
            if slashes > 0 and pipes == 0:
                break
            if pipes > 0 and slashes > 0:
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    rc = proc.returncode
    if rc not in (0, -13, 141, -15):
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"Phasing check failed rc={rc}\nSTDERR:\n{stderr}")

    if slashes > 0 and pipes == 0:
        return False
    if pipes > 0 and slashes == 0:
        return True
    if pipes > 0 and slashes > 0:
        return False
    return header_says_phased


def contig_for_canonical_chrom(cfg: ToolConfig, vcf_gz: Path, chrom: str) -> str:
    contigs = get_vcf_contigs(cfg, vcf_gz)
    mapping = detect_canonical_chrom_mapping(contigs)
    if chrom in mapping:
        return mapping[chrom]
    if len(contigs) == 1:
        return contigs[0]
    raise RuntimeError(f"Could not find contig for canonical chr{chrom} in {vcf_gz}")


def rename_chrom_if_needed(cfg: ToolConfig, in_vcf: Path, old_contig: str, new_contig: str, out_vcf: Path) -> Path:
    if old_contig == new_contig:
        return in_vcf

    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    map_txt = out_vcf.parent / f"rename_{old_contig}_to_{new_contig}.txt"
    map_txt.write_text(f"{old_contig}\t{new_contig}\n")

    run([cfg.bcftools, "annotate", "--rename-chrs", str(map_txt), "-Oz", "-o", str(out_vcf), str(in_vcf)])
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf


def clean_snps_biallelic(cfg: ToolConfig, in_vcf: Path, out_vcf: Path) -> Path:
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    run([
        cfg.bcftools, "view",
        "-v", "snps",
        "-m2", "-M2",
        "-e", 'ALT="."',
        "-Oz", "-o", str(out_vcf),
        str(in_vcf),
    ])
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf


def normalize_vcf(cfg: ToolConfig, in_vcf: Path, fasta: Path, out_vcf: Path) -> Path:
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    run([
        cfg.bcftools, "norm",
        "-m", "-any",
        "-f", str(fasta),
        "--check-ref", "w",
        "-Oz", "-o", str(out_vcf),
        str(in_vcf),
    ])
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf


def extract_chrom_variant_vcf(cfg: ToolConfig, input_vcf: Path, contig: str, out_vcf: Path) -> None:
    cmd = [
        cfg.bcftools, "view",
        "-r", contig,
        "-v", "snps",
        "-m2", "-M2",
        "-e", 'ALT="."',
        "-Oz", "-o", str(out_vcf),
        str(input_vcf),
    ]
    run(cmd)
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)


def read_samples_from_vcf_header(cfg: ToolConfig, vcf: Path) -> List[str]:
    hdr = run([cfg.bcftools, "view", "-h", str(vcf)], capture_stdout=True).stdout or ""
    for line in reversed(hdr.splitlines()):
        if line.startswith("#CHROM"):
            parts = line.rstrip("\n").split("\t")
            return parts[9:] if len(parts) > 9 else []
    return []


def _iter_pos_ref_alt(cfg: ToolConfig, vcf: Path) -> Iterable[Tuple[int, str, str]]:
    proc = popen_lines([cfg.bcftools, "query", "-f", "%POS\t%REF\t%ALT\n", str(vcf)])
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        pos_s, ref, alt = line.split("\t")
        yield int(pos_s), ref, alt
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bcftools query failed on {vcf} rc={rc}\nSTDERR:\n{stderr}")


# ---------------------------------------------------------------------
# QC against SNP manifest
# ---------------------------------------------------------------------

def build_manifest_pos_to_alleles(snp_df: pd.DataFrame) -> Dict[int, set]:
    m: Dict[int, set] = defaultdict(set)
    for row in snp_df.itertuples(index=False):
        m[int(row.pos)].add((str(row.ref), str(row.alt)))
    return m


def allele_concordance_check_streaming_vs_manifest(
    cfg: ToolConfig,
    *,
    chrom: str,
    target_vcf: Path,
    snp_df: pd.DataFrame,
    max_examples: int = 12,
) -> AlleleConcordanceStats:
    """
    For each TARGET variant POS, check whether its (REF,ALT) appears among any
    SNP-manifest alleles at that POS.
    """
    manifest_map = build_manifest_pos_to_alleles(snp_df)

    shared = 0
    exact = 0
    inv = 0
    other = 0
    missing_in_model = 0
    examples: List[str] = []

    for tpos, tref, talt in _iter_pos_ref_alt(cfg, target_vcf):
        shared += 1
        model_pairs = manifest_map.get(tpos)

        if not model_pairs:
            missing_in_model += 1
            if len(examples) < max_examples:
                examples.append(f"POS={tpos} target={tref}>{talt} (NO_MANIFEST_AT_POS)")
            continue

        if (tref, talt) in model_pairs:
            exact += 1
        elif (talt, tref) in model_pairs:
            inv += 1
            if len(examples) < max_examples:
                examples.append(f"POS={tpos} target={tref}>{talt} (INVERTED_vs_manifest)")
        else:
            other += 1
            if len(examples) < max_examples:
                sample = ", ".join([f"{r}>{a}" for (r, a) in list(model_pairs)[:3]])
                examples.append(f"POS={tpos} target={tref}>{talt} manifest_candidates=[{sample}] (MISMATCH)")

    exact_pct = (100.0 * exact / shared) if shared else 0.0
    inv_pct = (100.0 * inv / shared) if shared else 0.0
    other_pct = (100.0 * other / shared) if shared else 0.0

    LOG.info("Allele concordance chr%s over TARGET records:", chrom)
    LOG.info("  target_records_checked = %d", shared)
    LOG.info("  exact_match            = %d (%.6f%%)", exact, exact_pct)
    LOG.info("  inverted               = %d (%.6f%%)", inv, inv_pct)
    LOG.info("  other_mismatch         = %d (%.6f%%)", other, other_pct)
    LOG.info("  missing_in_manifest_at_pos = %d", missing_in_model)

    if examples:
        LOG.warning("chr%s mismatch examples (first %d):", chrom, len(examples))
        for e in examples:
            LOG.warning("  %s", e)

    return AlleleConcordanceStats(
        shared_pos=shared,
        exact_match=exact,
        inverted_ref_alt=inv,
        other_mismatch=other,
        missing_in_model=missing_in_model,
        exact_match_pct=exact_pct,
        inverted_pct=inv_pct,
        other_mismatch_pct=other_pct,
        examples=examples,
    )

def run_beagle_phasing(cfg: ToolConfig, in_vcf: Path, out_prefix: Path, genetic_map: Optional[Path]) -> Path:
    if cfg.beagle_jar is None:
        raise ValueError("beagle_jar must be provided to phase unphased targets.")

    cmd = [
        cfg.java, "-jar", "-Xmx200g", str(cfg.beagle_jar),
        f"gt={in_vcf}",
        f"out={out_prefix}",
        f"nthreads={cfg.threads}",
    ]
    if genetic_map:
        cmd.append(f"map={genetic_map}")

    run(cmd)

    phased_vcf = Path(str(out_prefix) + ".vcf.gz")
    if not phased_vcf.exists():
        raise RuntimeError(f"Beagle phasing output not found: {phased_vcf}")
    ensure_index(cfg, phased_vcf, prefer="tbi", force=True)
    return phased_vcf


def run_beagle_imputation(cfg: ToolConfig, gt_vcf: Path, ref_vcf: Path, out_prefix: Path, genetic_map: Optional[Path]) -> Path:
    if cfg.beagle_jar is None:
        raise ValueError("beagle_jar must be provided for Beagle imputation.")

    cmd = [
        cfg.java, "-jar", "-Xmx200g", str(cfg.beagle_jar),
        f"gt={gt_vcf}",
        f"ref={ref_vcf}",
        f"out={out_prefix}",
        "impute=true",
        f"nthreads={cfg.threads}",
    ]
    if genetic_map:
        cmd.append(f"map={genetic_map}")

    run(cmd)

    out_vcf = Path(str(out_prefix) + ".vcf.gz")
    if not out_vcf.exists():
        raise RuntimeError(f"Beagle imputation output not found: {out_vcf}")
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf

def _iter_vcf_data_lines(cfg: ToolConfig, vcf: Path) -> Iterable[str]:
    proc = popen_lines([cfg.bcftools, "view", "-H", str(vcf)])
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line:
            yield line
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bcftools view -H failed on {vcf} rc={rc}\nSTDERR:\n{stderr}")

def build_key_to_tail_list(cfg: ToolConfig, vcf: Path) -> Dict[Tuple[int, str, str], List[List[str]]]:
    m: Dict[Tuple[int, str, str], List[List[str]]] = defaultdict(list)
    for line in _iter_vcf_data_lines(cfg, vcf):
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        pos = int(cols[1])
        ref = cols[3]
        alt = cols[4]
        if "," in alt:
            continue
        tail = cols[5:]
        m[(pos, ref, alt)].append(tail)
    return m

def write_final_vcf_in_manifest_order(
    cfg: ToolConfig,
    *,
    chrom: str,
    snp_df: pd.DataFrame,
    header_source_vcf: Path,
    beagle_fill_vcf: Optional[Path],
    target_fill_vcf: Path,
    out_vcf_gz: Path,
) -> None:
    """
    Emit final VCF in exact SNP-manifest row order.
    For each manifest record (POS,REF,ALT), fill FORMAT+SAMPLES from:
      - beagle_fill_vcf (preferred) if available and contains exact allele
      - else target_fill_vcf if contains exact allele
      - else missing (./.)
    """
    samples = read_samples_from_vcf_header(cfg, header_source_vcf)
    n_samples = len(samples)

    LOG.info("FINAL assembly chr%s:", chrom)
    LOG.info("  manifest_rows     = %d", len(snp_df))
    LOG.info("  header_source     = %s", header_source_vcf)
    LOG.info("  beagle_fill_vcf   = %s", str(beagle_fill_vcf) if beagle_fill_vcf else "(none)")
    LOG.info("  target_fill_vcf   = %s", target_fill_vcf)
    LOG.info("  n_samples         = %d", n_samples)

    beagle_map = build_key_to_tail_list(cfg, beagle_fill_vcf) if beagle_fill_vcf else {}
    target_map = build_key_to_tail_list(cfg, target_fill_vcf)

    tmp_vcf = out_vcf_gz.with_suffix("")
    if tmp_vcf.suffix != ".vcf":
        tmp_vcf = tmp_vcf.with_suffix(".vcf")

    hdr = run([cfg.bcftools, "view", "-h", str(header_source_vcf)], capture_stdout=True).stdout or ""

    with tmp_vcf.open("w") as fout:
        fout.write(hdr)

        missing_count = 0
        beagle_used = 0
        target_used = 0
        total_manifest = 0

        for row in snp_df.itertuples(index=False):
            total_manifest += 1
            chrom_s = str(row.chrom)
            pos = int(row.pos)
            vid = str(row.rsid)
            ref = str(row.ref)
            alt = str(row.alt)

            key = (pos, ref, alt)

            tail: Optional[List[str]] = None
            if key in beagle_map and beagle_map[key]:
                tail = beagle_map[key].pop(0)
                beagle_used += 1
            elif key in target_map and target_map[key]:
                tail = target_map[key].pop(0)
                target_used += 1
            else:
                missing_count += 1
                sample_cols = ["./."] * n_samples
                tail = [".", "PASS", ".", "GT"] + sample_cols

            out_cols = [chrom_s, str(pos), vid, ref, alt] + tail
            fout.write("\t".join(out_cols) + "\n")

    LOG.info("FINAL assembly chr%s done:", chrom)
    LOG.info("  total_manifest_records = %d", total_manifest)
    LOG.info("  filled_from_beagle     = %d", beagle_used)
    LOG.info("  filled_from_target     = %d", target_used)
    LOG.info("  missing_unfilled       = %d", missing_count)

    out_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    run([cfg.bcftools, "view", "-Oz", "-o", str(out_vcf_gz), str(tmp_vcf)])
    ensure_index(cfg, out_vcf_gz, prefer="tbi", force=True)

    try:
        tmp_vcf.unlink()
    except Exception:
        pass

class LAIPipeline:
    def __init__(
        self,
        cfg: ToolConfig,
        templates: Templates,
        bundle_dir: Path,
        workdir: Path,
        *,
        impute_engine: str,
        qc_strict: bool,
        min_exact_match_pct: float,
        require_zero_inversions: bool,
        require_zero_other_mismatch: bool,
        reference_fasta: Optional[Path],
        auto_normalize_on_qc_fail: bool,
        split_beagle_multiallelics: bool = True,
    ):
        self.cfg = cfg
        self.templates = templates
        self.bundle_dir = bundle_dir
        self.workdir = workdir
        self.impute_engine = impute_engine

        self.qc_strict = qc_strict
        self.min_exact_match_pct = min_exact_match_pct
        self.require_zero_inversions = require_zero_inversions
        self.require_zero_other_mismatch = require_zero_other_mismatch

        self.reference_fasta = reference_fasta
        self.auto_normalize_on_qc_fail = auto_normalize_on_qc_fail
        self.split_beagle_multiallelics = split_beagle_multiallelics

    def reference_split_vcf_for(self, chrom: str) -> Path:
        if not self.templates.reference_split_template:
            raise RuntimeError("--reference-split-template is required for imputation.")
        return Path(self.templates.reference_split_template.format(chrom=chrom))

    def map_for(self, chrom: str) -> Optional[Path]:
        if not self.templates.genetic_map_template:
            return None
        return Path(self.templates.genetic_map_template.format(chrom=chrom))

    def _prepare_ref_for_target_contig(self, chrom: str, target_contig: str, chrom_dir: Path) -> Path:
        ref_raw = self.reference_split_vcf_for(chrom)
        if not ref_raw.exists():
            raise RuntimeError(f"Missing reference VCF for chr{chrom}: {ref_raw}")

        ref_contig = contig_for_canonical_chrom(self.cfg, ref_raw, chrom)
        ref_renamed = rename_chrom_if_needed(
            self.cfg,
            ref_raw,
            old_contig=ref_contig,
            new_contig=target_contig,
            out_vcf=chrom_dir / f"reference.chr{chrom}.renamed.vcf.gz",
        )
        ensure_index(self.cfg, ref_renamed, prefer="tbi", force=True)
        return ref_renamed

    def _maybe_norm_target(self, chrom: str, target_chr_vcf: Path, chrom_dir: Path, tag: str) -> Path:
        if self.reference_fasta is None:
            return target_chr_vcf

        norm = normalize_vcf(
            self.cfg,
            target_chr_vcf,
            self.reference_fasta,
            chrom_dir / f"target.chr{chrom}.{tag}.norm.vcf.gz",
        )
        clean = clean_snps_biallelic(
            self.cfg,
            norm,
            chrom_dir / f"target.chr{chrom}.{tag}.clean.vcf.gz",
        )
        return clean

    def _split_multiallelic_if_needed(self, vcf: Path, chrom: str, chrom_dir: Path, tag: str) -> Path:
        if not self.split_beagle_multiallelics:
            return vcf
        out = chrom_dir / f"{tag}.split_biallelic.chr{chrom}.vcf.gz"
        run([self.cfg.bcftools, "norm", "-m", "-any", "-Oz", "-o", str(out), str(vcf)])
        ensure_index(self.cfg, out, prefer="tbi", force=True)
        return out

    def _qc_gate(self, chrom: str, allele_stats: AlleleConcordanceStats) -> bool:
        failures: List[str] = []
        if allele_stats.exact_match_pct < self.min_exact_match_pct:
            failures.append(
                f"allele_exact_match_pct={allele_stats.exact_match_pct:.6f}% < min_exact_match_pct={self.min_exact_match_pct:.6f}%"
            )
        if self.require_zero_inversions and allele_stats.inverted_ref_alt > 0:
            failures.append(f"inversions={allele_stats.inverted_ref_alt}")
        if self.require_zero_other_mismatch and allele_stats.other_mismatch > 0:
            failures.append(f"other_mismatch={allele_stats.other_mismatch}")

        if failures:
            msg = "QC FAILED for chr{}:\n  - ".format(chrom) + "\n  - ".join(failures)
            LOG.error(msg)
            if self.qc_strict:
                raise RuntimeError(msg)
            return False

        LOG.info("QC PASSED for chr%s", chrom)
        return True

    def run(self, input_vcf: Path) -> List[ChromStats]:
        LOG.info("=== PIPELINE START ===")
        LOG.info("Input VCF: %s", input_vcf)
        LOG.info("Bundle dir: %s", self.bundle_dir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        ensure_index(self.cfg, input_vcf, prefer="tbi", force=False)
        contigs = get_vcf_contigs(self.cfg, input_vcf)
        chrom_to_target_contig = detect_canonical_chrom_mapping(contigs)
        if not chrom_to_target_contig:
            raise RuntimeError("Could not map any canonical chromosomes (1-22,X,Y,MT) in input VCF header.")

        bundle_chroms = set(available_bundle_chroms(self.bundle_dir))
        chroms_to_process: List[str] = []
        for chrom in sorted(bundle_chroms, key=int):
            if chrom in chrom_to_target_contig:
                chroms_to_process.append(chrom)
            else:
                LOG.warning("Bundle has chr%s but input VCF does not; skipping", chrom)

        stats: List[ChromStats] = []

        for chrom in chroms_to_process:
            target_contig = chrom_to_target_contig[chrom]
            chrom_dir = self.workdir / f"chr{chrom}"
            chrom_dir.mkdir(parents=True, exist_ok=True)

            LOG.info("------------------------------------------------------------------")
            LOG.info("Processing chr%s (target_contig=%s)", chrom, target_contig)

            # 1) Extract target chromosome
            target_chr_vcf = chrom_dir / f"target.chr{chrom}.vcf.gz"
            extract_chrom_variant_vcf(self.cfg, input_vcf, target_contig, target_chr_vcf)

            # 2) Load bundle SNP manifest for this chromosome, rewritten to target contig naming
            snp_df = combined_snp_manifest_for_chrom(self.bundle_dir, chrom, target_contig=target_contig)

            # 3) Prepare reference VCF for Beagle
            ref_vcf = self._prepare_ref_for_target_contig(chrom, target_contig, chrom_dir)

            # 4) Optional normalization target if QC fails
            target_pre = target_chr_vcf

            # 5) QC pass 1 vs SNP manifest
            allele_stats = allele_concordance_check_streaming_vs_manifest(
                self.cfg,
                chrom=chrom,
                target_vcf=target_pre,
                snp_df=snp_df,
            )
            qc_passed = self._qc_gate(chrom, allele_stats)

            if (not qc_passed) and self.reference_fasta is not None and self.auto_normalize_on_qc_fail:
                LOG.warning("QC failed for chr%s; retrying QC after target normalization vs FASTA.", chrom)
                target_pre = self._maybe_norm_target(chrom, target_chr_vcf, chrom_dir, tag="QC_RETRY")
                allele_stats = allele_concordance_check_streaming_vs_manifest(
                    self.cfg,
                    chrom=chrom,
                    target_vcf=target_pre,
                    snp_df=snp_df,
                )
                qc_passed = self._qc_gate(chrom, allele_stats)

            # 6) Phase / impute
            target_is_phased = is_vcf_phased(self.cfg, target_pre, target_contig)

            phased_or_imputed: Optional[Path] = None
            if self.impute_engine == "beagle":
                beagle_prefix = chrom_dir / f"target.imputed.by_beagle.chr{chrom}"
                imputed_vcf = run_beagle_imputation(
                    self.cfg,
                    gt_vcf=target_pre,
                    ref_vcf=ref_vcf,
                    out_prefix=beagle_prefix,
                    genetic_map=self.map_for(chrom),
                )
                imputed_vcf = self._split_multiallelic_if_needed(imputed_vcf, chrom, chrom_dir, tag="beagle_out")
                phased_or_imputed = imputed_vcf

            elif self.impute_engine == "none":
                if target_is_phased:
                    phased_or_imputed = target_pre
                else:
                    phased_prefix = chrom_dir / f"target.phased.by_beagle.chr{chrom}"
                    phased_or_imputed = run_beagle_phasing(self.cfg, target_pre, phased_prefix, self.map_for(chrom))
                    phased_or_imputed = self._split_multiallelic_if_needed(
                        phased_or_imputed, chrom, chrom_dir, tag="beagle_phased"
                    )

            elif self.impute_engine == "minimac4":
                raise RuntimeError("minimac4 path not implemented in this bundle-aware version.")

            else:
                raise RuntimeError(f"Unknown impute engine: {self.impute_engine}")

            # 7) Emit final VCF in exact bundle SNP-manifest order
            final_vcf = chrom_dir / f"final.for_model.chr{chrom}.vcf.gz"
            write_final_vcf_in_manifest_order(
                self.cfg,
                chrom=chrom,
                snp_df=snp_df,
                header_source_vcf=phased_or_imputed if phased_or_imputed else target_pre,
                beagle_fill_vcf=phased_or_imputed if phased_or_imputed else None,
                target_fill_vcf=target_pre,
                out_vcf_gz=final_vcf,
            )

            final_n = bcftools_count_records(self.cfg, final_vcf)
            manifest_n = len(snp_df)
            if manifest_n != final_n:
                LOG.error(
                    "chr%s: FINAL != manifest record count (should not happen). manifest=%d final=%d",
                    chrom, manifest_n, final_n
                )
            else:
                LOG.info("chr%s: FINAL matches manifest record count exactly: %d", chrom, final_n)

            stats.append(
                ChromStats(
                    chrom=chrom,
                    input_contig=target_contig,
                    total_manifest_records=manifest_n,
                    qc_passed=qc_passed,
                    allele_exact_match_pct=allele_stats.exact_match_pct,
                    allele_inverted=allele_stats.inverted_ref_alt,
                    allele_other_mismatch=allele_stats.other_mismatch,
                    target_is_phased=target_is_phased,
                    phased_or_imputed_vcf=str(phased_or_imputed) if phased_or_imputed else None,
                    final_model_order_vcf=str(final_vcf),
                )
            )

            LOG.info("chr%s DONE. Final (manifest-order) VCF: %s", chrom, final_vcf)

        LOG.info("=== PIPELINE END ===")
        return stats

def main() -> int:
    ap = argparse.ArgumentParser(
        description="LAI pipeline: per-chrom extraction, SNP-manifest harmonization, (optional) normalization, Beagle imputation, and final emission in exact bundle SNP order."
    )

    ap.add_argument("--input-vcf", required=True, type=Path)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--bundle-dir", required=True, type=Path)

    ap.add_argument("--reference-split-template", default=None)
    ap.add_argument("--impute-engine", choices=["beagle", "none", "minimac4"], default="beagle")
    ap.add_argument("--genetic-map-template", default=None)

    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bcftools", default="bcftools")
    ap.add_argument("--java", default="java")
    ap.add_argument("--beagle-jar", type=Path, default=None)
    ap.add_argument("--minimac4", default="minimac4")

    ap.add_argument("--qc-strict", action="store_true")
    ap.add_argument("--min-exact-match-pct", type=float, default=99.999)
    ap.add_argument("--allow-inversions", action="store_true")
    ap.add_argument("--allow-other-mismatch", action="store_true")

    ap.add_argument("--reference-fasta", type=Path, default=None)
    ap.add_argument("--auto-normalize-on-qc-fail", action="store_true")
    ap.add_argument("--no-split-beagle-multiallelics", action="store_true")

    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()
    setup_logging(args.log_level)

    templates = Templates(
        reference_split_template=args.reference_split_template,
        genetic_map_template=args.genetic_map_template,
    )
    cfg = ToolConfig(
        bcftools=args.bcftools,
        java=args.java,
        beagle_jar=args.beagle_jar,
        minimac4=args.minimac4,
        threads=args.threads,
    )

    pipe = LAIPipeline(
        cfg,
        templates,
        args.bundle_dir,
        args.workdir,
        impute_engine=args.impute_engine,
        qc_strict=bool(args.qc_strict),
        min_exact_match_pct=args.min_exact_match_pct,
        require_zero_inversions=not args.allow_inversions,
        require_zero_other_mismatch=not args.allow_other_mismatch,
        reference_fasta=args.reference_fasta,
        auto_normalize_on_qc_fail=args.auto_normalize_on_qc_fail,
        split_beagle_multiallelics=not args.no_split_beagle_multiallelics,
    )

    stats = pipe.run(args.input_vcf)

    print("\n=== SUMMARY (compact) ===")
    for s in stats:
        print(
            f"chr{s.chrom} qc={'PASS' if s.qc_passed else 'FAIL'} "
            f"exact={s.allele_exact_match_pct:.6f}% "
            f"inv={s.allele_inverted} other={s.allele_other_mismatch} "
            f"manifest_records={s.total_manifest_records} final={s.final_model_order_vcf}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())