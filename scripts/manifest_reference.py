#!/usr/bin/env python3
"""Reference data manifest + hash registry (Tier 3.3a).

Walks a TARDIS reference directory and writes MANIFEST.json containing
SHA256, size, mtime for every file plus a few derived schema attributes
(line_list row count, n_macro_levels, n_shells). Use to:

  * pin reference version per production run (write next to outputs)
  * detect drift between runs claiming the same reference dir
  * detect partial regeneration (one file rebuilt, others stale)

Usage:
    manifest_reference.py <ref_dir> [--out MANIFEST.json] [--compare other.json]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_record(path: str, rel: str) -> dict:
    st = os.stat(path)
    return {
        "path": rel,
        "size": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_of(path),
    }


def schema_attrs(ref: str) -> dict:
    """Cheap derived attrs without loading full arrays."""
    out: dict = {}
    try:
        cfg_p = os.path.join(ref, "config.json")
        if os.path.isfile(cfg_p):
            with open(cfg_p) as f:
                out["config"] = json.load(f)
    except Exception as e:
        out["config_error"] = str(e)
    try:
        ll = os.path.join(ref, "line_list.csv")
        if os.path.isfile(ll):
            out["n_lines_actual"] = sum(1 for _ in open(ll)) - 1
    except Exception as e:
        out["n_lines_error"] = str(e)
    try:
        mar = os.path.join(ref, "macro_atom_references.csv")
        if os.path.isfile(mar):
            out["n_macro_levels_actual"] = sum(1 for _ in open(mar)) - 1
    except Exception as e:
        out["n_macro_levels_error"] = str(e)
    return out


def build_manifest(ref: str) -> dict:
    files = []
    for root, _, names in os.walk(ref):
        for name in sorted(names):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, ref)
            try:
                files.append(file_record(full, rel))
            except OSError as e:
                print(f"  [warn] skip {rel}: {e}", file=sys.stderr)
    return {
        "ref_dir":  os.path.abspath(ref),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "schema": schema_attrs(ref),
        "files": files,
    }


def compare(a: dict, b: dict) -> int:
    fa = {f["path"]: f for f in a["files"]}
    fb = {f["path"]: f for f in b["files"]}
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    differ = sorted(p for p in (set(fa) & set(fb)) if fa[p]["sha256"] != fb[p]["sha256"])
    print(f"  ref A: {a['ref_dir']}")
    print(f"  ref B: {b['ref_dir']}")
    print(f"  files only in A: {len(only_a)}")
    for p in only_a[:20]:
        print(f"    + {p}")
    print(f"  files only in B: {len(only_b)}")
    for p in only_b[:20]:
        print(f"    + {p}")
    print(f"  files with different SHA256: {len(differ)}")
    for p in differ[:20]:
        print(f"    Δ {p}")
        print(f"        A: {fa[p]['sha256'][:16]}... size={fa[p]['size']:>12d} mtime={fa[p]['mtime_iso']}")
        print(f"        B: {fb[p]['sha256'][:16]}... size={fb[p]['size']:>12d} mtime={fb[p]['mtime_iso']}")
    if only_a or only_b or differ:
        return 1
    print("  IDENTICAL")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("--out", default=None,
                    help="manifest output path (default: <ref_dir>/MANIFEST.json)")
    ap.add_argument("--compare", help="compare ref_dir's manifest against this JSON")
    args = ap.parse_args()

    if not os.path.isdir(args.ref_dir):
        print(f"[FAIL] not a directory: {args.ref_dir}", file=sys.stderr)
        return 2

    print(f"=== building manifest for {args.ref_dir} ===")
    mf = build_manifest(args.ref_dir)
    out_path = args.out or os.path.join(args.ref_dir, "MANIFEST.json")
    with open(out_path, "w") as f:
        json.dump(mf, f, indent=2)
    print(f"  wrote {out_path}  ({len(mf['files'])} files)")

    if args.compare:
        print(f"\n=== diff vs {args.compare} ===")
        with open(args.compare) as f:
            other = json.load(f)
        rc = compare(mf, other)
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
