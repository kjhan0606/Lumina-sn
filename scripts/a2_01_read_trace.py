#!/usr/bin/env python3
"""Prepare and summarize gated A2-01 runtime read traces.

The production tree is never rewritten.  ``prepare`` copies the current source
files to a requested scratch/build directory, wraps only the canonical census
read expressions, and adds a counter runtime.  The resulting executable is
still gated at runtime by ``LUMINA_A2_OWNER_TRACE=1``; without that variable it
does not create a trace file or print anything.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from a2_01_census_contract import SITES, token_matches, validate


CPU_SOURCES = [
    "lumina_main.c",
    "lumina_transport.c",
    "a2_02c_segment_capture.c",
    "lumina_plasma.c",
    "lumina_element_wide.c",
    "lumina_atomic.c",
    "lumina_cmfgen.c",
]
CUDA_SOURCES = [
    "lumina_cuda.cu",
    "lumina_bf_gemm.cu",
    "lumina_nlte_gemm.cu",
    "lumina_nlte_assemble.cu",
    "lumina_cmf_solve.cu",
    "lumina_atomic.c",
    "lumina_plasma.c",
    "lumina_element_wide.c",
    "lumina_cmfgen.c",
]
TRACE_ACCESSES = {"read", "readwrite", "device_read"}
RAW_SCHEMA = "lumina-a2-01-runtime-read-trace-v1"
SUMMARY_SCHEMA = "lumina-a2-01-runtime-read-summary-v1"


HEADER = r'''#ifndef LUMINA_A2_01_TRACE_RUNTIME_H
#define LUMINA_A2_01_TRACE_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif
void a2_01_trace_init(void);
void a2_01_trace_flush(void);
double a2_01_trace_host_read(unsigned int site_id, double value);
#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#ifdef A2_01_TRACE_DEFINE_DEVICE_GLOBALS
__device__ unsigned long long a2_01_trace_device_counts[157];
__device__ int a2_01_trace_device_on;
#else
extern __device__ unsigned long long a2_01_trace_device_counts[157];
extern __device__ int a2_01_trace_device_on;
#endif
static __device__ __forceinline__ double
a2_01_trace_device_read(unsigned int site_id, double value) {
    if (a2_01_trace_device_on && site_id < 157U)
        atomicAdd(&a2_01_trace_device_counts[site_id], 1ULL);
    return value;
}
#endif

#endif
'''


CPU_RUNTIME = r'''#include "a2_01_trace_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long counts[157];
static int gate = -1;
static int registered = 0;
static int flushed = 0;

static int enabled_from_env(void) {
    const char *value = getenv("LUMINA_A2_OWNER_TRACE");
    return value && strcmp(value, "1") == 0;
}

void a2_01_trace_init(void) {
    if (__atomic_load_n(&gate, __ATOMIC_ACQUIRE) < 0) {
        int desired = enabled_from_env();
        int expected = -1;
        __atomic_compare_exchange_n(&gate, &expected, desired, 0,
                                    __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (__atomic_load_n(&gate, __ATOMIC_ACQUIRE) &&
        !__atomic_exchange_n(&registered, 1, __ATOMIC_ACQ_REL))
        atexit(a2_01_trace_flush);
}

double a2_01_trace_host_read(unsigned int site_id, double value) {
    if (__atomic_load_n(&gate, __ATOMIC_ACQUIRE) < 0) a2_01_trace_init();
    if (__atomic_load_n(&gate, __ATOMIC_RELAXED) && site_id < 157U)
        __atomic_fetch_add(&counts[site_id], 1ULL, __ATOMIC_RELAXED);
    return value;
}

void a2_01_trace_flush(void) {
    const char *path;
    FILE *stream;
    unsigned int index;
    if (!__atomic_load_n(&gate, __ATOMIC_ACQUIRE) ||
        __atomic_exchange_n(&flushed, 1, __ATOMIC_ACQ_REL)) return;
    path = getenv("LUMINA_A2_OWNER_TRACE_PATH");
    if (!path || !*path) path = "a2_01_read_trace.tsv";
    stream = fopen(path, "w");
    if (!stream) return;
    fprintf(stream, "schema\t%s\n", "lumina-a2-01-runtime-read-trace-v1");
    fprintf(stream, "lane\tcpu\ncompleted\t1\n");
    for (index = 0; index < 157U; ++index)
        fprintf(stream, "A2R%03u\t%llu\n", index + 1U,
                __atomic_load_n(&counts[index], __ATOMIC_RELAXED));
    fclose(stream);
}
'''


CUDA_RUNTIME = r'''#define A2_01_TRACE_DEFINE_DEVICE_GLOBALS 1
#include "a2_01_trace_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long host_counts[157];
static int gate = -1;
static int registered = 0;
static int flushed = 0;

static int enabled_from_env(void) {
    const char *value = getenv("LUMINA_A2_OWNER_TRACE");
    return value && strcmp(value, "1") == 0;
}

extern "C" void a2_01_trace_init(void) {
    unsigned long long zeros[157] = {0};
    if (gate < 0) gate = enabled_from_env();
    if (!gate || registered) return;
    registered = 1;
    cudaMemcpyToSymbol(a2_01_trace_device_counts, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(a2_01_trace_device_on, &gate, sizeof(gate));
    atexit(a2_01_trace_flush);
}

extern "C" double a2_01_trace_host_read(unsigned int site_id, double value) {
    if (gate < 0) a2_01_trace_init();
    if (gate && site_id < 157U)
        __atomic_fetch_add(&host_counts[site_id], 1ULL, __ATOMIC_RELAXED);
    return value;
}

extern "C" void a2_01_trace_flush(void) {
    const char *path;
    FILE *stream;
    unsigned long long device_counts[157];
    unsigned int index;
    if (!gate || flushed) return;
    flushed = 1;
    cudaDeviceSynchronize();
    if (cudaMemcpyFromSymbol(device_counts, a2_01_trace_device_counts,
                             sizeof(device_counts)) != cudaSuccess)
        memset(device_counts, 0, sizeof(device_counts));
    path = getenv("LUMINA_A2_OWNER_TRACE_PATH");
    if (!path || !*path) path = "a2_01_read_trace.tsv";
    stream = fopen(path, "w");
    if (!stream) return;
    fprintf(stream, "schema\t%s\n", "lumina-a2-01-runtime-read-trace-v1");
    fprintf(stream, "lane\tcuda\ncompleted\t1\n");
    for (index = 0; index < 157U; ++index)
        fprintf(stream, "A2R%03u\t%llu\n", index + 1U,
                host_counts[index] + device_counts[index]);
    fclose(stream);
}
'''


def selected_source_names(kind: str) -> set[str]:
    return set(CPU_SOURCES if kind == "cpu" else CUDA_SOURCES)


def wrap_expression(kind: str, site_index: int, expression: str, access: str) -> str:
    if access == "device_read":
        if kind != "cuda":
            raise ValueError("device read selected for a non-CUDA tree")
        function = "a2_01_trace_device_read"
    else:
        function = "a2_01_trace_host_read"
    return f"{function}({site_index}U, ({expression}))"


def instrument_text(path: str, text: str, kind: str) -> tuple[str, list[int]]:
    lines = text.splitlines(keepends=True)
    replacements: dict[int, list[tuple[int, int, str, int]]] = defaultdict(list)
    source_name = Path(path).name
    if source_name not in selected_source_names(kind):
        return text, []
    for index, site in enumerate(SITES):
        if site.path != path or site.access not in TRACE_ACCESSES:
            continue
        if site.access == "device_read" and kind != "cuda":
            continue
        line = lines[site.line - 1]
        spans = token_matches(line, site.token)
        start, end = spans[site.occurrence - 1]
        replacement = wrap_expression(kind, index, line[start:end], site.access)
        replacements[site.line - 1].append((start, end, replacement, index))
    touched: list[int] = []
    for line_index, edits in replacements.items():
        original = lines[line_index]
        last_start = len(original) + 1
        for start, end, replacement, site_index in sorted(edits, reverse=True):
            if end > last_start:
                raise ValueError(f"overlapping trace sites in {path}:{line_index + 1}")
            original = original[:start] + replacement + original[end:]
            last_start = start
            touched.append(site_index)
        lines[line_index] = original
    if touched:
        lines.insert(0, '#include "a2_01_trace_runtime.h"\n')
    return "".join(lines), sorted(touched)


def inject_init(text: str) -> str:
    pattern = re.compile(r"(\bint\s+main\s*\([^)]*\)\s*\{)")
    replaced, count = pattern.subn(r"\1\n    a2_01_trace_init();", text, count=1)
    if count != 1:
        raise ValueError("could not inject trace initialization into main")
    return replaced


def prepare(repo: Path, build_dir: Path, kind: str) -> dict[str, object]:
    errors = validate(repo)
    if errors:
        raise ValueError("; ".join(errors))
    if build_dir.exists():
        raise ValueError(f"build directory already exists: {build_dir}")
    source_out = build_dir / "src"
    source_out.mkdir(parents=True)
    selected = selected_source_names(kind)
    touched: list[int] = []
    for source in sorted((repo / "src").iterdir()):
        if not source.is_file():
            continue
        target = source_out / source.name
        if source.name not in selected:
            shutil.copy2(source, target)
            continue
        transformed, indices = instrument_text(
            f"src/{source.name}", source.read_text(encoding="utf-8"), kind
        )
        if source.name == ("lumina_main.c" if kind == "cpu" else "lumina_cuda.cu"):
            transformed = inject_init(transformed)
        target.write_text(transformed, encoding="utf-8")
        touched.extend(indices)
    (source_out / "a2_01_trace_runtime.h").write_text(HEADER, encoding="utf-8")
    runtime_name = "a2_01_trace_runtime.c" if kind == "cpu" else "a2_01_trace_runtime.cu"
    (source_out / runtime_name).write_text(
        CPU_RUNTIME if kind == "cpu" else CUDA_RUNTIME, encoding="utf-8"
    )
    expected = [
        index
        for index, site in enumerate(SITES)
        if site.access in TRACE_ACCESSES
        and Path(site.path).name in selected
        and (kind == "cuda" or site.access != "device_read")
    ]
    if sorted(touched) != expected:
        raise ValueError(
            f"instrumentation coverage mismatch touched={len(touched)} expected={len(expected)}"
        )
    trace_map = {
        "schema": "lumina-a2-01-instrumented-tree-v1",
        "kind": kind,
        "source_head_informational": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
        ).stdout.strip(),
        "instrumented_site_count": len(touched),
        "instrumented_site_ids": [f"A2R{index + 1:03d}" for index in sorted(touched)],
    }
    (build_dir / "trace_map.json").write_text(
        json.dumps(trace_map, indent=2) + "\n", encoding="utf-8"
    )
    return trace_map


def compile_tree(build_dir: Path, kind: str, binary: Path, gpu_arch: str) -> None:
    source_dir = build_dir / "src"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if kind == "cpu":
        command = [
            "gcc", "-O2", "-Wall", "-Wextra", "-std=c11", "-I", str(source_dir),
            "-o", str(binary),
            *[str(source_dir / name) for name in CPU_SOURCES],
            str(source_dir / "a2_01_trace_runtime.c"), "-lm",
        ]
    else:
        command = [
            "nvcc", "-O2", "-rdc=true", f"-arch={gpu_arch}", "-std=c++14",
            "-Xcompiler", "-fopenmp",
            "-DLUMINA_HAS_CUDA_BF_GEMM", "-I", str(source_dir), "-o", str(binary),
            *[str(source_dir / name) for name in CUDA_SOURCES],
            str(source_dir / "a2_01_trace_runtime.cu"),
            "-lm", "-lcublas", "-Xcompiler", "-fopenmp",
        ]
    subprocess.run(command, check=True)


def read_raw(path: Path) -> tuple[str, list[int]]:
    metadata: dict[str, str] = {}
    counts = [0] * 157
    seen: set[int] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected two tab-separated fields")
        key, value = fields
        match = re.fullmatch(r"A2R(\d{3})", key)
        if match:
            index = int(match.group(1)) - 1
            if not 0 <= index < 157 or index in seen:
                raise ValueError(f"{path}:{number}: invalid/duplicate site {key}")
            counts[index] = int(value)
            seen.add(index)
        else:
            metadata[key] = value
    if metadata.get("schema") != RAW_SCHEMA or metadata.get("completed") != "1":
        raise ValueError(f"{path}: incomplete or wrong-schema trace")
    if metadata.get("lane") not in {"cpu", "cuda"}:
        raise ValueError(f"{path}: invalid lane")
    if len(seen) != 157:
        raise ValueError(f"{path}: site coverage {len(seen)} != 157")
    return metadata["lane"], counts


def summarize(paths: list[Path]) -> dict[str, object]:
    lanes: list[str] = []
    totals = [0] * 157
    per_lane: dict[str, list[int]] = {}
    for path in paths:
        lane, counts = read_raw(path)
        if lane in per_lane:
            raise ValueError(f"duplicate lane {lane}")
        lanes.append(lane)
        per_lane[lane] = counts
        totals = [left + right for left, right in zip(totals, counts)]
    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    for index, (site, count) in enumerate(zip(SITES, totals)):
        static_read = site.access in TRACE_ACCESSES
        applicable = (
            ("cpu" in lanes and Path(site.path).name in set(CPU_SOURCES) and site.access != "device_read")
            or ("cuda" in lanes and Path(site.path).name in set(CUDA_SOURCES))
        )
        if not static_read:
            status = "NOT_A_SCALAR_READ"
        elif not applicable:
            status = "PENDING_LANE_EXECUTION"
        elif count > 0:
            status = "READ_OBSERVED"
        else:
            status = "STATIC_READ_NOT_OBSERVED"
            mismatches.append(f"A2R{index + 1:03d} {site.path}:{site.line} {site.symbol}")
        rows.append(
            {
                "site_id": f"A2R{index + 1:03d}",
                "file_line": f"{site.path}:{site.line}",
                "symbol": site.symbol,
                "role": site.role,
                "static_access": site.access,
                "read_count": count,
                "status": status,
                "lane_counts": {lane: per_lane[lane][index] for lane in lanes},
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "lanes": lanes,
        "row_count": len(rows),
        "read_observed": sum(row["status"] == "READ_OBSERVED" for row in rows),
        "static_read_not_observed": len(mismatches),
        "pending_lane_execution": sum(row["status"] == "PENDING_LANE_EXECUTION" for row in rows),
        "not_a_scalar_read": sum(row["status"] == "NOT_A_SCALAR_READ" for row in rows),
        "trace_census_mismatches": mismatches,
        "rows": rows,
    }


def summary_markdown(document: dict[str, object]) -> str:
    rows = document["rows"]
    assert isinstance(rows, list)
    lines = [
        "# A2-01 런타임 read trace",
        "",
        f"- lanes: {', '.join(document['lanes'])}",
        f"- READ_OBSERVED: {document['read_observed']}",
        f"- STATIC_READ_NOT_OBSERVED: {document['static_read_not_observed']}",
        f"- PENDING_LANE_EXECUTION: {document['pending_lane_execution']}",
        f"- NOT_A_SCALAR_READ: {document['not_a_scalar_read']}",
        "",
        "| site | 파일:행 | 심볼 | 역할 | 정적 access | read count | 상태 |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['site_id']} | {row['file_line']} | {row['symbol']} | {row['role']} | "
            f"{row['static_access']} | {row['read_count']} | {row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def runtime_selftest(repo: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="a2_01_trace_selftest_") as raw:
        root = Path(raw)
        (root / "a2_01_trace_runtime.h").write_text(HEADER, encoding="utf-8")
        (root / "a2_01_trace_runtime.c").write_text(CPU_RUNTIME, encoding="utf-8")
        fixture = root / "fixture.c"
        fixture.write_text(
            '#include "a2_01_trace_runtime.h"\n#include <stdio.h>\n'
            'int main(void){double x=2.0;a2_01_trace_init();'
            'x=a2_01_trace_host_read(0U,x);x=a2_01_trace_host_read(0U,x);'
            'x=a2_01_trace_host_read(1U,x);printf("%.1f\\n",x);return 0;}\n',
            encoding="utf-8",
        )
        binary = root / "fixture"
        subprocess.run(
            ["gcc", "-O2", "-std=c11", "-I", str(root), "-o", str(binary),
             str(fixture), str(root / "a2_01_trace_runtime.c")], check=True
        )
        off = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, check=True)
        if (root / "trace.tsv").exists() or off.stdout != "2.0\n":
            raise ValueError("OFF-gate fixture changed output or emitted a trace")
        environment = dict(__import__("os").environ)
        environment["LUMINA_A2_OWNER_TRACE"] = "1"
        environment["LUMINA_A2_OWNER_TRACE_PATH"] = str(root / "trace.tsv")
        on = subprocess.run(
            [str(binary)], text=True, stdout=subprocess.PIPE, env=environment, check=True
        )
        lane, counts = read_raw(root / "trace.tsv")
        if on.stdout != off.stdout or lane != "cpu" or counts[:2] != [2, 1]:
            raise ValueError("runtime counter selftest failed")
        cpu_tree = root / "cpu_tree"
        trace_map = prepare(repo, cpu_tree, "cpu")
        if int(trace_map["instrumented_site_count"]) <= 0:
            raise ValueError("source transformer instrumented no CPU sites")
        print(
            "PASS A2_01_TRACE_SELFTEST gate_off_stdout_sha256="
            f"{hashlib.sha256(off.stdout.encode()).hexdigest()} "
            f"cpu_instrumented_sites={trace_map['instrumented_site_count']} counts=2,1"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--kind", choices=["cpu", "cuda"], required=True)
    prepare_parser.add_argument("--build-dir", type=Path, required=True)
    prepare_parser.add_argument("--compile", action="store_true")
    prepare_parser.add_argument("--binary", type=Path)
    prepare_parser.add_argument("--gpu-arch", default="sm_80")
    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("trace", nargs="+", type=Path)
    summarize_parser.add_argument("--json-out", type=Path, required=True)
    summarize_parser.add_argument("--table-out", type=Path, required=True)
    sub.add_parser("selftest")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.mode == "prepare":
            trace_map = prepare(repo, args.build_dir.resolve(), args.kind)
            if args.compile:
                if args.binary is None:
                    raise ValueError("--compile requires --binary")
                compile_tree(args.build_dir.resolve(), args.kind, args.binary.resolve(), args.gpu_arch)
            print(
                f"PASS A2_01_TRACE_PREPARE kind={args.kind} "
                f"instrumented_sites={trace_map['instrumented_site_count']} "
                f"build_dir={args.build_dir.resolve()}"
            )
            return 0
        if args.mode == "summarize":
            document = summarize([path.resolve() for path in args.trace])
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.table_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            args.table_out.write_text(summary_markdown(document), encoding="utf-8")
            print(
                "PASS A2_01_TRACE_SUMMARY rows=157 "
                f"observed={document['read_observed']} "
                f"not_observed={document['static_read_not_observed']} "
                f"pending={document['pending_lane_execution']}"
            )
            return 0
        runtime_selftest(repo)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"FAIL A2_01_READ_TRACE {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
