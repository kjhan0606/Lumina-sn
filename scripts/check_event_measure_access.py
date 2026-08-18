#!/usr/bin/env python3
"""NE4 structural gate for the MC-EVT event-measure consumers.

The positive check is not sufficient by itself: a structural gate also has to
demonstrate that it rejects an injected accessor bypass.  The in-memory
negative controls below exercise each registered consumer without modifying
the working tree.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent / "src"


def function_body(text: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", text, re.S)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    start = match.end() - 1
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise RuntimeError(f"unterminated function: {name}")


def inspect_sources(transport: str, plasma: str, cuda: str) -> list[str]:
    failures = []

    cpu_t03 = function_body(transport, "single_packet_loop")
    cpu_a208 = function_body(plasma, "a208_publish_cpu_opacity")
    gpu_t03 = function_body(cuda, "transport_kernel")
    gpu_vpacket = function_body(cuda, "d_trace_virtual_packet")

    for label, body, accessor in (
        ("CPU-T03", cpu_t03, "bf_event_measure_get"),
        ("CPU-A208", cpu_a208, "bf_event_measure_get"),
        ("GPU-T03", gpu_t03, "d_bf_event_measure_get"),
        ("GPU-VPACKET", gpu_vpacket, "d_bf_event_measure_get"),
    ):
        if accessor not in body:
            failures.append(f"{label}: missing {accessor}")

    for label, body, required_tokens in (
        ("CPU-T03", cpu_t03,
         ("event_status != BF_EVENT_MEASURE_OK",
          "pkt->status = PACKET_REABSORBED", "event_measure_t03_blocks")),
        ("CPU-A208", cpu_a208,
         ("event_status != BF_EVENT_MEASURE_OK",
          "a208_publication_free(&candidate); return 3")),
        ("GPU-T03", gpu_t03,
         ("event_status != BF_EVENT_MEASURE_OK",
          "d_bf_event_measure_record_block(event_status, 0)",
          "pkt_status = 2")),
        ("GPU-VPACKET", gpu_vpacket,
         ("event_status != BF_EVENT_MEASURE_OK",
          "d_bf_event_measure_record_block(event_status, 1)", "return")),
    ):
        for token in required_tokens:
            if token not in body:
                failures.append(f"{label}: missing non-OK policy token {token!r}")

    for label, body, pattern in (
        ("CPU-T03", cpu_t03, r"\bbf->(?:event_chi_bf|chi_bf)\s*\["),
        # A208 legitimately reads bf->chi_bf to publish the signed net
        # coefficient.  Only direct access to the separate event grid bypasses
        # the event-measure accessor here.
        ("CPU-A208", cpu_a208, r"\bbf->event_chi_bf\s*\["),
        ("GPU-T03", gpu_t03, r"\bd_(?:bf_event_chi|chi_bf)\s*\["),
        ("GPU-VPACKET", gpu_vpacket,
         r"\bd_(?:bf_event_chi|chi_bf)\s*\["),
    ):
        if re.search(pattern, body):
            failures.append(f"{label}: direct event-grid indexing bypass")

    forbidden = {
        "removed scalar getter": "bf_get_event_measure(",
        "GPU direct opacity getter": "d_bf_get_chi(",
        "GPU silent grid fallback":
            "(bf_event_enabled && d_bf_event_chi)\n"
            "                  ? d_bf_event_chi : d_chi_bf",
    }
    joined = "\n".join((transport, plasma, cuda))
    for label, token in forbidden.items():
        if token in joined:
            failures.append(f"{label}: {token!r}")

    if re.search(r"event_(?:bf|measure)\s*=.*?bf->(?:event_)?chi_bf\s*\[",
                 cpu_a208, re.S):
        failures.append("CPU-A208: direct event-grid indexing bypass")

    return failures


def inject_into_function(text: str, name: str, statement: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{",
                      text, re.S)
    if not match:
        raise RuntimeError(f"function not found for injection: {name}")
    return text[:match.end()] + "\n    " + statement + text[match.end():]


def run_negative_controls(transport: str, plasma: str, cuda: str) -> list[str]:
    cases = (
        ("CPU-T03", inject_into_function(
            transport, "single_packet_loop",
            "double ne4_injected = bf->chi_bf[0]; (void)ne4_injected;"),
         plasma, cuda),
        ("CPU-A208", transport, inject_into_function(
            plasma, "a208_publish_cpu_opacity",
            "double ne4_injected = bf->event_chi_bf[0]; (void)ne4_injected;"),
         cuda),
        ("GPU-T03", transport, plasma, inject_into_function(
            cuda, "transport_kernel",
            "double ne4_injected = d_chi_bf[0]; (void)ne4_injected;")),
        ("GPU-VPACKET", transport, plasma, inject_into_function(
            cuda, "d_trace_virtual_packet",
            "double ne4_injected = d_chi_bf[0]; (void)ne4_injected;")),
    )
    missed = []
    for expected_label, t_src, p_src, c_src in cases:
        failures = inspect_sources(t_src, p_src, c_src)
        expected = f"{expected_label}: direct event-grid indexing bypass"
        if expected not in failures:
            missed.append(expected_label)
    return missed


def inspect_gpu_owner(owner_source: str) -> list[str]:
    body = function_body(owner_source, "gpu_opacity_production_bind")
    if not re.search(
        r"production_opacity\.view\.bf_event_measure_provenance\s*=\s*"
        r"p->bf_event_measure_provenance",
        body,
        re.S,
    ):
        return ["canonical GPU opacity owner does not publish event provenance"]
    return []


def main() -> int:
    transport = (ROOT / "lumina_transport.c").read_text()
    plasma = (ROOT / "lumina_plasma.c").read_text()
    cuda = (ROOT / "lumina_cuda.cu").read_text()
    gpu_owner = (ROOT / "gpu_opacity_kernels.cu").read_text()
    failures = inspect_sources(transport, plasma, cuda)

    if failures:
        for failure in failures:
            print(f"[E-NE4][FAIL] {failure}", file=sys.stderr)
        return 1
    print("[E-NE4][PASS] all CPU/GPU event consumers use the status accessor, "
          "block non-OK status, and retain no chi_bf event fallback")
    owner_failures = inspect_gpu_owner(gpu_owner)
    if owner_failures:
        for failure in owner_failures:
            print(f"[E-E2][STATIC][FAIL] {failure}", file=sys.stderr)
        return 1
    print("[E-E2][STATIC][PASS] canonical GPU opacity owner publishes "
          "event-measure provenance")
    missed = run_negative_controls(transport, plasma, cuda)
    if missed:
        print("[E-NE4][NEGATIVE-CONTROL][FAIL] missed=" + ",".join(missed),
              file=sys.stderr)
        return 1
    print("[E-NE4][NEGATIVE-CONTROL][PASS] injections=4 detected=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
