#!/usr/bin/env python3
"""NE4 structural gate for the MC-EVT event-measure consumers.

The positive check is not sufficient by itself: a structural gate also has to
demonstrate that it rejects an injected accessor bypass.  The in-memory
negative controls below exercise each registered consumer without modifying
the working tree.  CPU-A208 is measured at its implementation definition; its
public wrapper is separately pinned to exact pure delegation.
"""

from pathlib import Path
import re
import sys

from gate_source_lib import body, body_raw, inject_at_head


ROOT = Path(__file__).resolve().parent.parent / "src"
A208_PUBLISH_WRAPPER = """{
    return a208_publish_cpu_opacity_impl(
        opacity, bf, atom, plasma, nlte, epoch, a208_counters());
}"""
NEGATIVE_CONTROL_CASES = (
    (
        "CPU-T03",
        0,
        "single_packet_loop",
        "double ne4_injected = bf->chi_bf[0]; (void)ne4_injected;",
        "CPU-T03: direct event-grid indexing bypass",
    ),
    (
        "CPU-A208",
        1,
        "a208_publish_cpu_opacity_impl",
        "double ne4_injected = bf->event_chi_bf[0]; (void)ne4_injected;",
        "CPU-A208: direct event-grid indexing bypass",
    ),
    (
        "GPU-T03",
        2,
        "transport_kernel",
        "double ne4_injected = d_chi_bf[0]; (void)ne4_injected;",
        "GPU-T03: direct event-grid indexing bypass",
    ),
    (
        "GPU-VPACKET",
        2,
        "d_trace_virtual_packet",
        "double ne4_injected = d_chi_bf[0]; (void)ne4_injected;",
        "GPU-VPACKET: direct event-grid indexing bypass",
    ),
)


def inspect_sources(transport: str, plasma: str, cuda: str) -> list[str]:
    failures: list[str] = []

    cpu_t03 = body(transport, "single_packet_loop")
    cpu_a208 = body(plasma, "a208_publish_cpu_opacity_impl")
    gpu_t03 = body(cuda, "transport_kernel")
    gpu_vpacket = body(cuda, "d_trace_virtual_packet")

    if body_raw(plasma, "a208_publish_cpu_opacity") != A208_PUBLISH_WRAPPER:
        failures.append(
            "CPU-A208: a208_publish_cpu_opacity wrapper is not exact pure delegation"
        )

    for label, consumer_body, accessor in (
        ("CPU-T03", cpu_t03, "bf_event_measure_get"),
        ("CPU-A208", cpu_a208, "bf_event_measure_get"),
        ("GPU-T03", gpu_t03, "d_bf_event_measure_get"),
        ("GPU-VPACKET", gpu_vpacket, "d_bf_event_measure_get"),
    ):
        if accessor not in consumer_body:
            failures.append(f"{label}: missing {accessor}")

    for label, consumer_body, required_tokens in (
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
            if token not in consumer_body:
                failures.append(f"{label}: missing non-OK policy token {token!r}")

    for label, consumer_body, pattern in (
        ("CPU-T03", cpu_t03, r"\bbf->(?:event_chi_bf|chi_bf)\s*\["),
        # A208 legitimately reads bf->chi_bf to publish the signed net
        # coefficient.  Only direct access to the separate event grid bypasses
        # the event-measure accessor here.
        ("CPU-A208", cpu_a208, r"\bbf->event_chi_bf\s*\["),
        ("GPU-T03", gpu_t03, r"\bd_(?:bf_event_chi|chi_bf)\s*\["),
        ("GPU-VPACKET", gpu_vpacket,
         r"\bd_(?:bf_event_chi|chi_bf)\s*\["),
    ):
        if re.search(pattern, consumer_body):
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


def run_negative_controls(transport: str, plasma: str, cuda: str) -> list[str]:
    sources = (transport, plasma, cuda)
    missed: list[str] = []
    for index, (label, source_index, function, statement, expected) in enumerate(
        NEGATIVE_CONTROL_CASES, start=1
    ):
        mutated_sources = list(sources)
        try:
            mutated_sources[source_index] = inject_at_head(
                sources[source_index], function, statement
            )
        except RuntimeError:
            missed.append(f"injection-{index}-not-applied")
            continue
        if mutated_sources[source_index] == sources[source_index]:
            missed.append(f"injection-{index}-not-applied")
            continue
        failures = inspect_sources(*mutated_sources)
        if expected not in failures:
            missed.append(f"injection-{index}-wrong-reason")
            continue
        print(
            "[E-NE4][NEGATIVE-CONTROL][DETECTED] "
            f"injection={index} consumer={label} reason={expected}"
        )
    return missed


def inspect_gpu_owner(owner_source: str) -> list[str]:
    owner = body(owner_source, "gpu_opacity_production_bind")
    if not re.search(
        r"production_opacity\.view\.bf_event_measure_provenance\s*=\s*"
        r"p->bf_event_measure_provenance",
        owner,
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
