#!/usr/bin/env python3
"""Build-only fixture for A2-02C capture gate OFF/ON behavioral parity."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

from a2_02c_segment_replay import read_capture


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r'''
#include "lumina.h"
#include "a2_02c_segment_capture.h"
double get_doppler_factor(double r,double mu,double t) {
    return 1.0-mu*r/(C_SPEED_OF_LIGHT*t);
}
int main(void) {
    double ri[1]={1.0e14},ro[1]={1.1e14},vi[1]={0},vo[1]={0},vol[1]={1.0e43};
    Geometry g={1,ri,ro,vi,vo,1.0e6};
    RPacket p={1.0e14,0.5,2.0e15,0.5,0,0,PACKET_IN_PROCESS,0};
    RPacket before=p;
    a2_02c_capture_begin(1,1,&g,vol,2.0e-40);
    a2_02c_capture_segment(&p,1,1.0e10,g.time_explosion);
    a2_02c_capture_end();
    if (memcmp(&p,&before,sizeof(p))!=0) return 7;
    printf("science=17\n");
    return 0;
}
'''


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a2_02c_capture_gate_", dir="/tmp") as raw:
        work = Path(raw)
        source = work / "fixture.c"
        source.write_text(FIXTURE)
        binary = work / "fixture"
        subprocess.run([
            "gcc", "-O2", "-std=c11", "-I", str(ROOT / "src"),
            str(source), str(ROOT / "src/a2_02c_segment_capture.c"), "-lm",
            "-o", str(binary),
        ], check=True)
        off_path = work / "off.bin"
        off_env = dict(os.environ)
        off_env.update({"LUMINA_A2_02C_SEGMENT_CAPTURE": "0",
                        "LUMINA_A2_02C_CAPTURE_GENERATION": "1",
                        "LUMINA_A2_02C_CAPTURE_PATH": str(off_path)})
        off = subprocess.run([str(binary)], env=off_env, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if off_path.exists():
            raise RuntimeError("gate OFF created a capture")
        on_path = work / "on.bin"
        on_env = dict(os.environ)
        on_env.update({"LUMINA_A2_02C_SEGMENT_CAPTURE": "1",
                       "LUMINA_A2_02C_CAPTURE_GENERATION": "1",
                       "LUMINA_A2_02C_CAPTURE_PATH": str(on_path)})
        on = subprocess.run([str(binary)], env=on_env, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if off.stdout != on.stdout or off.stderr != on.stderr:
            raise RuntimeError("gate ON changed stdout/stderr")
        header, records = read_capture(on_path)
        if header["segment_count"] != 1 or records[0]["packet_id"] != 0:
            raise RuntimeError("capture record identity mismatch")
    print("A2_02C_CAPTURE_GATE_SELFTEST PASS off_file=0 on_file=1 output_parity=byte packet_mutation=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
