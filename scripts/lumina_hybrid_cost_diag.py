#!/usr/bin/env python3
"""고정 원입력 표본의 두 kernel 혼합 실행을 같은 CPU 참조와 대조한다."""
import argparse
from pathlib import Path
import lumina_native_cost_diag as diag
from lumina_gpu_hybrid import HybridDriver

diag.CompactDriver=HybridDriver

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    raise SystemExit(diag.gpu(args.root.resolve(),three_piece_reference=
        Path('/gpfs/kjhan/cmfgen_runs/lumina_prefix569899_global_20261022-IZWHDt/cpu-three-pieces.json')))
