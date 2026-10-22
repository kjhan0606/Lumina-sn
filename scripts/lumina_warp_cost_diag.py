#!/usr/bin/env python3
"""같은 실제35열/제한867질의에서 협력 커널을 기존 CPU 참조와 대조한다."""
import argparse
from pathlib import Path
import lumina_native_cost_diag as diag
from lumina_gpu_compact import WarpCompactDriver

diag.CompactDriver=WarpCompactDriver

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    raise SystemExit(diag.gpu(args.root.resolve()))
