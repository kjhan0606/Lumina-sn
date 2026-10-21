#!/usr/bin/env python3
"""B의 별도 봉인 실행. 입력/참조 외곽만 재사용하고 C 수송은 선택하지 않는다."""
import argparse
from pathlib import Path
import lumina_gpu_local as local
from lumina_gpu_linear import Runtime, LinearDriver

local.SCHEMA='lumina-gpu-linear-native-local-v1'
local.LIMITS=dict(local.LIMITS,engine='SPACE_LINEAR_MOMENT_POSITIVE_SOURCE_B',
    cache_integrator_workspace_GiB=1,source_relative=1e-8,spatial_relative=1e-5,
    normalization='PER_FREQUENCY_RELATIVE_NO_GLOBAL_BRIGHT_NORM')
local.Runtime=Runtime
local.GpuDriver=LinearDriver

def prepare(root):
    old=Path('/gpfs/kjhan/cmfgen_runs/lumina_gpu_batch_local_20261021-xSx3Z2')
    fixture=[old/'tests'/p for p in ('test_lumina_nodal_driver.py','test_lumina_compiled_input.py',
        'test_lumina_outer_atmosphere.py','test_lumina_frequency_history.py')]
    saved=Path('/gpfs/kjhan/cmfgen_runs/lumina_hybrid_pair_20261021-revhwm/own16-work/test_same_material_own_j_resid0/history-own-j.json')
    local.prepare(root,extra_files=['native.so',*fixture,saved])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('path');parser.add_argument('--sha256')
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.path)
    else:raise SystemExit(local.run(args.path,args.sha256))
