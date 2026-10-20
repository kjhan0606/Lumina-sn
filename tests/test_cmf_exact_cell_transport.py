"""실제 경계/셀 이동과 확산 광도의 직접 수리 시험. 큰 물리 계산 아님."""
from pathlib import Path
import os
import subprocess


def test_physical_cell_and_diffusion_boundary(tmp_path):
    root=Path(__file__).resolve().parents[1]
    binary=tmp_path/'cmf_cell'
    subprocess.run([os.environ.get('CC','cc'),'-O2','-Wall','-Wextra','-Werror',
        '-std=c11','-fopenmp','-I'+str(root/'src'),
        str(root/'tests/cmf_exact_cell_selftest.c'),
        str(root/'src/cmf_exact_sliding.c'),str(root/'src/cmf_error_envelope.c'),
        '-lm','-fopenmp','-o',str(binary)],check=True,timeout=60)
    result=subprocess.run([str(binary)],check=True,text=True,capture_output=True,
        env={**os.environ,'OMP_NUM_THREADS':'1'},timeout=60)
    assert 'CMF_CELL_SELFTEST PASS' in result.stdout


def test_cell_streaming_matches_full_buffer_all_outputs(tmp_path):
    root=Path(__file__).resolve().parents[1]
    results=[]
    for reference in (False,True):
        binary=tmp_path/('full' if reference else 'streamed')
        output=binary.with_suffix('.bin')
        command=[os.environ.get('CC','cc'),'-O2','-Wall','-Wextra','-Werror','-std=c11',
            '-fopenmp','-I'+str(root/'src')]
        if reference:command+=['-DCMF_EXACT_TEST_FULL_CELL_BUFFER']
        subprocess.run(command+[str(root/'tests/cmf_cell_streaming_selftest.c'),
            str(root/'src/cmf_exact_sliding.c'),str(root/'src/cmf_error_envelope.c'),
            '-lm','-fopenmp','-o',str(binary)],check=True,timeout=60,capture_output=True)
        subprocess.run([str(binary),str(output)],check=True,timeout=60,
            env={**os.environ,'OMP_NUM_THREADS':'1'},capture_output=True)
        results.append(output.read_bytes())
    assert results[0]==results[1]
