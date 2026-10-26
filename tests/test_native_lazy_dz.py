"""실제 native 초기화 helper의 관측 투영을 비교한다. 물리 수렴 시험 아님."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import prepare_cmfgen_lazy_dz as D
from prepare_cmfgen_mom_capture import FLAGS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_block(block):
    return '\n'.join(line.strip() for line in block.splitlines())


class NativeFixture:
    def __init__(self, native, root):
        self.native, self.root = native, root
        self.hashes = {}
        for name in D.SEALED:
            D.verify((native / name).read_bytes(), name)
            self.hashes[name] = sha(native / name)
        self.main = (native / 'new_main/cmfgen_sub.f').read_text()
        if hashlib.sha256(self.main.encode()).hexdigest() != (
                '092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374'):
            raise ValueError('검토한 main 신원 변경')
        self.init_bytes = (native / 'new_main/mod_subs/init_line_opac_var_v2.f').read_bytes()
        self.set_line = (native / 'new_main/mod_subs/set_line_opac.f').read_text()
        self.modules = root / 'modules'
        self.modules.mkdir()
        self.command(['gfortran', *FLAGS, '-c', native / 'tools/set_kind_module.f',
                      native / 'new_main/mod_subs/line_mod.f'], self.modules)

    def command(self, command, cwd):
        result = subprocess.run(list(map(str, command)), cwd=cwd, check=False,
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    def compile(self, mode, memory=False, bad_mask=False, omit_column_reset=False):
        name = mode + ('-memory' if memory else '') + ('-bad-mask' if bad_mask else '')
        name += '-no-column-reset' if omit_column_reset else ''
        folder = self.root / name
        folder.mkdir()
        init = self.init_bytes.decode() if mode == 'original' else D.initializer(self.init_bytes)
        source = D.helper(init)
        if bad_mask:
            source = source.replace('IF(LINE_STORAGE_USED(L) .AND.',
                                    'IF(L .EQ. 1 .AND. LINE_STORAGE_USED(L) .AND.')
        (folder / 'helper.f').write_text(source)
        if memory:
            template = Path(__file__).with_name('native_lazy_dz_memory.f90').read_text()
        else:
            template = Path(__file__).with_name('native_lazy_dz_driver.f90').read_text()
            main = self.main if mode == 'original' else D.main_source(self.main)
            for row, marker in (('NL', 'LOW'), ('NUP', 'UP')):
                block = D.retirement_block(row, mode != 'original')
                if main.count(block) != 1:
                    raise ValueError('실제 해제 블록 불일치')
                template = template.replace('! RETIRE_' + marker + '_BLOCK', free_block(block))
            reset = '\t      CALL ZERO_2D_MAT(dZ(1,1,1,SIM_INDX),NM,NUM_BNDS*ND)'
            if init.count(reset) != 1:
                raise ValueError('새 strong 전체 컬럼 reset 변경')
            template = template.replace('! NEW_COLUMN_RESET', '' if omit_column_reset else reset.strip())
            start = self.set_line.index('\t        I=LINE_LOC(FIRST_LINE)')
            end = self.set_line.index('\t        UP_POINTER(I)=0', start) + len('\t        UP_POINTER(I)=0')
            template = template.replace('! FORCE_RECLAIM_BLOCK', free_block(self.set_line[start:end]))
            start = self.set_line.index('          LINE_STORAGE_USED(1:MAX_SIM)=.FALSE.')
            end = self.set_line.index('          NUM_OF_WEAK_LINES=0.0_LDP', start)
            end += len('          NUM_OF_WEAK_LINES=0.0_LDP')
            template = template.replace('! SWEEP_RESET_BLOCK', free_block(self.set_line[start:end]))
        (folder / 'driver.f90').write_text(template)
        self.command(['gfortran', *FLAGS, '-I' + str(self.modules), '-c', 'helper.f'], folder)
        self.command(['gfortran', '-fopenmp', '-O', '-ffree-line-length-none', '-fcheck=all',
            '-I' + str(self.modules), 'driver.f90', 'helper.o',
            self.modules / 'set_kind_module.o', self.modules / 'line_mod.o',
            '-L' + str(self.native / 'lib'), '-lsubs', '-lpgplt', '-lmod',
            self.native / 'lib/libmy_lapack.a', self.native / 'lib/libmy_blas.a',
            '-o', 'test.exe'], folder)
        return folder / 'test.exe'

    def check_unchanged(self):
        if any(sha(self.native / name) != digest for name, digest in self.hashes.items()):
            raise ValueError('시험 중 native 소스 변경')


class SourceTests(unittest.TestCase):
    def test_unique_retirement_site_and_barrier(self):
        source = D.retirement_block('NL') + D.retirement_block('NUP')
        result = D.main_source(source)
        self.assertEqual(result.count('CALL ZERO_dZ'), 2)
        self.assertNotIn('dZ(NL,:,:,:)', result)
        self.assertIn('!$OMP END PARALLEL WORKSHARE\n\t        CALL ZERO_dZ', result)
        for invalid in (result, source * 2, 'unrelated'):
            with self.assertRaises(ValueError):
                D.main_source(invalid)

    def test_source_identity_is_mandatory(self):
        with self.assertRaises(ValueError):
            D.initializer(b'changed')


@unittest.skipUnless(os.environ.get('CMFGEN_SOURCE_ROOT'), '명시 native 소스 필요')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='lumina-lazy-dz-')
        cls.fixture = NativeFixture(Path(os.environ['CMFGEN_SOURCE_ROOT']), Path(cls.tmp.name))
        cls.executables = {mode: cls.fixture.compile(mode) for mode in ('original', 'corrected')}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.check_unchanged()
        cls.tmp.cleanup()

    def execute(self, executable, label, threads=1):
        folder = self.fixture.root / label
        folder.mkdir()
        result = subprocess.run([str(executable)], cwd=folder, capture_output=True,
            text=True, timeout=30, env=dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE'))
        return result, (folder / 'projection.bin').read_bytes()

    def test_actual_lifecycle_projection_and_consumer_sum(self):
        allowed = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
        reference = None
        for threads in ((1, 2) if allowed >= 2 else (1,)):
            for mode in ('original', 'corrected'):
                result, binary = self.execute(self.executables[mode], f'{mode}-{threads}', threads)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('LIFECYCLE_PASS', result.stdout)
                reference = binary if reference is None else reference
                self.assertEqual(binary, reference, f'{mode}/OMP{threads}')

    def test_negative_control_missing_column_reset_is_detected(self):
        executable = self.fixture.compile('corrected', omit_column_reset=True)
        result, _ = self.execute(executable, 'negative-column-reset')
        self.assertNotEqual(result.returncode, 0)

    def test_negative_control_clearing_only_one_live_column_is_detected(self):
        executable = self.fixture.compile('corrected', bad_mask=True)
        result, binary = self.execute(executable, 'negative-mask')
        reference_result, reference = self.execute(self.executables['original'], 'negative-reference')
        self.assertEqual(reference_result.returncode, 0)
        self.assertTrue(result.returncode != 0 or binary != reference)


def memory_probe(out):
    if out.exists():
        raise FileExistsError('기존 메모리 증거 보존')
    if os.environ.get('SLURM_MEM_PER_NODE') != '4096' or os.environ.get('SLURM_CPUS_PER_TASK') != '1':
        raise ValueError('큰 가상배열 시험은 Slurm CPU1/4GiB에서만 실행')
    out.mkdir()
    fixture = NativeFixture(Path(os.environ['CMFGEN_SOURCE_ROOT']), out)
    rows = {}
    # 원본 full8000은1224133에서4GiB OOM이었다. 같은 실패를 반복하지 않는다.
    # 수정본은 full8000, 원본은 같은 가상 예약의 앞256컬럼만 지운다.
    # 원본 prefix의 전체 주소 구간도2.75GiB 이하라 페이지 크기와 무관하게 제한된다.
    for mode, scope in (('corrected', 'full'), ('original', 'prefix256')):
        executable = fixture.compile(mode, memory=True)
        result = fixture.command([executable, scope], executable.parent)
        (executable.parent / 'stdout.txt').write_text(result.stdout)
        rss = re.search(r'VmHWM:\s*(\d+)\s*kB', result.stdout)
        active_sum = re.search(r'ACTIVE_SUM\s+(\S+)', result.stdout)
        huge = re.search(r'AnonHugePages:\s*(\d+)\s*kB', result.stdout)
        if rss is None or active_sum is None:
            raise ValueError('메모리 실측 누락')
        rows[mode] = dict(max_rss_KiB=int(rss[1]), active_sum=float(active_sum[1]),
                          anonymous_hugepages_KiB=(int(huge[1]) if huge else None),
                          zeroing_scope=scope, zeroed_column_range=(8000 if scope == 'full' else 256),
                          executable_sha256=sha(executable), stdout_sha256=sha(executable.parent / 'stdout.txt'))
    fixture.check_unchanged()
    if rows['original']['active_sum'] != rows['corrected']['active_sum']:
        raise ValueError('사용 strong 결과 불일치')
    if (rows['original']['max_rss_KiB'] <= rows['corrected']['max_rss_KiB'] + 16*1024 or
        rows['corrected']['max_rss_KiB'] >= 256*1024):
        raise ValueError('광범위 행 touch 제거 효과를 재현하지 못함')
    report = dict(state='DIRECT_MEMORY_EFFECT_REPRODUCED', virtual_dZ_bytes=92194560000,
        touched_strong_columns=4, row_cleared=7, input_sha256=fixture.hashes, measurements=rows,
        original_full_loop_measured_successfully=False,
        performance_comparison_scope='수정 full8000 대 원본 prefix256; 같은 전체 작업량의 속도 비교 아님',
        full_native_peak_memory_proven=False, full_native_physical_equivalence_proven=False)
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--memory-probe':
        memory_probe(Path(sys.argv[2]))
    else:
        unittest.main()
