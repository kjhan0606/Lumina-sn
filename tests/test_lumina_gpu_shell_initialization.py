"""시스템 프로필 초기화와 실제 계산 전 엄격 옵션 복원을 합성 검증한다."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=('submit_lumina_continue_accepted.slurm','submit_lumina_resident_observer.slurm')


def execute(tmp_path,name,profile,after,*,old_order=False):
    stub=tmp_path/'profile.sh';stub.write_text(profile)
    prefix=(ROOT/'scripts'/name).read_text().split('export OPENBLAS_NUM_THREADS=',1)[0]
    assert prefix.count('source /etc/profile')==1
    prefix=prefix.replace('source /etc/profile','source "$LUMINA_TEST_PROFILE"')
    if old_order:prefix=prefix.replace('set -eo pipefail','set -euo pipefail')
    env={k:v for k,v in os.environ.items() if k not in ('BASHRCSOURCED','BASH_ENV','ENV')}
    env['LUMINA_TEST_PROFILE']=str(stub)
    return subprocess.run(['bash','--noprofile','--norc','-c',prefix+'\n'+after],
                          env=env,capture_output=True,text=True,timeout=10)


@pytest.mark.parametrize('name',SCRIPTS)
def test_unset_profile_variable_is_allowed_and_strict_options_restored(tmp_path,name):
    result=execute(tmp_path,name,
        'if [ -z "$BASHRCSOURCED" ]; then BASHRCSOURCED=Y; fi\n'
        'module() { return 0; }\n',
        '[[ $- == *e* && $- == *u* && :$SHELLOPTS: == *:pipefail:* ]]\n'
        'test "$BASHRCSOURCED" = Y\n')
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('name',SCRIPTS)
def test_old_initialization_order_reproduces_unbound_variable_failure(tmp_path,name):
    result=execute(tmp_path,name,
        'if [ -z "$BASHRCSOURCED" ]; then BASHRCSOURCED=Y; fi\n'
        'module() { return 0; }\n','printf SHOULD_NOT_START\n',old_order=True)
    assert result.returncode!=0 and 'BASHRCSOURCED: unbound variable' in result.stderr
    assert 'SHOULD_NOT_START' not in result.stdout


@pytest.mark.parametrize('name',SCRIPTS)
def test_profile_failure_still_stops_before_calculation(tmp_path,name):
    result=execute(tmp_path,name,'return 7\n','printf SHOULD_NOT_START\n')
    assert result.returncode==7 and 'SHOULD_NOT_START' not in result.stdout


@pytest.mark.parametrize('name',SCRIPTS)
def test_undefined_variable_after_initialization_is_not_silently_accepted(tmp_path,name):
    result=execute(tmp_path,name,'module() { return 0; }\n',
                   'unset LUMINA_UNDEFINED_TEST_VARIABLE\n'
                   ': "$LUMINA_UNDEFINED_TEST_VARIABLE"\n')
    assert result.returncode!=0 and 'unbound variable' in result.stderr
