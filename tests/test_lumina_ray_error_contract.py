"""기존 양의 맵 대수로 국소 오차와 경로 누적 오차를 구별한다.

작은 해석적 반례이며 실제 native90 장의 오차 측정은 아니다.
"""
import math
from pathlib import Path
import sys

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from lumina_linear_native import Map,Positive


@pytest.mark.parametrize('intensity',[1e-80,1.,1e80])
def test_passing_every_local_ratio_does_not_prove_final_phot_accuracy(intensity):
    zero=Positive();one=Positive.of(1.)
    # 참 전달률1은 각 근사 T의 log오차구간 안에 있다. source는 양쪽 모두 정확히0.
    delta=5e-6;local=Map(Positive.of(1.+delta),zero,zero,math.log1p(delta))
    value=initial=Positive.of(intensity);whole=Map(one,zero,zero,0.)
    for _ in range(30):
        value,error=local.apply(value)
        assert error<=value*1e-5
        whole=whole.then(local)
    propagated,uncertainty=whole.apply(initial)
    bias=propagated.difference(initial)
    assert not bias<=initial*1e-4  # 모든 local통과에도 최종값은1e-4 밖이다.
    assert bias<=uncertainty       # 기존 맵대수로 누적하면 이 반례를 감싼다.
    assert value.difference(propagated)<=uncertainty
