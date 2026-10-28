# Lageunha 물질응답 수동 실행 준비 — 2026-10-28

## 현재 상태와 목적

사용자가 Lageunha 수동 실행을 지시했다. 직전 응답에서는 실행 대상이 완성된
F2의 고정온도 물질응답·CMFGEN 비교인지 확인을 요청했다. 현재 그 대상 확인과
추가 단회 권한은 대기 상태이며, 이번 준비를 실제 F2 실행으로 표시하지 않는다.
원 격자 이차 대역 계산은 내부 blue 이력의 공간 자유도가 부족해 생산/수렴
참조에서 제외됐으므로 다시 연장하지 않는다.

이번 구현은 이미 있는 새 장 소비·BF/BB/FF·공통전하/SE·동일 반경 Si/S 비교를
Slurm 밖에서도 같은 자원 한도로 실행하기 위한 위치 변경이다. 물리식·온도
처리·원자 과정·수용 기준·반복 한도는 바꾸지 않는다. 새 전주파수 수송,
released-T 계산, 복사평형/공동수렴, observer 실행은 포함하지 않는다.

## 변경과 사용 방법

- `scripts/run_lumina_new_field_response_lageunha.sh`: Lageunha에서 직접 실행하는
  진입점이다. 기존 컴파일·봉인 소스·응답 실행기를 그대로 재사용한다.
- 실제 affinity를 허용 CPU 중16개로 고정하고 주소공간을32GiB로 제한한다.
  OpenMP16, BLAS1, GPU0이며 빌드 포함 벽4시간, 내부 응답13800초 한도다.
  timeout의 강제 종료 유예는30초다. 자동 재시도/이어달리기는 없다.
- `lumina_new_field_response.py`는 명시한 수동 모드에서 실제 호스트·affinity·
  주소공간 제한을 확인한다. Slurm 변수를 가짜로 채우지 않는다. 기존 Slurm
  실행도 유지하며 실행 방식·호스트·PID·제한을 기존 결과 JSON에 기록한다.
- 새 parent의 봉인 소스/바이너리와 새 `run`에만 출력한다. 기존 run이 있으면
  실행하지 않으며 원 입력과 완료 결과를 덮어쓰지 않는다.

실제 추가 단회 권한과 입력이 확정되면 Lageunha에서 아래 형태로 실행한다.
명령 기재는 실행 또는 추가 계산권 사용을 뜻하지 않는다.

```bash
bash scripts/run_lumina_new_field_response_lageunha.sh \
  "$NEW_PARENT" "$SEALED_COMMIT" \
  --preparation "$PREPARATION" --preparation-sha "$PREPARATION_SHA" \
  --feedback "$FEEDBACK" --feedback-sha "$FEEDBACK_SHA"
```

F1 실측은 전체11080초(약3시간5분), BF9613초(약2시간40분),
최대 RSS 약5.1GiB였다. F2 완료 보장은 아니며4시간에는 약55분의 여유가 있다.
F1 단회 승인 소진을 F2 승인으로 재해석하지 않는다.

## 직접시험과 남은 증거

Lageunha에서 CPU1 affinity·4GiB·150초 한도로 아래 시험을 수행했다.
**30 passed / 4.82초**였다. 쉘 두 파일의 `bash -n`과 `git diff --check`도 통과했다.
새 시험은 수동 자원 확인과 기존 Slurm 경로를 검증하며, 실제 BF/SE 생산
계산이나 물리 동질성을 검증한 것으로 표시하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
  python3 -m pytest -q -p no:cacheprovider \
  tests/test_lumina_new_field_response.py \
  tests/test_lumina_ion_number_flows.py tests/test_lumina_model_balance.py
```

실제 F2 후속 실행의 판단에는 정확한 완성 J/모멘트·수용 물질 계보, 재생성한
율과 물질응답, 모든 원래 반경의 Si/S 비교가 필요하다. 그 유한 응답이 끝나도
고정온도·비수렴 CMFGEN70 비교이며 산란/RE/공동수렴과 최종 스펙트럼의
동질성을 입증하지 않는다. 전체 목표는 계속 **NOT_MET**다.
