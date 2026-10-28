# Lageunha 물질응답 수동 실행 준비 — 2026-10-28

## 현재 상태와 목적

사용자가 Lageunha 수동 실행을 지시했고, 추가 확인 뒤 「승인함」으로 완성된
F2의 고정온도 물질응답·CMFGEN 비교 추가1회를 승인했다. 승인 범위는
CPU16·32GiB·최대4시간·GPU0·재시도0이다. 11:59:31 KST에 아래 전용 root에서
수동 실행기를 시작했다. 실행 착수를 완료 응답이나 물리 동질성으로 표시하지 않는다.
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

재현 명령 형식은 아래와 같다. 이번 승인1회는 아래 실행에 사용했으며, 명령
기재가 별도 재실행·이어달리기 승인을 뜻하지 않는다.

```bash
bash scripts/run_lumina_new_field_response_lageunha.sh \
  "$NEW_PARENT" "$SEALED_COMMIT" \
  --preparation "$PREPARATION" --preparation-sha "$PREPARATION_SHA" \
  --feedback "$FEEDBACK" --feedback-sha "$FEEDBACK_SHA"
```

F1 실측은 전체11080초(약3시간5분), BF9613초(약2시간40분),
최대 RSS 약5.1GiB였다. F2 완료 보장은 아니며4시간에는 약55분의 여유가 있다.
F1 단회 승인 소진을 F2 승인으로 재해석하지 않았으며 이번 사용자 추가 승인을
별도 사용했다.

## 승인된 실제 실행

- root basename: `lumina_F2_response_lageunha_20261028-bBbs8U`
- 시작: 2026-10-28 11:59:31 KST. 빌드 포함4시간 제한, 강제 종료 유예30초.
- 소스: 직접시험한 `36c8aa14bef236d7ba03442bfe8a3df0920bb56c`을 별도 트리에
  봉인한다. 이후 공유 저장소 변경은 실행 중 소스에 반영하지 않는다.
- preparation SHA256:
  `4707762a819fee7f013fb8655ad5d6a738c9f64cfd1ddc897e5138caa9478887`
- 완성 F2 feedback SHA256:
  `e4229694191e6dfbc338bd1edb928c75c17516989262a05c7802471cbf986e5c`
- 입력 J SHA256:
  `fc66fc815d7e8c67028c269944e01227611497f7bdfa129bff863a18c1f69308`
- 입력 moments SHA256:
  `88b0bd627ee3b2a2fb72252a5fed4d0496c2f8abb62b42020b53861a0348f1a4`
- 이 continuation의 로컬 폴더 이름은 `F-1`이나 feedback의
  `actual_physical_sweep=2`인 **실제 F2**다. 기존 F1 응답을 재사용하지 않는다.
- 로그는 `manual.log`, 진행/완료 기록은 `run/result.json`, 비교는
  `run/Si-S-comparison.csv`다. 정확한 내부 입력 주소는 원 feedback과 결과에 보존한다.
- 실행 전 LagEunha load0.61/0.23/0.90, 가용 메모리 약983GiB였고 같은 응답
  실행 프로세스는 없었다. 새 Slurm 작업·GPU 수송·다른 프로젝트 변경은 없다.
- 첫 실제 확인에서 Python PID3564426이 실행 중이었고 CPU 사용이 관측됐다.
  `result.json`은 실제 `manual-lageunha`·affinity0–15·주소공간32GiB를 기록했고,
  위 F2 J/모멘트와 수용 물질 계보 확인을 통과한 `NEW_LINEAR_BB` 단계였다.
  같은 결과의 최대 RSS 필드는 저장 시점의 값이며 현재 프로세스 RSS와 구분한다.
- F1 실측을 대입한 예상 완료는 약15:05 KST다. 같은 실행시간을 보장하지 않으며
  빌드 포함 제한은 약15:59:31 KST다. 실제 종료·응답 비교를 확인해야 완료다.

## 13:58 KST BF 완료 후 상태

동일 PID3564426의 전주파수 BF 전적분1회가7007.6597초에 완료됐다.
`NEW_FIELD_COMMON_CHARGE`로 전환됐으며 generation11의 실제 동적 물질
평가가 시작됐다. 이 관측 시점의 완료 평가0개·전하 수렴 미판정 상태를 보존한다.
저장 기록의 peak RSS는4321032KiB(약4.1GiB)이며 실행 오류는 없다.
진행 기록의 wall_seconds는 마지막 저장 시각의 값이므로 현재 경과시간과
동일시하지 않는다. 프로세스의 실제 CPU 시간 증가와 함께 확인했다.

F1과 비슷한5회 평가라면 남은 약20–30분을 예상하지만, 반복수와 비용을
보장하지 않는다. 앞선15:05 KST 예상은 BF 완료 전의 실측 외삽 이력이다.
입력·소스·자원·횟수·4시간 상한·원본 보존 규칙은 바꾸지 않았다.
이 전환은 최종 이온분포/RE/산란 수렴 또는 CMFGEN 동질성 완료가 아니다.

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
