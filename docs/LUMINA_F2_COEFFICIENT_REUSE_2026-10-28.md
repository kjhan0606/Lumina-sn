# 새 자체장의 T/ne 응답을 위한 완료 F2 계수 재사용

## 목적과 현재 범위

최종 목표는 같은 물질·복사장의 공동수렴과 CMFGEN 물리량·스펙트럼
동등성이다. 기존 Astra의 감쇠 물질 블록 → 계수 → 산란 수송 방향을
유지한다. 이번 변경은 고정 새 장의 전90점 온도·공통 전자밀도 보정에서
이미 완료한 기준 BF 전적분을 되풀이하지 않기 위한 준비다. 새 전역 물질
계산, F3 수송, 온도 수렴 또는 최종 목표 달성으로 기록하지 않는다.

중요한 물리 가정·해법 변경과 핵심 동등성 판단은 Astra에 묶어서 맡긴다.
이번 계수 재사용은 기존 자문의 구현이며 같은 질문으로 재호출하지 않았다.
일상적인 구현·직접시험은 운전자가 담당한다.

## 소비 계약과 보존

`lumina_new_field_response.py`의 기본 경로는 유지한다.
`--temperature-evaluations 5..12`와 `--resume-result`, `--resume-result-sha`를
함께 지정한 경우에만 완료 고정온도 응답을 읽는다. 봉인 결과의 장·수용
물질·원자자료·기준 T/ne, 계수 생산 소스·바이너리와 실제 BF/BB/FF
식별자·경로 배열을 확인한다. 서로 다른 장의 계수나 온도가 바뀐 BF율을
고정온도 캐시로 사용하는 근사는 허용하지 않는다.

재사용 대상은 `bound-rates.npy`, `route-roster.npz`, `jbar.npy`,
`line-status.npy`, `FF-coefficients.npz`다. 새 출력 디렉터리로 복사하며
이전 원본은 수정하지 않는다. bound율은 원 수용 물질의 기준율이다.
완료 응답의 최종 ne는 시작 추정값으로만 쓰고, 현재 코드의 실제
`ModelResponse.evaluate`와 `ResponseBalance.evaluate`를 반드시 거친다.
저장 열장부를 새 후보의 열장부로 대입하지 않는다. 전하가 닫히지 않으면
기존 전하 초기화가 수행된다. 이후 온도 변경 BF는 기존 전체 적분을 한다.

사용하는 완료 F2 root는 `lumina_F2_response_lageunha_20261028-bBbs8U/run`,
결과 SHA256은
`b6d411ff28971492280275d2201e9d84a65315c0ad2613f04444e7b270b5fbea`다.
오류 시 새 부분 결과를 보존하며 이전 F2와 참조 자료는 변경하지 않는다.
선택 기능을 제거해도 기존 기본 경로와 원 결과는 유지된다.

## 직접시험과 실물 입력 확인

Lageunha CPU1 affinity·주소공간4GiB·외부150초, GPU0에서 합성 회귀시험은
**88 passed / 3.33초**였다. 잘못된 보고서 해시, 원자자료·장·물질·생산자
혼합, 손상 파일, 비유한 전하·선율, 잘못된 배열/온도, 재개 인수 누락을
거부했다. 기존 열보정·깊이 결합·전하·BF 회귀도 함께 통과했다.

격리 소스·시험 root는 `lumina_completed_seed_tests_20261028-xyj3Nh/source`다.
그 소스에서 다음 명령으로 재현한다. 서버와 위 CPU·메모리·시간 한도를
유지하며 로그인 노드에서 수치 시험하지 않는다.

```bash
python3 -m pytest -q -p no:cacheprovider \
  tests/test_lumina_completed_field_seed.py \
  tests/test_lumina_new_field_response.py \
  tests/test_lumina_fixed_field_thermal.py \
  tests/test_lumina_coupled_material.py \
  tests/test_lumina_charge_projection.py \
  tests/test_model_material_rates.py
```

실제 F2의90점·1,115,020주파수 파일을 같은 서버 CPU1·4GiB·180초로
읽기 전용 확인했다. **30.84초**에74개 입력/소비 파일의 신원, BF/BB/FF
식별자, 전 BF 경로 배열, 기준 occupation의 정확한 일치를 확인했다.
실제 선·BF·FF 생산자로 식별자와 기준 trial을 재구성했으나 새 BF 적분,
SE 해, 열평가 또는 수송은 수행하지 않았다. 이는 실물 계수 소비 확인이며
전역 물질응답 전체 실행의 통과 증거가 아니다.

보존 기록은 시험 root의 `actual-seed-verification.json`이다. 검증한
`lumina_new_field_response.py`의 SHA256은
`a71b680bbf3c763900a88f09f33e431c7b1b6ec9c4c8e2babd0daee22ec77554`,
`lumina_completed_field_seed.py`는
`b75a87cb9dbf3fdb7291c6eb5bee35d6a73efa9d695dc5ddc948f5a4f97d90e8`다.
첫 읽기 전용 확인 명령은 입력 경로 문자열을 `Path`로 바꾸지 않은 운전자
명령 오류로 중단됐고, 명령을 수정해 위 확인을 완료했다. 물리 재실행은 없다.

## 다음 실제 계산의 시간 판단

초기 BF 전적분 실측7007.66초와 선율/FF 재생산을 줄일 수 있다. 하지만
최소 Newton 시도5평가에서도 T+/T−와 온도가 이동한 방향 후보1개의
**BF 전적분 약3회**가 남는다. ne+/ne−는 첫 기준 온도에서만 기존 캐시를
사용한다. 한 번 온도가 이동하면 다음 반복의 ne 섭동 비용도 달라진다.

Lageunha의 BF 실측만 합쳐 약5시간50분, grammar의 이전 실측9613.37초를
적용하면 약8시간1분이다. 기준 SE/열 재평가, 두 ne 평가, FF·보존 비용을
더해 첫 시도는 **대략6–9시간 규모**로 예상한다. 이는 과거 실측을 사용한
추정이지 새 온도의 보장 상계나 최종 공동수렴 완료 시간은 아니다.

권고 실행 범위는 Lageunha CPU16·32GiB·최대12시간·기준 재평가와 열보정
최대5평가·단회·재시도0·GPU0·새 수송0이다. 첫 방향 후보가 거부되면
수용 상태와 모든 섭동 결과를 보존하고 자동 반복하지 않는다. 기존4시간
단회 권한과 실행기 한도는 확대하지 않았으며 이 새 범위를 실행하지 않았다.
실제 실행 전 별도 자원·시간·횟수 권한이 필요하다. 후보와 내부 잔차의
변화는 CMFGEN70 분율에 가까워지는지와 분리해 평가해야 한다.

## 후속 사용자 승인

이후 사용자 「한도 승인함」으로 위 CPU16·32GiB·최대12시간·단회·
열보정 최대5평가 범위를 승인했다. 운영 판단은 운전자가 맡고 Astra에는
비용 대비 효용 자문을 더 의뢰하지 않는다. 최신 실행 범위와 보존 규칙은
`LUMINA_F2_THERMAL_RUN_2026-10-28.md`를 따른다. 이 승인은 평가 횟수
확대나 F3 수송·다른 추가 계산을 자동 승인하지 않는다.
