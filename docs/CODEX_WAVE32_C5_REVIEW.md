# Codex C5 — A5 패치 리뷰

최종 판정: **FAIL**. Rung1만 완전 해소로 인정하며, Rung2–4에는 소스 수준의 잔여 우회가 있습니다.

검토 범위는 지정된 네 패치뿐입니다. 작업 트리, B5, C4 문서 원문은 열거나 실행하지 않았습니다. 아래 인용은 패치 파일 자체의 행 번호입니다.

1. **[PASS] Rung1 — I/O 오류 분류**

`ew_finish_file()`이 `fflush`, `ferror`, `fclose`를 모두 검사하고 오류를 보존합니다([rung1:31](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung1_ew_io_fail_closed.patch:31)). 패치에 노출된 모든 직접 종료 경로도 공통 완료 검사로 교체되었습니다([rung1:80](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung1_ew_io_fail_closed.patch:80), [rung1:107](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung1_ew_io_fail_closed.patch:107), [rung1:125](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung1_ew_io_fail_closed.patch:125)). `/dev/full` 음성 대조도 동일 production helper를 사용합니다([rung1:146](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung1_ew_io_fail_closed.patch:146)).

2. **[FAIL] Rung2 — atomic 재스윕**

19개 정수 telemetry counter는 공통 atomic helper로 전환됐습니다([rung2:18](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch:18), [rung2:38](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch:38)). 그러나 file-static `ew_cap` 아래 production 경로에 다음 비-atomic 누적이 그대로 남습니다.

```c
ew_cap.expected_outflow[channel][j] += rate;
```

근거: [rung2:16](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch:16), [rung2:94](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch:94). 패치는 “separate ownership”이라는 주석만 제시하며, thread-local 저장소·동기화·소유권 강제 코드는 제시하지 않습니다. 추가 self-test도 helper의 19개 정수만 병렬 검증하고 이 production 누적은 시험하지 않습니다([rung2:291](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch:291)). 따라서 패치 단독으로 병렬 영역의 잔여 비-atomic 증가가 없다고 판정할 수 없습니다.

3. **[FAIL] Rung3 — override 계약의 은닉 우회**

CLI 경로는 비계약 기대값을 기본 거부하고([rung3:35](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch:35)), 명시적 override를 `NON-CONTRACT`와 종료 코드 2로 표시합니다([rung3:55](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch:55), [rung3:68](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch:68)).

하지만 공개 Python 함수 `check_artifact(..., non_contract_override=True)`는 계약 상태가 태깅되지 않은 기존 결과 tuple을 그대로 반환하며, `NON-CONTRACT` 표시와 RC=2는 CLI의 호출 후 처리에만 존재합니다([rung3:27](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch:27), [rung3:60](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch:60)). 테스트 역시 CLI만 검증합니다([rung3:91](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch:91)). 직접 import하는 소비자는 비계약 결과를 정상 결과와 구별하지 못하므로 은닉 우회가 남습니다.

4. **[FAIL] Rung4 — NaN 안전 감사의 전체 적용**

행렬·보존행·flux 결과에는 `isfinite` 검사가 추가됐습니다([rung4:8](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:8), [rung4:27](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:27), [rung4:61](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:61), [rung4:82](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:82)).

그러나 다음 비교는 그대로입니다.

```c
(ip >= 0 && n_elem > 0.0) ? density / n_elem : 0.0
```

`n_elem == NaN`이면 비교가 거짓이 되어 `f=0.0`으로 정상화되고, 뒤의 `isfinite(f)` 검사를 우회합니다([rung4:98](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:98)). 또한 유한한 `tau`들의 합이 overflow한 뒤 `tau_all`만 무한대가 되면 `tau_boundary / tau_all`이 0으로 축소될 수 있는데, `+=` 직후 유한성 검사가 없습니다([rung4:103](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:103)). 추가 fixture도 `b[0]`, flux 입력, matrix NaN만 seed하고 이 두 경로는 다루지 않습니다([rung4:126](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:126), [rung4:138](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch:138)).

부가 불변식 판정:

- 신규 물리 clamp: **0**
- 신규 identity 변형: **0**

`INFINITY` 반환·대입은 실패 폐쇄 sentinel이며 물리값 clamp가 아닙니다. 기존 `nstar_cap`도 “rejection; never capped” 계수만 atomic화했습니다([rung2:227](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch:227)).