최종 판정: **FAIL-TOPOLOGY**. 소스 정적 재검만 수행했으며 수정·실행·git은 하지 않았습니다.

1. **행렬 identity/checksum·채널 완전성 — FAIL**

   - checksum은 외부 기준 checksum과 비교하지 않고, 조립 전후의 자기 checksum만 비교합니다. 따라서 “동일 ARTIS projection/frozen state”를 증명하지 못합니다: [lumina_element_wide.c:1142](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1142), [lumina_element_wide.c:1180](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1180).
   - generic `col_ion_*` collision 자료는 실제 조립에 소비되지만 checksum에는 포함되지 않습니다: [lumina_plasma.c:15385](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15385), checksum 종료 [lumina_element_wide.c:602](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:602).
   - sigma가 없는 continuum은 expected 분모에서도 제외되어 local `100%`가 자료 완전성을 뜻하지 않습니다: [lumina_element_wide.c:487](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:487).
   - 채널 inventory gate는 7개 중 네 채널과 조건부 NT-BF만 검사합니다. NT-BB와 AUTOION/DR의 기대 활성 여부는 gate에 없습니다: [lumina_element_wide.c:1238](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1238).

2. **ARTIS target 구조 등가성·합성 Kramers 제거 — PASS(소스 구조 한정)**

   - CSR의 모든 target route를 순회하며 target별 threshold와 `g_upper`를 다시 계산합니다: [lumina_element_wide.c:266](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:266), [lumina_element_wide.c:303](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:303).
   - route probability는 네 bf 방향에 정확히 한 번 적용됩니다: [lumina_element_wide.c:369](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:369).
   - sigma 없는 level은 EW에서 비활성 처리되고, producer가 legacy Kramers 블록 전에 `continue`합니다: [lumina_element_wide.c:255](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:255), [lumina_plasma.c:15551](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15551).
   - 이는 구조 구현 PASS이며, 실제 ARTIS matrix support/rate oracle PASS를 의미하지는 않습니다.

3. **fail-closed 실기준선 — FAIL**

   - EW gate가 켜지면 후보 검증 전에 전역 NLTE layout이 31→33 slot으로 바뀝니다: [lumina_plasma.c:13973](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:13973).
   - 이에 따라 S/Fe IV line이 모든 셸에서 NLTE line으로 매핑됩니다: [lumina_plasma.c:14103](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14103).
   - 후보 실패 시 pair solver는 다시 실행되지만 layout은 복구되지 않습니다: [lumina_plasma.c:17099](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17099).
   - 이후 모든 새로 매핑된 IV line의 opacity/source가 전 셸에서 갱신됩니다: [lumina_plasma.c:16847](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16847), [lumina_plasma.c:16904](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16904). 따라서 fallback은 실제 31-slot 기준선이 아닙니다.

4. **발화계측 정직성 — FAIL**

   - `clamp_floor_freeze_anchor_repair_fallback_count`는 실제 각 site 계측이 아니라 네 내부 카운터의 합일 뿐입니다: [lumina_element_wide.c:1236](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1236).
   - manifest의 pair-owner/save-restore/pin/topstage 호출 수는 계측값이 아니라 literal `0`으로 기록됩니다: [lumina_element_wide.c:1282](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1282).
   - `kramers_fallback`은 선언·합산·출력되지만 증분 site가 없습니다: [lumina_element_wide.c:194](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:194), [lumina_element_wide.c:207](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:207). 현재 경로상 Kramers가 우회된 것은 맞지만, 독립 발화 계기는 아닙니다.

5. **§7 clamp C48/C65 처분 — FAIL**

   - **C65는 적절히 우회**됩니다. `STAGE4` 동시 설정을 거부하고: [lumina_element_wide.c:101](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:101), stage4 `b_k` cap 소비점에서도 EW 대상은 제외됩니다: [lumina_plasma.c:10030](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10030).
   - 그러나 **C48 `SUPER_CUTOFF`은 실제 atomic projection을 변경**합니다: [lumina_atomic.c:761](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:761).
   - EW는 이를 자동으로 활성 SL layout에 반영하고: [lumina_plasma.c:14087](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14087), 단지 projected-level 수를 기록할 뿐 PASS gate에서 금지하지 않습니다: [lumina_element_wide.c:1128](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1128), [lumina_element_wide.c:1247](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1247). §7의 C48 발화 0 계약을 만족하지 않습니다.

6. **SVD/rcond·scaled residual — PASS(명시된 세 항 한정)**

   - normal equation이 아닌 full Golub–Reinsch SVD입니다: [lumina_element_wide.c:702](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:702).
   - `rcond_1`은 `||A||₁`과 LU로 구한 실제 inverse-column norm을 사용합니다: [lumina_element_wide.c:775](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:775).
   - scaled residual은 물리 `Araw,x`에서 `max(inflow,outflow,n_i/t_ref)` 분모로 계산되고 PASS gate에 연결됩니다: [lumina_element_wide.c:1056](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1056), [lumina_element_wide.c:1258](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1258).
   - 별도 잔여: equilibration은 ARTIS의 `f=sqrt(col_norm/row_norm)` 유사변환이 아니라 독립 row/column 정규화입니다: [lumina_element_wide.c:855](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:855).

`SCOPE_FAIL` 분리 정당성은 **FAIL**입니다. 분기 자체는 topology/numerics가 통과하고 boundary만 실패한 경우로 잘 분리됐습니다: [lumina_element_wide.c:1264](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1264). 하지만 boundary 증거는 Lumina의 I/V population과 Sobolev opacity뿐이며 rate/heating 및 비교 모델 자료의 합집합을 검사하지 않습니다: [lumina_element_wide.c:1214](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1214), [lumina_element_wide.c:1231](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:1231). 따라서 `SCOPE_FAIL` 라벨의 분기 논리는 타당하지만, 반대로 `EW_PASS`를 허용할 만큼 scope coverage가 완결되지 않았습니다.