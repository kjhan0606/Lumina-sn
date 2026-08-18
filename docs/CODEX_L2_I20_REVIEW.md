검수 중 운전석의 별도 커밋 `1502056`이 추가되어 최종 HEAD 기준으로 읽었습니다. 저는 파일 수정·삭제·커밋을 하지 않았습니다.

### Q1 — OK

근거: [cmfgen_parser.py:147](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:147), [cmfgen_parser.py:158](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:158), [cmfgen_parser.py:187](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:187), [expand_atomic_data_cmfgen.py:649](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:649), [expand_atomic_data_cmfgen.py:663](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:663), [expand_atomic_data_cmfgen.py:703](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:703), [genosc_v6.f:187](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:187), [genosc_v6.f:287](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:287) — 파서가 `ID == n_loaded+1`인 행만 `levels[n_loaded]`에 넣고 `per_ion_g[k]`도 같은 `levs[k]`에서 채우므로, transition의 1-based `i,j`에 대한 `gs[i-1], gs[j-1]`은 정확히 같은 base이다.

### Q2 — DEFECT

근거: [expand_atomic_data_cmfgen.py:574](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:574), [expand_atomic_data_cmfgen.py:711](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:711), [genosc_v6.f:223](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:223), [genosc_v6.f:313](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:313), [L1_R5_REPAIR_ACCEPTANCE.md:23](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/layer1/L1_R5_REPAIR_ACCEPTANCE.md:23) — 현재 핀 덱에서는 선 수가 같지만 조건은 동치가 아니어서, 유한·비영 `lam_A`이면서 `dE_cm<=0`/비유한이면 신 코드만 더 버리고, `lam_A`가 NaN/∞이면서 유한 양의 `dE_cm`이면 구 코드만 버리며, 특히 CMFGEN은 역전 에너지를 경고만 하고 차이를 제곱하므로 신 필터는 일반 입력에서 CMFGEN과 달라진다.

### Q3 — OK

근거: [genosc_v6.f:278](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:278), [genosc_v6.f:303](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:303), [genosc_v6.f:305](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:305), [genosc_v6.f:344](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v6.f:344), [expand_atomic_data_cmfgen.py:716](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:716), [lumina_atomic.c:1249](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1249) — CMFGEN은 원본 음수 `f`를 먼저 절댓값으로 만든 뒤 필요하면 자체 GF 필터 결과만 다시 음수화하며, Lumina 런타임은 `f_lu`만 읽고 `f_ul`은 읽지 않으므로 원본 부호를 보존할 소비자는 없다.

### Q4 — DEFECT

근거: [finalize_cmfgen_ref_npy.py:84](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:84), [finalize_cmfgen_ref_npy.py:94](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:94), [finalize_cmfgen_ref_npy.py:108](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:108), [finalize_cmfgen_ref_npy.py:111](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:111), [finalize_cmfgen_ref_npy.py:188](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:188), [finalize_cmfgen_ref_npy.py:220](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:220), [finalize_cmfgen_ref_npy.py:224](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:224), [lumina_plasma.c:10933](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10933) — 상세평형·`f_ul`·NPY·일반 τ 경로는 각각 명시적 B/f/endpoint를 쓰지만, RadEq ETLA 시험-τ는 아직 `1.4992e-16*A_ul*(g_up/g_lo)*lambda²`로 `f_lu`를 재구성하여 원본 f 대신 A 유래임을 가정하고, 상수 반올림만으로도 약 `4.15e-6` 차이가 난다.

### Q5 — DEFECT

근거: [expand_atomic_data_cmfgen.py:574](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:574), [expand_atomic_data_cmfgen.py:653](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:653), [expand_atomic_data_cmfgen.py:662](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:662) — `lam_A != 0`가 선 포함 여부를 결정하고 원본 `t['A']` 합이 `levels.csv.metastable`을 결정하므로 raw λ/A 소비가 두 곳 남아 수리는 계약상 불완전하다.

지정한 다른 경로에서는 raw osc λ/A 소비를 찾지 못했다: sigma는 준위·phot를 사용하고([expand_atomic_data_cmfgen.py:1507](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:1507)), macro-atom은 수리된 `L['A_ul']`을 사용하며([expand_atomic_data_cmfgen.py:871](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:871)), zeta는 복사/levels 기반이고([expand_atomic_data_cmfgen.py:1693](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:1693)), ma_radrecomb는 phot와 상위 이온 준위만 사용하며([build_ma_radrecomb_target.py:191](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_ma_radrecomb_target.py:191)), `merge_*`와 patch는 생성된 CSV만 소비한다([merge_iron_peak_III_bb.py:62](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/merge_iron_peak_III_bb.py:62), [merge_as_planE_into_capraise.py:274](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/merge_as_planE_into_capraise.py:274), [merge_fe2_m1e2.py:360](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/merge_fe2_m1e2.py:360), [patch_transprob_aul_weighted.py:49](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/patch_transprob_aul_weighted.py:49)); `src/`도 line-list 열만 적재한다([lumina_atomic.c:1249](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1249)).

### Q6 — DEFECT

근거: [expand_atomic_data_cmfgen.py:337](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:337), [expand_atomic_data_cmfgen.py:340](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:340), [expand_atomic_data_cmfgen.py:349](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:349), [expand_atomic_data_cmfgen.py:351](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:351), [expand_atomic_data_cmfgen.py:404](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:404) — 경로의 마지막 `atomic|atomic_local` 뒤가 선택적 `cmfgen/` + 알려진 원소 + 알려진 Roman stage + 정규식 `\d{1,2}[a-z]{3}\d{2}` vintage + 최소 한 후속 성분이면 루트 위치와 추가 하위경로를 묻지 않으며, 실제 파일·4종 링크·동일 이온 조건만 만족하면 임의 디렉터리의 파일도 통과한다.

### Q7 — DEFECT

근거: [deck_regen.py:40](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen.py:40), [L1_R5_REPAIR_ACCEPTANCE.md:28](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/layer1/L1_R5_REPAIR_ACCEPTANCE.md:28), [L1_R5_REPAIR_ACCEPTANCE.md:34](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/layer1/L1_R5_REPAIR_ACCEPTANCE.md:34) — `macro_atom_data.csv`를 검증된 “macro_atom”으로 보아도 다음 18개 재생성 산출물은 미대조다:

`atom_masses.csv`, `atomic_data_cmfgen.h5`, `atomic_vintage_manifest.csv`, `cmfgen_sigma_bf.bin`, `coldata_cmfgen_manifest.csv`, `ionization_energies.csv`, `level_multiplicity.csv`, `line2macro_level_upper.npy`, `kshape_contract.txt`, `ma_radrecomb_target.bin`, `ma_radrecomb_target_manifest.csv`, `macro_atom_references.csv`, `transition_probabilities.npy`, `tau_sobolev.npy`, `verification.log`, `zeta_data.npy`, `zeta_ions.csv`, `zeta_temps.csv`.

또한 G5는 선 수·이온 집합만 보고([L1_R5_REPAIR_ACCEPTANCE.md:23](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/layer1/L1_R5_REPAIR_ACCEPTANCE.md:23)) endpoint 4-tuple 집합과 세 line-id 연동 산출물의 의미적 정합성을 확인하지 않았고, 비오염 13이온의 `≤1e-8` 조건은 검증되지 않은 채 기대치 자체가 틀렸음이 사후 확인됐다([L1_R5_REPAIR_ACCEPTANCE.md:55](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/layer1/L1_R5_REPAIR_ACCEPTANCE.md:55)).

### Q8 — DEFECT

추가로 확인된 결함은 다음과 같다.

- [expand_atomic_data_cmfgen.py:873](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:873)는 static macro internal-up B를 `1/(8πh)`로 만들지만 finalize의 Jν 규약은 `1/(2h)`이고([finalize_cmfgen_ref_npy.py:84](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:84)), finalize는 그 raw weight를 재계산하지 않고 정규화만 하므로([finalize_cmfgen_ref_npy.py:224](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/finalize_cmfgen_ref_npy.py:224)) up/down 혼합 블록의 static internal-up 확률이 `4π`만큼 작다.

- 오프라인 검증기는 `A_new_cmfe`와 `A_cmf`를 같은 식으로 만들고, `f_new=f_src`, G3/G4도 자기 식과 자기 식을 비교한다([l1_r5_repair_offline_check.py:69](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/l1_r5_repair_offline_check.py:69), [l1_r5_repair_offline_check.py:77](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/l1_r5_repair_offline_check.py:77), [l1_r5_repair_offline_check.py:88](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/l1_r5_repair_offline_check.py:88)); 더구나 PASS 식에는 G5 구 덱 선 수 비교가 전혀 없다([l1_r5_repair_offline_check.py:109](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/l1_r5_repair_offline_check.py:109)).

- 현재 `deck_regen.py`는 `verification.log`를 재생성 대상으로 분류해 복사를 막지만([deck_regen.py:47](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen.py:47), [deck_regen.py:65](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen.py:65)) 어떤 단계도 이를 만들지 않은 채 reference 파일 완전성을 검사하므로([deck_regen.py:132](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen.py:132), [deck_regen.py:165](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen.py:165)), 기본 reference에 해당 파일이 있는 현재 상태에서는 생성기가 완결 판정에 도달할 수 없고 그 reference 로그 자체도 ERROR다([verification.log:1](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/verification.log:1)).

- 범용 생성기는 `level_multiplicity`에 덱 경로만 넘기지만([deck_regen.py:140](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen.py:140)), 해당 빌더는 여전히 `cmfgen_config_lumina.yml`의 자체 vintage를 선택한다([bake_level_multiplicity.py:91](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/bake_level_multiplicity.py:91), [bake_level_multiplicity.py:154](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/bake_level_multiplicity.py:154)).

## 가장 중요한 발견 3개

1. **Q5:** raw `lam_A` 선 필터와 raw `A` 기반 metastable 판정이 남아 있어 “λ/A 열 무소비” 계약이 아직 성립하지 않는다.
2. **Q4:** RadEq ETLA 시험-τ 경로가 보존된 원본 `f_lu`를 쓰지 않고 `A_ul`에서 다시 역산한다.
3. **Q8:** static macro internal-up 확률의 B 규약이 finalize와 `4π` 불일치하며, 오프라인 게이트는 대부분 자기동일 비교라 이 결함을 검출할 수 없다.