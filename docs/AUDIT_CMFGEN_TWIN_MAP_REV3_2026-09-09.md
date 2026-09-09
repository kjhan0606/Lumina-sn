# 운전자 사전감사 — DET-CMFGEN-TWIN-MAP 계약 개정 3

- 감사자: 운전자
- 최종 재감사 시각: `2026-09-09T10:45:51+09:00`
- 대상 기안: `docs/AMEND_CMFGEN_TWIN_MAP_REV3_2026-09-09.md`
- 대상 SHA-256: `1edabed5b8430d8f6e9a982d638c48d6c96d9604bc6eb58b685193f6e65759ae`
- 적용 전 계약: `docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md`
- 적용 전 계약 SHA-256: `b71f4b3594c11bcbebbde5243120852f2d189b3ee2ecf75f4ba2b3cc7e0e8595`
- 감사 범위: 계약 개정 설계·물리 가정·제외·수용기준·증거 계획·자원 상한
- 감사 결과: **Q-GOAL=`PASS`, Q-LEAN=`APPROPRIATE`, 전체=`PASS`**

이 PASS는 위 SHA-256의 계약 개정 설계에만 적용한다. 구현·selftest·runner 자격·staging·GPU
실행·물리 결과·Opus5 완료감사를 통과시킨 판정이 아니다.

## Q-GOAL — PASS

1. **최종 목적에 닿는 칸:** 개정 3은 CMFGEN 대비 50개 paired 셸에서 여섯 원소의 Saha/NLTE
   단계 분율, 이온 천장, NLTE 권위비와 벌크 밀도 정규화 평균전하 차이를 채운다. 이 지도는
   D-C-2와 D-D 중 어느 물리 수리를 먼저 시험할지 정하는 데 직접 소비된다.
2. **측정인가 정비인가:** 중점 최근접 표본, 질량 규약, 천장 의미와 집계식의 수정은 계측기의
   비교 자격을 고치는 오프라인 정비다. 등록 GPU 1회가 내는 `Δf`, `R_SN`, `d_Z`, `M_Z`는
   관측이다. 개정이나 실행 자체를 물리 수리 성공으로 부르지 않는다.
3. **더 가까운 대안:** D-C-2와 D-D는 최종 물리 수리에 더 가깝지만 어느 항을 우선할지에 필요한
   같은 paired 지도를 이미 제공하지 않는다. 종전 BASE의 `d_e`는 위치·벌크 밀도·질량 규약이
   섞여 등록 비교량의 대안이 될 수 없다. 추가 CMFGEN 런이나 BASE 재런도 필요하지 않다.
4. **건너뛰면 잃는 것:** 표본 위치 인공물과 벌크 밀도 배율을 분리한 평균전하 지도, IGE 질량
   규약 점검, 최고 단계 누적통 처리, NLTE 준위 피복과 권위 간극, 개입 크기별 개선 방향을 잃는다.
   그러면 후속 수리의 우선순위를 기존의 교란된 중앙값으로 정하게 된다.

따라서 목적 연결, 단계 성격, 대안 부재와 생략 비용이 모두 명시되어 **Q-GOAL=`PASS`**다.

## Q-LEAN — APPROPRIATE

### L1 — 각 계측의 소비 지점

| 계측 | 직접 소비 |
|---|---|
| 중점 최근접 anchor와 `d_rho` | 위치 정합 확인과 셸 pairing |
| `N_Z`, `n_atom,model`, CMFGEN `n_atom` | 분율 분모·질량 규약 확인·`⟨Z⟩` |
| 직접/누적 최고 단계와 중성 원값 | 유효한 `Δf` 좌표와 Saha 닫힘식 |
| NLTE 준위 키·`n_ion_total`·`R_SN` | 덤프 결손과 Saha/NLTE 권위 간극 분리 |
| `d_Z`, 질량가중치, `S_Z`, paired sign census | E-Z 판정과 공간적 부호 구조 |
| 사전고정 `|ΔT|` 5층 | 개입이 큰 셸에서의 개선·악화 노출 |
| 원 `n_e`·`d_e` | 벌크 밀도 배율이 포함된 역사적 대조와 잔차 분해 |

소비되지 않는 신규 물리량은 없다.

### L2 — 게이트 중복

개정 3은 독립 facts 벡터, 자동 물리 gate, 자동 TM 분기, 새 증거 장부를 추가하지 않는다.
닫힘·행 합·질량·좌표·세대 검사는 기존 원자료 판독의 작동 확인이며 독립 감사 단계가 아니다.

### L3 — 단계 구성

하나의 과학 질문과 하나의 rollback 경계를 가진 묶음이다. 계약 적용 → sol 구현·직접 selftest →
운전자 코드 검수 → 오프라인 준비·runner 자격 → 등록 GPU 1회 → 기존 9개 산출물 판독 → Opus5
완료감사·운전자 재현으로 끝난다. 별도 획득런이나 소질문별 감사로 쪼개지 않는다.

### L4 — 삭제 후보와 처분

- 공간 상관 셸을 독립 표본으로 취급하면서 판정에는 쓰지 않던 sign-test p값을 삭제했다.
- `anchors.csv`와 물리 지도에 중복 기록하려던 `n_atom`을 anchor에서 삭제했다.
- 잘못된 최고 단계 직접 비교와 누적 비교의 중복 좌표를 삭제했다.
- E-Z로 교체되는 종전 E-NE 판정 문단 전체를 교체 범위에 넣어 신·구 판정식의 병존을 막았다.
- 새 runner·새 산출 파일·새 증거 장부·새 gate·새 획득런은 만들지 않는다.
- 기존 9개 파일은 입력 identity, 좌표, loader 입력, BASE 대조, writer 세대, 사전식, 자원,
  셀 원값, 집계·증거 참조로 각각 소비되므로 더 줄이면 수용기준이나 재현성이 사라진다.

따라서 남은 범위는 목적에 필요한 최소 묶음이며 **Q-LEAN=`APPROPRIATE`**다.

## 설계 결함 점검과 감사 중 수정

1. Lumina의 `lumina_plasma_state.csv`는 `shell_id,T_e,n_e`만 공개한다. 기안의 초기 “공개
   `n_atom`” 전제를 폐기하고, 봉인 조성 전체에서 산출한 `n_atom,model`과 CMFGEN 공개
   `n_atom`의 출처를 분리했다.
2. 완전 단계 닫힘식은 전체 Saha 생산 모집단에만 적용하고 II·III 준위만 내는 NLTE에는 적용하지
   않도록 고쳤다.
3. 참조 최고 단계가 누적통이면 Lumina 최고 개별단계와 직접 비교하지 않고, 의미가 확인된 경우에만
   `q≥Q_Z`를 합친다. 의미가 미확인이면 fail-closed 상태로 남긴다.
4. `⟨Z⟩`는 벌크 밀도의 직접 배율을 제거하지만 전리 상태 자체의 밀도 의존성을 없애지는 않는다고
   주장 상한을 바로잡았다.
5. IGE는 toy06의 56Ni→56Co→56Fe 사슬에 맞춰 `A=56`을 쓰고, `n_atom,model`은 비교 대상
   여섯 원소가 아니라 봉인 조성 전체를 합산하도록 명시했다.
6. 개정 2의 E-NE·최소속도 표본·astra 분장이 남아 있던 현행 운영 문단과 재현표를 E-Z·중점
   최근접 표본·운전자 분장으로 바꿨다. 과거 집행 기록 속 당시 표현은 역사 증거로 보존했다.

위 수정 뒤 개정안의 `OLD` 블록 36개는 적용 전 계약에 각각 정확히 한 번 존재한다. 감사 PASS를
개정안의 상태표와 현행 운영 문단에 반영한 뒤 최종 해시에 대해 같은 문면·Q-GOAL·Q-LEAN과 쌍
identity를 다시 확인했으며 결론은 변하지 않았다.

## 실행 수용기준

- 개정안 36개 `OLD`/`NEW` 교체가 정확히 적용되고 기존 집행 기록은 보존될 것.
- 50개 셸 모두 결과 비의존 중점 최근접 anchor가 존재하며 보간 없이 결정적으로 재현될 것.
- 양쪽 `n_e`, Lumina 산출 `n_atom,model`, CMFGEN 공개 `n_atom`의 출처와 정의역이 분리될 것.
- IGE `A=56`, 전체 조성 합, `d_rho`, 참조 행 합, Saha 닫힘, 최고 단계 의미, NLTE 키와
  `R_SN`이 원값에서 재현될 것.
- 전 50셸의 `d_Z`가 유효할 때만 질량가중 E-Z를 판정하고, 결손 셸을 버린 재정규화는 금지할 것.
- 기존 9개 증거파일, GPU 최대 8 GPU-hour, 런 CPU 최대 128 할당 CPU-hour, 오프라인 단일 CPU
  누적 1 CPU-hour, 신규 16 GiB, 등록 GPU 1회 상한을 지킬 것.
- 물리 출력은 관측 등급에 머물고 자동 TM 판정은 생성하지 않을 것.

하나라도 충족되지 않으면 staging 또는 실행을 멈추고 원값을 보존한다.

## 재현 명령

```bash
sha256sum docs/AMEND_CMFGEN_TWIN_MAP_REV3_2026-09-09.md
git show HEAD:docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md | sha256sum

perl -0777 -e 'my ($a,$t)=@ARGV; open my $af,"<",$a or die $!; local $/; my $x=<$af>; open my $tf,"<",$t or die $!; my $y=<$tf>; my @b=($x =~ /^OLD:\n\n```markdown\n(.*?)\n```/msg); my $bad=0; for my $b (@b){my $q=quotemeta($b); my $n=()=($y =~ /$q/g); $bad++ if $n != 1;} print "old_blocks=".scalar(@b)." bad_occurrence_count=$bad\n"; exit($bad ? 1 : 0);' \
  docs/AMEND_CMFGEN_TWIN_MAP_REV3_2026-09-09.md \
  <(git show HEAD:docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md)

head -n 3 \
  /gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on/work/lumina_plasma_state.csv
head -n 7 data/standart_data1/toy06/phys_toy06_cmfgen.txt
sed -n '563,582p' /gpfs/kjhan/lumina/cmfgen_map/worktree/scripts/cmfgen_twin_map.py
```

감사 시 재현 결과: `old_blocks=36 bad_occurrence_count=0`; Lumina 공개 상태 헤더는
`shell_id,T_e,n_e`, CMFGEN phys 헤더는 `rho`, `ne`, `natom`을 포함한다.

## 결론

개정 3 설계는 집행 기록 18의 다섯 차단과 함께 기록된 폐합·피복 한계를 최소 변경으로 해소한다.
**Q-GOAL=`PASS`, Q-LEAN=`APPROPRIATE`, 전체 사전감사=`PASS`.** 위 해시의 개정안은 본 계약에
적용됐고 36개 `NEW` 블록의 단일 존재를 확인했다. 다음 허용 행동은 문서 프리플라이트다.
