# 독립 감사 결론

`docs/FABLE_UV_CENSUS.md`는 열람하지 않았고, 파일 수정·실행·커밋 없이 `git log/show/blame`, 기존 문서·메모리·런 로그만 읽었다.

핵심 판정은 다음과 같다.

1. **현재 s8 결정론 UV 장의 11.98배 진폭을 직접 만드는 원인은 확정됐다.**  
   `chi_coherent = chi_es + (1−eps_l)chi_line`가 형광으로 다른 빈에 나가야 할 선 불투명도를 같은 빈의 `J`로 재주입한다. `eps_eff=1.90567×10⁻⁴`, 재순환 이득 `5247.4904×`; 과잉에 필요한 이득 `5247.4106×`와 0.00152%로 닫힌다. MC에서는 같은 coarse bin을 이탈하는 비율이 95.1164%다. [E8:10](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:10), [E8:19](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:19)

2. **이 결함은 07-07 이후 새로 들어온 것이 아니다.**  
   물리적 line-epsilon 분할은 커밋 `43e509e`, 2026-06-11에 “EXPERIMENTAL, do not enable in production”으로 도입됐다. 07-07에는 이미 네이티브 UV 42.9%가 관측됐다. 따라서 이후 변경은 기존 편중을 악화·완화하거나 드러냈을 뿐, 최초 도입자로 판정할 수 없다. 현재 E8/E12 캡처는 이 경로를 `LUMINA_CMFGEN_LINE_EPS_PHYS=1`로 사용한다. [캡처 설정:27](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:27)

3. **진폭은 폐합됐지만 형상은 미해결이다.**  
   MC 파괴율을 대입하면 BALL은 0.9323×로 예측에 적중하지만 B1=4.916×, B0=8.291×가 남는다. 무편향 full 형광행렬도 B2→B0 흐름 때문에 B0를 26.43×로 악화시켰고, 단일-pass emergent 추정은 UV 42.9→40.41%에 그쳤다. [E9:78](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md:78), [E12:164](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E12.md:164)

4. **`p_iup≈88%`를 커밋 `192a2c3`의 IUP-JBLUE가 “만들었다”고 판정할 수 없다.**  
   JBLUE 전 로그에서도 강한 UV Fe III가 `p_iup=0.9918`이었다. JBLUE 단독에서는 0.9618, parity26에서는 0.8935, BINFIELD 단일변경 후 0.8915였다. 즉 높은 up-branch는 JBLUE 이전부터 존재했고, 현재 88%는 누적된 `J·population·beta·collision/BF/k-packet` 상태의 결과다. [pre-JBLUE 로그:40840](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx/stdout.log:40840), [JBLUE 로그:39887](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_jbl/stdout.log:39887), [parity26:32012](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity26/stdout.log:32012), [parity28:32044](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity28/stdout.log:32044)

5. **프로방넌스 한계가 있다.**  
   관련 마지막 Git 커밋은 `47bfa20`(07-18)이고, 현재 `src/*` 다수가 modified, E1–E13 문서가 untracked다. 따라서 07-18 이후의 “모든 소스 변경”을 커밋 단위로 증명할 수 없다. 아래 계보는 커밋된 변경과 날짜가 박힌 문서·런 로그를 구분한, 현재 증거로 가능한 전수 상한이다.

---

## A. 사슬 단계별 전수표

| 단계 | ① 가정 | ② 검증 상태 | ③ 단독 UV 편중 가능성 | ④ 단일-인자 판별 |
|---|---|---|---|---|
| MC 수송 물리 | Sobolev macro-atom의 흡수·분기·탈출과 패킷 에너지 장부가 올바르고, MC estimator가 같은 동결 상태의 장을 나타낸다. | Stage31의 인증 formal solver가 같은 `chi,eta`로 `J_MC` UV의 97.7181%를 재현했다. 따라서 전체 UV 진폭에 대한 “MC 수송 연산자 단독 결함”은 기각됐다. 다만 B2와 Fe III Γ는 반대 방향 잔차다. [7D:9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_BENCH7D.md:9) | 이론적으로는 잘못된 분기·탈출이 UV를 만들 수 있다. 그러나 관측된 전체 11.98×는 수송을 결정론으로 바꿔도 살아남으므로 **주원인 아님**. | 기존 7D가 사실상 단일인자 시험이다. 추가 시험은 같은 `chi,eta`에서 `J_MC/J_det`를 선·이온별로 분해해 B2/Fe III만 재검사. |
| 이벤트 분류 | line absorption과 직후 terminal emission이 같은 interaction에 정확히 대응하고, prefix 표본이 전체 사건을 대표한다. | E8 prefix는 970.6M 중 앞 400M만 저장한 비무작위 표본이라 대표성이 미검증이었다. E11 direct accumulator는 RNG·packet state를 건드리지 않고 cap 없이 계측하도록 설계됐고, E12에서 509,203,774 사건 중 509,047,721건을 분류했다. [E8:23](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:23), [E12:36](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E12.md:36) | 분류기는 원래 스펙트럼을 바꾸지 않는 관찰자다. 따라서 **네이티브 UV를 단독 생성할 수 없음**. 잘못 분류하면 수리행렬의 진단 결론만 오염시킨다. | prefix 행렬과 E11 full direct 행렬의 B2→B0 부호·크기를 비교. full에서도 같은 부호가 살아남아 prefix 편향만으로 E10 실패를 설명하는 가설은 기각됐다. |
| 형광행렬 누적 | 입력 line/bin의 에너지를 terminal 출력 bin으로 조건부 정규화하면 실제 redistribution operator가 된다. global 및 세 shell group이 s8 적용에 충분하다. | E12 full 장부의 global energy closure는 `3.04×10⁻⁸`, column closure 최대 `2.047×10⁻¹³`; k-packet energy share 2.04205%다. 산술은 검증됐지만 실제 line/shell/source covariance는 보존되지 않는다. [E12:36](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E12.md:36), [E12:184](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E12.md:184) | 원래 네이티브 스펙트럼에는 후처리 진단이므로 직접 원인이 아니다. 그러나 잘못된 global operator를 수리 source에 적용하면 B2 power를 B0로 보내 **수리본을 UV 쪽으로 악화**시킬 수 있다. | 동일 source에서 `R=I`, global R, s8 소유 group R만 교체. 나머지 `chi,J,EPAY`는 고정한다. |
| 이진 덤프 | `LFMAT001` schema, endianness, iteration, 크기와 내부 SHA가 정확하며 완료 iteration만 기록된다. | E11은 header·ledger·SHA·중복/nonfinite/closure 검사를 명세했고, 왜곡 fixture의 closure 1.2를 거부했다. [E11:79](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E11.md:79) | 생산 수송과 분리된 파일이므로 **네이티브 UV 생성 불가**. stale/partial dump이면 E10/E12 진단만 거짓이 된다. | 한 byte 변조·iteration mismatch·truncation fixture가 모두 fail-closed하는지 재사용. 이미 closure distortion은 검출됨. |
| 판독 | line index 방향, bin 경계, 누락 edge 처리와 그룹 합산이 writer 계약과 동일하다. | E13에서 native index convention이 정상이고 mirror 해석은 더 악화됐다. E11 reader는 duplicate/nonfinite/group-sum을 검사한다. [E13:47](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md:47) | 역시 진단 소비자이므로 **원래 42.9%를 만들 수 없음**. 방향을 뒤집으면 수리 진단의 UV/광학 흐름만 거꾸로 해석한다. | identity fixture, native/mirror 두 판독, writer side-ledger 합을 대조. mirror는 이미 반증됐다. |
| 재분배 적용 | energy-weighted R을 현재 line-return source에 적용할 수 있고, 미관측 edge=0 처리와 global/shell 투영이 물리적이다. | 산술 closure는 E10에서 `2.22×10⁻¹⁶`, E12에서 약 `10⁻¹⁴`. 하지만 E10 prefix는 B0 8.29→20.91, E12 full은 B0 26.43으로 악화했다. full에서도 B2→B0가 54.92%다. [E10:66](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E10.md:66), [E12:88](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E12.md:88) | **수리본 형상에는 단독 문제 가능**. 원인은 R 자체의 산술보다 “어떤 activation energy·shell·line source에 R을 적용했는가”의 owner mismatch다. 네이티브 증상의 최초 원인은 아니다. | 동일 R과 동일 source에서 `identity ↔ global ↔ s8 owner-resolved`만 바꾼다. pre-EPAY line activation ledger가 없으면 신규 계측 1회가 필요하므로 현재는 UNRESOLVED. |
| `chi/eta` 재구성 | Lumina의 `eps_l B(T_e)` thermal eta와 `(1−eps_l)chi_line J_same-bin`이 CMFGEN의 population-based `A_ul n_u` emissivity 및 겹친 선 transfer를 근사한다. EPAY 재형상이 이를 보존한다. | 이 가정은 **반증**됐다. Lumina는 선을 단일 1000-bin 중심에 넣고 `eps_l wB`를 사용하지만 CMFGEN은 `n_l,n_u`로 `chi_l`, `A_ul n_u`로 `eta_l`을 직접 만든다. E1 population swap은 2.79%만 개선, E6 `A_ul n_u` 단순 교체는 22.46× 악화, E8/E9는 same-bin 재순환 진폭을 폐합했다. [E1:212](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E1.md:212), [E1:225](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E1.md:225), [E1:249](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E1.md:249), [E9:78](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md:78) | **예. 확정된 최대 원인.** `S_fixed`가 CMFGEN의 1/438인데 같은-bin line recycling이 이를 5247× 증폭해 11.98× 장을 만든다. | E9의 `eps_MC` 치환은 이미 진폭 falsifier를 통과했다. 최종 단일인자 시험은 같은 population·continuum·grid에서 current `eps/B/EPAY` line assembly만 population-native `chi_l,eta_l`로 교체하고 stage31을 1회 푸는 것. |
| stage31 수송 | frozen `chi,eta`를 정확하고 안정적으로 푸는 선형 formal solver이며 인구·source를 다시 갱신하지 않는다. | 잔차 `9.42×10⁻⁷`, clamp 0, 입력 SHA 동일, `J_det/J_MC=0.977181`이다. E9 수정 source에서도 잔차 `8.18×10⁻⁷`, 3회 SHA 동일이다. [7D:9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_BENCH7D.md:9), [E9:104](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md:104) | 전체 UV를 새로 만들지 않고 입력 `chi,eta`의 과잉을 재현한다. 단 B2·Fe III의 비국소/해상도 잔차는 남는다. | current source와 identity/population-native source를 동일 solver에 투입. solver 자체 판별은 7D로 완료. |
| 대역 집계 | 주파수 순서·bin 폭·부분 overlap·`Fνdν` 단위가 양 코드에서 일관되고 B0–B4/BALL 정의가 고정돼 있다. | 7D는 CMFGEN velocity mapping과 적분보존 bin averaging을 사용했다. E13 native 색인은 정상이며 mirror는 악화했다. [7D:112](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_BENCH7D.md:112), [E13:47](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md:47) | 단위·순서 오류라면 큰 가짜 UV가 가능하지만, native/mirror·grid/SHA·대역별 반복 결과로 **11.98× 주원인 가설은 기각**. | 현재 1000-bin 적분을 원 fine spectrum 직접 적분과 대조하고, band edge를 ±½ bin 이동. 전체 BALL이 유지되면 집계 무죄. |
| CMFGEN 대조 | 같은 epoch, shell/velocity, luminosity normalization, source run 및 스펙트럼 column을 비교한다. | 7D의 CMFGEN 보간은 명시적이다. 같은 데이터 ARTIS 20.2% 기록은 있으나 실행 디렉터리·commit·band·column·적분 명령이 보존되지 않아 독립 재현은 UNRESOLVED다. [E13:263](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md:263) | 잘못된 comparator는 격차를 가짜로 만들 수 있다. 다만 07-07 same-data ARTIS/Lumina 분리는 알고리즘 문제라는 독립 방향 증거다. [handoff:12](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_fluor_rewiring_handoff.md:12) | 정확한 ARTIS 20.2% recipe와 CMFGEN `ETA/CHI` dump의 build manifest를 복원해 같은 band integration을 재실행. 현재는 UNRESOLVED. |

---

## B. 변경 계보

### 1. 07-07에 이미 들어와 있던 선행 변경

| 시점 | 변경 | 당시 지표와 판정 |
|---|---|---|
| 05-31 | super-level 투영 | A/B에서 UV/blue 0.65→0.64로 사실상 null, NIR만 개선했다. 따라서 현재 UV 편중의 도입 원인으로 지지되지 않는다. [super-level A/B:7](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_superlevel_ab_result.md:7) |
| 06-11 | line-epsilon/coherent split, `379510e`·`43e509e` | `LUMINA_CMFGEN_LINE_EPS_PHYS`가 실험 기능으로 도입됐다. 이것이 E8에서 확인된 same-bin recycling의 **최초 확인 가능한 구현 계보**다. 당시 문구는 production 금지였고, production config로 처음 승격된 정확한 시점은 UNRESOLVED. |
| 06월 이전~ | 고정 1000-bin·line-center 투영 | 현재 assembler가 각 선을 단일 중심 bin에 넣는 것은 확정됐지만, 최초 도입 커밋은 이번 증거에서 특정하지 못했다. 따라서 “07-07 이후 변경”이 아니라 **상속된 구조**다. [E1:249](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E1.md:249) |
| 07-03, `66586d6` | TF32 NLTE rate lane | `Rbf=KᵀJ`의 CPU/GPU 차이는 최대 상대 약 `6×10⁻⁵`, 속도 528×였다. 현재 parity59 캡처는 `LUMINA_NLTE_ASSEMBLE_GPU=0`이므로 TF32 lane이 비활성이다. **E8/E12 원인 불가**. [TF32 memory:7](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_nlte_rates_gemm.md:7), [캡처 설정:93](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:93) |
| 07-05, `8e97890`, `6d08a35` | EPAY 도입·재형상 | 최초 epay4의 UV 57.6→31.9%는 이후 gain runaway 오염으로 무효화됐다. 수정 epay7은 온도 RMS 20.4%·formal corr 0.663이지만 같은 시점 UV 지표는 없다. **완화 가능성은 있으나 최초 수치는 증거로 사용 불가**. |
| 07-06, `0ac817c` | EPAY TAUBIN=10 | accidental blanket을 제거하며 “honest residual” UV 51.4% vs CMFGEN 23.4%를 다시 노출했다. 즉 편중 도입보다는 기존 결함의 은폐를 제거해 **겉보기 악화**시켰다. |
| 07-06, `ba187ab` | IDOWN_BETA·line-resolved Jbar | capped/escaper-biased epay20에서 UV 32.4 vs 23.4, corr 0.788. 편중을 완화했으나 cap 76–82% 때문에 절대 판정에는 제한이 있다. |
| 07-06/07, `ff58168` | k-packet 재주입 활성화 | UV 54.0→42.9%, green 8.5→12.4로 개선. 이 결과가 07-07 네이티브 기준이다. 즉 **42.9%를 도입한 변경이 아니라 54%를 완화한 변경**이다. |

### 2. 07-07 이후

| 시점 | 변경 | 도입·악화·완화 판정 |
|---|---|---|
| 07-07~08 | `LUMINA_MC_COEVOLVE` Stage0/1 및 `INJECT=2` | Stage0/1은 처음 shadow-only, OFF byte-identity가 07-08 확인됐다. 첫 full-scale은 s0–4 field가 dead이고 30–100× normalization offset; 색은 약간만 redder였다. **아키텍처 자체는 기존 UV를 제거하지 못했고, 최초 도입 원인도 아니다.** [co-evolve:10](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_coevolve_stage01_implemented.md:10), [co-evolve:43](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_coevolve_stage01_implemented.md:43) |
| 07-10~12 | photoion MC-field 재배선 P1 | α=0.5 transient에서 UV 28.7→14.6%, green 7.4→14.4%로 크게 완화했지만 outer `T_e` 폭주와 미수렴이 발생했다. α sweep도 비단조였다. 최종 안정 수리로 채택할 증거는 아니다. [P1:28](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_p1_photoion_mc_ionization.md:28), [P1:42](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_p1_photoion_mc_ionization.md:42) |
| 07-14, `192a2c3` | IUP-JBLUE, BF-NLTE, FB multi/CDF, cooling-kT, all-level GPH, event logger | 커밋 본문상 KPKT-FBUP만 기본 동작 bugfix이고 나머지는 기본 OFF gate다. IUP-JBLUE 단독은 corr −0.192, blue 31.0%, red/NIR 45.2%로 형광 회복에 실패했다. 따라서 “ARTIS-exact” 이름과 달리 **관측상 악화/실패**. [IUP verdict:26](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_iup_jblue_arm_2026-07-14.md:26) |
| 07-14 | KPKT-FBUP 실제 수리 | 이전 모든 KPACKET run에서 device `p_ff/p_fb/fb_nu=0`, ff/fb exit가 한 번도 발화하지 않았던 버그를 고쳤다. 이후 첫 발화는 EUV 166–355 Å trap, `J[mid] 350×`, corr 0.125로 **크게 악화**했다. [FBUP:16](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_bfnl_verdict_kpkt_fbup_bug.md:16), [FBUP:24](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_bfnl_verdict_kpkt_fbup_bug.md:24) |
| 07-15, `0476c83` | all-level NLTE GPH | IGE III→IV는 개선했으나 IME를 과이온화해 narrow corr 0.474→0.372. **혼합, 스펙트럼 악화**. |
| 07-15, `d57bc98` | alpha spingate | Fe III 재결합률을 4.5–6.7× 과대계상하던 spin-forbidden 항을 제한했다. 이온화 parity 수리이며 UV 진폭을 직접 폐합한 증거는 없다. 현재 캡처에는 ON이다. [캡처 설정:15](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:15) |
| 07-21~22 | ARTIS parity bundle: 24-bin fit의 1000-bin 재평가, CDF/field 소비, energy weighting, collision network 등 | 초기 parity1–9 일부 지표는 stale ion file로 오염돼 폐기됐다. 첫 정직한 parity10 이후의 결과만 유효하다. M-engine의 energy-dimension mixing 수리로 line emission을 복원했으나, UV 편중을 단일 폐합하지는 못했다. [campaign:27](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:27), [campaign:49](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:49) |
| 07-25 | C1 superbin TEPIN | `p_iup≈88.3%` 상태에서 EUV bin을 `T_e`에 pin했지만 b4는 2.90으로 남고, 실 EUV excess energy도 제거하지 못했다. **원인 아님/불충분**. [campaign:142](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:142) |
| 07-26 | IUP-BINFIELD, 1000-bin field 소비 | 단일 변경이 EUV forest emission을 13% 증가시키고 b9 15.8→19.5, b4 2.90→3.66으로 악화했다. `p_iup`도 0.8935→0.8915로 거의 불변이었다. **coarse-bin coupling이 악화 증폭기는 될 수 있지만 `p_iup≈88%`의 생성자는 아님**. [campaign:144](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:144) |
| 07-27경 | KPEMISS/SE-pop CDF source population 수리 | parity29 Fe III `p_iup=0.8717`, `p_idown=0.1068`. 위쪽 확률을 일부 낮췄지만 여전히 지배적이며, 형광 구조를 해결하지 못했다. [parity29:29992](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity29/stdout.log:29992) |
| 07-30 | JBAR damping unify | raw/EMA 장의 분열을 제거한 배선 위생 수리. ARTIS는 raw 단일장을 쓴다는 근거는 있으나, 강한 UV를 실질적으로 낮춘 단일변수 결과는 없다. [campaign:918](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:918) |
| 07-31 | `MA_LINE_DESTRUCT` 1→0 | 동일 `C_down`을 pre-roll과 terminal에서 두 번 추첨한 결함이 확정됐다. 단일변수 OFF에서 formal −21.8968%, s0 `T_e` −2896.8 K. 이는 UV 과잉을 만든 채널이라기보다 ON 상태가 우연히 일부 radiation을 파괴하던 **오염/부분 완화 채널**이다. [double draw:1](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_P06_DOUBLE_DRAW_VERDICT.md:1), [A/B:3](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_PARITY57_V3_ADVERSARIAL.md:3) |
| 07-31 | Wave 1: bf stim-recomb·neutral·spingate | 세 버그 수리가 oracle에서 인증됐다. parity60에서 D4 OFF와 합치면 formal 17.74→13.91, −21.6%였지만 bf 수리 자체 순효과는 +0.18 formal로 작았다. **직접 UV 해결책 아님**. [campaign:1093](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:1093), [campaign:1121](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:1121) |
| 07-31 | Wave 2: continuum CDF | 실제 `nu_cmf`, `nu_edge/nu` 분기, clamp 제거까지 검증했지만 physical event 효과는 생산 run으로 측정되지 않았다. **UV 영향 UNRESOLVED**. parity59 E8 설정에는 이 Wave2 fix gate가 없다. |
| 07-31~08-01 | Wave 3: element-wide SE·`M_V` | 초기 두 라운드는 무개선/악화. 08-01 s0 Fe에서만 최초 EW_PASS, s8 변화 `2.43×10⁻¹¹`; 따라서 E8 s8 UV의 원인이 아니다. [campaign:1151](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:1151), [campaign:1168](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:1168) |
| 08-01 | Stage31 | 기존 `chi,eta`를 무잡음으로 재현한 진단 solver다. UV를 도입하지 않았고 수송 단독 원인을 기각했다. |
| 08-02 | E1–E13 교체·행렬 시험 | 모두 frozen/offline 진단이다. E9 scalar 교체는 진폭 HIT, E10/E12 matrix 적용은 형상 FAIL. 원래 42.9%를 만든 production 변경은 아니다. |

### 현재 캡처 설정에서 특별히 주의할 점

E8/E12가 읽은 parity59 캡처는 동시에 다음을 사용한다.

- EPAY=2, TAUBIN=10: [설정:32](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:32)
- KPACKET=1, EWEIGHT=1, IDOWN_BETA=1: [설정:76](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:76)
- COEVOLVE+CONSUME+INJECT=2: [설정:88](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:88)
- super-level K=100: [설정:126](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:126)
- **결함으로 확정되어 후일 production OFF된 `MA_LINE_DESTRUCT=1`**: [설정:85](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:85)
- TF32 rate assembly OFF: [설정:94](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_FLUORMAT.env:94)

따라서 E8의 5247× 폐합은 강하지만, 그 절대 상태를 “최종 production baseline”으로 부르면 안 된다. parity59는 D4 double-draw 오염을 포함한다. 다만 D4 OFF의 21.9% 효과는 11.98× 과잉의 크기를 설명하지 못하고, E8의 same-bin recycling 판정을 뒤집지도 않는다.

---

## `p_iup≈88%`의 기원 판정

현재 캡처의 강한 UV Fe III는 shell 0에서 `p_iup=0.8819`, shell 3에서 0.8859다. radiative-only terminal 계산은 Fe III 89.6351%, Fe II 98.4018%, 전체 실측 92.7362%로 색인과 복사 분기 자체는 이론에 맞는다. [E13:220](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md:220), [E13:248](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md:248)

계보상 수치는 다음과 같다.

- JBLUE 이전 `a10_kx`: Fe III `0.9918/0.0067`
- IUP-JBLUE: `0.9618/0.0325`
- parity26: `0.8935/0.0887`
- BINFIELD 단일변경 parity28: `0.8915/0.0904`
- KPEMISS parity29: `0.8717/0.1068`
- parity60: `0.9023/0.0829`

따라서:

- **IUP-JBLUE가 높은 `p_iup`을 생성했다: REFUTED.**
- **BINFIELD가 생성했다: REFUTED.**
- **KPEMISS가 완화했다: 부분 지지, 규모 약 2%p이며 여전히 up 지배.**
- **특정 단일 변경이 현재 88%를 만들었다: UNRESOLVED.**

확정 가능한 좌표는 internal-up의 `beta × 실제 J × stimulated correction × population ratio`와 그 뒤 collision/BF/k-packet·normalization/damping이다. E13도 이 좌표까지만 특정하고 단일 원인은 미해결로 둔다. [E13:236](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md:236)

또한 `192a2c3`은 IUP-JBLUE를 “ARTIS-exact”라고 불렀지만, 현재 로컬 ARTIS 설정은 `DETAILED_LINE_ESTIMATORS_ON=false`라 line-specific JBLUE 소비자가 아니고 binned `radfield(nu)`를 사용한다. BINFIELD가 구성상 더 가깝지만 실제 단일 A/B에서는 악화했다. [campaign:1014](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:1014), [campaign:1027](/home/kjhan/.claude/projects/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/memory/project_artis_parity_campaign.md:1027)

---

## C. 최종 후보 3개와 단일-인자 시험

### 1. 같은-bin coherent line recycling — CONFIRMED, 압도적 1위

크기:

- coherent opacity 97.7713%
- `S_fixed/CMFGEN=0.00228248`
- 재순환 이득 5247.49×
- 실제 필요한 이득과 0.00152% 일치
- MC 파괴율 대입 후 BALL 0.9323× 적중

시험:

- E8/E9와 같은 frozen population·continuum·grid를 사용한다.
- current `eps_l B + (1−eps_l)chi_lJ` line assembly만 population-native `chi_l[n_l,n_u]`, `eta_l=A_uln_u`로 교체한다.
- EPAY, continuum, boundary, stage31 solver는 그대로 둔다.
- 판별: BALL이 O(1)로 붕괴하면 진폭 원인 최종 확정; B0/B1이 남으면 형상 원인과 분리된다.

이 시험은 E1이 제안한 current-vs-population-native 동시 조립과 동일하다. [E1:281](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E1.md:281)

### 2. EPAY/activation owner와 source–matrix covariance — UNRESOLVED, 형상 1위

크기:

- full 행렬에서도 B2→B0 54.92%
- B0/CMFGEN 26.43×
- single-pass emergent UV 개선은 필요한 양의 13.02%뿐
- E12 자체 잔여 순위도 EPAY/activation owner를 1위, line projection/covariance를 2위로 둔다. [E12:184](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E12.md:184)

시험:

- E12의 같은 full `R`과 같은 s8 `chi,J`를 고정한다.
- 변수 하나만 바꾼다: matrix column weight를 현재 post-EPAY line-return proxy에서 **실제 pre-EPAY, line·shell별 activation energy owner**로 교체한다.
- `R`, transport, bands, normalization은 불변.
- 판별: B2→B0와 B0/CMF가 크게 줄면 owner mismatch; 불변이면 R 자체의 실제 branching/주파수 구조가 원인이다.

현재 자산에는 완전한 pre-EPAY line-owner ledger가 없어 이 시험은 완전 오프라인으로 끝낼 수 없다. direct accumulator에 owner 필드만 추가한 1회 캡처가 필요하다. 그러므로 원인은 **UNRESOLVED**다.

### 3. internal-up probability assembly와 coarse-bin field coupling — UNRESOLVED

크기:

- 현재 강한 UV Fe III `p_iup≈0.882–0.886`
- JBLUE 이전에도 0.9918이므로 오래된 문제
- BINFIELD 단독은 `p_iup`을 낮추지 못하고 EUV forest를 13% 악화
- `p_iup`은 terminal energy 행렬과 동일한 가중 통계가 아니므로 88%를 직접 UV flux 결함으로 환산할 수 없다.

시험:

- 현재 capture의 population, `beta`, collision/BF/k-packet rate를 동결한다.
- internal-up에서 소비하는 J만 세 arm으로 replay한다: current JBLUE, exact bin-field, CMFGEN `Jν`/line-profile-integrated J.
- 난수 수송 없이 transition probability와 absorbing Markov cascade를 계산해 `p_iup`, terminal B0–B4, same-bin survival을 비교한다.
- 판별: CMFGEN J arm만 `p_iup`과 UV terminal을 동시에 크게 낮추면 field-owner 원인; 세 arm이 모두 높으면 population/beta/collision-rate assembly가 원인이다.

기존 parity27/28은 부분 음성대조지만 state가 함께 진화했으므로 이 고정상태 시험을 대체하지 못한다.

---

## 최종 책임 판정

- **원래 07-07의 UV 42.9%를 도입한 변경:** 07-07 이후 변경이 아님. 확인 가능한 최초 구현 계보는 06-11 `43e509e`의 physical line-epsilon/coherent split. 다만 이 기능이 어느 런 설정에서 처음 production 활성화됐는지는 **UNRESOLVED**.
- **현재 11.98× UV 장 진폭의 직접 원인:** `(1−eps_l)chi_line J_same-bin` 재순환 — **CONFIRMED**.
- **07-07 이후 가장 큰 악화 변경:** KPKT-FBUP 활성화 후 EUV trap/J 350×/corr 0.125, 그리고 BINFIELD의 forest +13% — 다만 둘 다 기존 42.9%의 최초 원인은 아니다.
- **가장 큰 완화 변경:** pre-baseline KPACKET ON의 UV 54.0→42.9; 이후 P1 transient의 28.7→14.6은 미수렴·outer runaway로 생산 수리 판정 불가.
- **현재 형상 실패 원인:** EPAY/activation owner 및 line/shell/source covariance가 최우선이나 **UNRESOLVED**.
- **`p_iup≈88%` 생성 변경:** 단일 변경 특정 불가. IUP-JBLUE·BINFIELD 생성설은 반증됐다.
- **MA_LINE_DESTRUCT:** 확정 버그이자 parity59 캡처 오염원이지만, 방향과 크기상 UV 과잉의 주범은 아니다.
- **TF32·Wave2·Wave3·super-level:** 현재 E8 진폭 원인에서 배제 또는 영향 미측정이다.
