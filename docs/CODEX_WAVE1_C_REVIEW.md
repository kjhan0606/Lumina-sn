## 총평: FAIL

신규 게이트 4개를 독립 정적 검토했습니다. A 보고서는 근거에서 제외했고 수정·빌드·실행·git은 하지 않았습니다.

| 게이트 | 판정 | 핵심 |
|---|---|---|
| `LUMINA_FIX_BF_STIM_RECOMB` | **FAIL(엄격 ARTIS 동등성)** / 핵심 공식 PASS | corrfactor 대수는 맞지만 target probability·clumping·target-map 독립성이 빠짐 |
| `LUMINA_FIX_BF_NEUTRAL` | **PASS(인덱스/부호)** | stage 0, `stage+1`, `chi-E_level` 모두 일관됨 |
| `LUMINA_FIX_BF_ETA_SPINGATE` | **PASS(술어 일관성)** | S1/S2/S3와 동일 공용 술어 |
| `LUMINA_FIX_BF_MULTI_EDGE` | **FAIL(완결성/OFF 계약)** | alias가 명시적 OFF를 이기며, 두 GPU 방출점의 Milne 처리가 다름 |

### 1. 공식 정확성

**Stimulated recombination — 핵심 대수 PASS**

항별 대응은 맞습니다.

- `nu_edge`, `sigma_bf`: [rpkt.cc:733](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:733>) ↔ [lumina_plasma.c:6842](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6842>), [lumina_plasma.c:6935](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6935>)
- \(n_u/n_l \cdot n_e \cdot {\rm SAHACONST}\cdot g_l/g_u\cdot T_e^{-3/2}\): [rpkt.cc:741](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:741>) ↔ [lumina_plasma.c:6889](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6889>), [lumina_plasma.c:6916](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6916>)
- \(\exp[-h(\nu-\nu_0)/(kT_e)]\), `max(0,1-stimfactor)`: [rpkt.cc:754](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:754>) ↔ [lumina_plasma.c:6945](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6945>)
- 최종 \(n_l\sigma\,corr\) 합산: [rpkt.cc:757](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:757>), [rpkt.cc:765](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:765>) ↔ [lumina_plasma.c:6943](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6943>), [lumina_plasma.c:6952](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6952>)

그러나 엄격 동등성은 FAIL입니다.

- ARTIS의 `allcont_probability[i]`가 Lumina 합산에는 없습니다: `rpkt.cc:757` 대 `lumina_plasma.c:6943`. 현 CMFGEN 대상은 단일 ground route라 \(p=1\)로 축약 가능하다는 자료 구조는 있지만([build_ma_radrecomb_target.py:19](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_ma_radrecomb_target.py:19>)), 일반 multi-target continuum에는 동등하지 않습니다.
- 정확한 upper target map은 별도 `LUMINA_MA_RADRECOMB`가 켜져야만 로드됩니다([lumina_cuda.cu:6131](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6131>), [lumina_cuda.cu:6145](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6145>)). Stim 게이트 단독은 ground fallback입니다([lumina_plasma.c:6889](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6889>)).
- ARTIS는 `clumpednne`를 사용하지만([rpkt.cc:676](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:676>)), Lumina는 `plasma->n_electron`을 직접 씁니다(`lumina_plasma.c:6919`). clump factor=1에서만 동일합니다.
- `SAHACONST`는 일치하지만, ARTIS와 Lumina의 \(h,k_B\) 상수가 서로 달라 bit/numeric parity는 아닙니다: [constants.h:19](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/constants.h:19>), [constants.h:32](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/constants.h:32>) 대 [lumina.h:21](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:21>).

**Neutral BF — PASS**

- stage는 상대 슬롯이 아니라 neutral=0인 절대 stage입니다: [lumina.h:411](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:411>), [lumina_atomic.c:828](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:828>).
- neutral의 이온화 에너지는 `(Z, stage=0)`으로 찾고([lumina_plasma.c:6737](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6737>)), 생성 이온은 `stage+1`로 찾습니다([lumina_plasma.c:6681](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6681>)).
- level threshold 부호도 \((\chi_{\rm ion}-E_l)/h\)로 맞습니다([lumina_plasma.c:6842](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6842>)).
- coarse/fine GEMM도 동일 stage 규약을 전달합니다: [lumina_bf_gemm.cu:73](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:73>), [lumina_bf_gemm.cu:311](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:311>).

**Spin 술어 — PASS**

공용 `spingate_level_forbidden()`은 “알려진 multiplicity이고 \(M\ne M_c\pm1\)”만 금지하며 unknown은 허용합니다([lumina_plasma.c:2894](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2894>)). eta가 바로 이 함수를 호출하므로([lumina_plasma.c:6962](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6962>)) S3([lumina_plasma.c:2960](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2960>)), S1([lumina_plasma.c:15261](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15261>)), S2([lumina_plasma.c:15540](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15540>)와 동일합니다.

### 2. 게이트 OFF byte-불변 적대 수색

정적 제어·데이터 흐름 기준입니다.

- `STIM_RECOMB=0`: **PASS** — 기존 GEMM 조건이 그대로 참이고([lumina_plasma.c:6653](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6653>)), corr 산술은 도달하지 않습니다.
- `NEUTRAL=0`: **PASS** — 기존 `stage<1` skip과 GPU zero-store가 유지됩니다([lumina_plasma.c:6735](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6735>), [lumina_bf_gemm.cu:77](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:77>)).
- `ETA_SPINGATE=0`: **PASS** — `&&` short-circuit로 술어가 호출되지 않습니다(`lumina_plasma.c:6964`).
- `MULTI_EDGE=0`: **FAIL(적대 환경)** — `LUMINA_KPKT_FB_MULTI=1`이면 명시적 신규 게이트 `0`도 OR에 의해 ON이 됩니다([lumina_plasma.c:6442](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6442>)). legacy alias까지 unset/0인 환경에서는 OFF 경로가 보존됩니다.

### 3. “왜 생겼나” 5대 근원 계보

1. **물리항 누락** — stim-recomb: 기존 `n_level*sigma`에 순방향 흡수만 있었음(`lumina_plasma.c:6943`).
2. **도메인 의미 혼동** — neutral BF: free-free의 “전하 0 제외” 규칙을 photoionization에도 적용한 `stage<1` skip([lumina_plasma.c:6729](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6729>)).
3. **술어 소유권/전파 누락** — eta spin: 원래 한 recombination owner에 있던 inline 규칙이 다른 생산자로 전파되지 않음([lumina_plasma.c:2803](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2803>)).
4. **대표값 축약** — multi-edge: 전체 emissivity 합을 dominant ion의 단일 edge로 축약([lumina_plasma.c:4921](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4921>)).
5. **생산자/backend 분기** — CPU/coarse GEMM/fine GEMM/GPU emission이 별도 구현되어 한 수정이 여러 경로에 반복돼야 함([lumina_bf_gemm.cu:272](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:272>), [lumina_cuda.cu:4812](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4812>), [lumina_cuda.cu:5618](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5618>)). Stim·neutral·multi 모두 이 계보가 겹칩니다.

### 4. 수리 누락·부작용

- **FAIL:** multi-edge의 두 GPU 방출점이 다릅니다. 첫 방출점은 level index와 `sigma_bf*nu²` rejection을 사용하지만([lumina_cuda.cu:4825](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4825>), [lumina_cuda.cu:4867](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4867>)), BF-activated 방출점은 level을 읽지 않고 단순 thermal tail만 씁니다([lumina_cuda.cu:5624](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5624>), [lumina_cuda.cu:5660](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5660>)).
- **FAIL/부분수리:** CDF는 전체 continuum이 아니라 상위 16개만 보존·재정규화합니다([lumina.h:34](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:34>), [lumina_plasma.c:5016](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:5016>)).
- **부작용:** multi-edge는 단순 edge 선택만 바꾸지 않고 `p_fb/p_ff` 전체 분기확률도 Kramers에서 `C_fb_real`로 교체합니다([lumina_plasma.c:5096](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:5096>)).
- **누락:** neutral excited level의 edge가 `nu_min` 아래이면 모든 상위 주파수에 기여해야 하지만 현재 level 전체를 skip합니다([lumina_plasma.c:6845](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6845>)).
- **부작용:** stim 게이트는 coarse GEMM을 막고 fine GEMM도 `-1` fallback시킵니다([lumina_plasma.c:6653](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6653>), [lumina_bf_gemm.cu:285](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:285>)). 정확성 목적이지만 성능과 sharp-edge 해상도는 후퇴합니다.
- **의도된 부작용:** spin gate는 photoionization을 남기고 recombination만 제거하므로 LTE detailed balance를 깹니다([lumina_plasma.c:2817](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2817>)).
- **자원 누수:** multi host 배열은 할당되지만([lumina_plasma.c:3989](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3989>)) 대응 host `free`가 `src`에 없습니다. device 배열만 해제됩니다([lumina_cuda.cu:3076](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3076>)).