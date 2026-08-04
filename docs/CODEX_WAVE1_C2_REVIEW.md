종합 **FAIL** — Wave 1 폐합 보류입니다. 정적 소스 판정이며 수정·실행·git·구현 보고서 열람은 하지 않았습니다.

1. `FIX_BF_STIM_RECOMB`: **PASS**

- ARTIS 기준은 corrfactor 계산 후 target probability를 곱하고 최종적으로 `nnlevel`을 곱합니다: [rpkt.cc](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:733>), [rpkt.cc](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:737>), [rpkt.cc](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:754>), [rpkt.cc](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:757>), [rpkt.cc](</home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:765>).
- Lumina는 target별 `n_upper/g_upper`, `clumped_ne`, ARTIS `SAHACONST`, `T_e^-1.5`를 계산합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6918), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6933), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6963).
- 각 target에 `p × max(0,1-stimfactor)`를 적용한 뒤 합산합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7000).
- CSR target probability의 로딩·검증도 연결되어 있습니다: [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1185), [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1211).
- clumping은 gate ON에서만 적재되고 smooth 기본값은 정확히 `1.0`입니다: [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:390).

2. `FIX_BF_MULTI_EDGE`: **FAIL**

OFF/alias 계약은 **PASS**입니다. 판정식은 canonical 변수가 존재하면 그 값을 사용하고, 없을 때만 alias를 참조합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6442).

| `FIX_BF_MULTI_EDGE` | alias 미설정 | alias=0 | alias=1 |
|---|---:|---:|---:|
| 미설정 | OFF | OFF | ON |
| 명시 0 | OFF | OFF | **OFF** |
| 비영(1 등) | ON | ON | ON |

따라서 적대 조합 `FIX=0, alias=1`에서도 명시 OFF가 이깁니다.

하지만 Milne 통일은 **실동작 기준 FAIL**입니다.

- 두 방출 지점 모두 공용 샘플러를 호출하는 구조 자체는 맞습니다: [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4523), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4905), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5674).
- 그러나 sigma 포인터는 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6850)에서 바인딩되고, 실제 BF GEMM 초기화는 그 뒤 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6961)에서 발생합니다.
- getter는 당시의 아직 초기화되지 않은 `g_bf_gemm.d_sigma_bf`를 그대로 반환합니다: [lumina_bf_gemm.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:403).
- 이후 sigma 포인터 재바인딩이 없습니다. 따라서 공용 샘플러의 sigma-weighted Milne 분기 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4532)는 정상 시작 순서에서 진입하지 못하고, flat-sigma thermal-tail fallback [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4558)으로 갑니다.

3. 기존 PASS 2건 비퇴행: **PASS**

소스상 나머지 Wave-1 두 건으로 판독되는 항목:

- Neutral BF: CPU의 stage-0 포함 분기와 GEMM 전달이 유지됩니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6751), [lumina_bf_gemm.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_bf_gemm.cu:73).
- η spin gate: 기존 공용 spin predicate를 그대로 사용합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2894), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7023).

4. OFF byte-불변: **PASS(정적 경로)**

OFF에서는 clump/target/scratch 및 multi-edge 테이블 할당이 발생하지 않고, device의 기존 single-edge 분기로 복귀하여 추가 RNG도 소비하지 않습니다: [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:280), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6733), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4910), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5691).