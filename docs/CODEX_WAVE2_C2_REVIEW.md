종합 PASS — **Wave 2 폐합 가능**.

1. **PASS — 실제 `nu_cmf` 기반 CDF**

   ARTIS의 `n_lower × target_probability × σ(nu) × max(0, 1-stim)` 항을 사건 주파수에서 route별로 재계산한 뒤 누적합니다. 사전 계산 bin CDF나 dominant absorber를 사용하지 않습니다. ARTIS 기준식은 [upstream rpkt.cc](https://github.com/artis-mcrt/artis/blob/b3970305566a180edd784e244029fb6902af891c/rpkt.cc#L697-L808)와 일치합니다.

   근거: [lumina_plasma.c:6539](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6539), [lumina_plasma.c:6574](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6574), [lumina_plasma.c:6596](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6596), [lumina_plasma.c:7236](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7236).

2. **PASS — `p_ion>1` 은폐 clamp 제거**

   CPU/GPU 모두 `p_ion = nu_edge / nu_cmf`를 그대로 사용하며 `min(1, …)` 또는 `fmin` clamp가 없습니다. 선택 가능한 route는 먼저 `nu_cmf >= nu_edge`를 만족해야 하므로 정상 입력에서 비율은 구조적으로 1 이하입니다.

   근거: [lumina_transport.c:579](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:579), [lumina_cuda.cu:5815](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5815), [lumina_plasma.c:6543](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6543), [lumina_cuda.cu:3524](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3524).

3. **PASS — GPU/CPU 동등성**

   route 기여식, log-frequency 보간, Kramers 대체식, stimulated correction, strict `cumulative > threshold`, 마지막 유효 route 보정 및 target/edge 반환이 대칭입니다.

   근거: [lumina_plasma.c:6539](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6539) ↔ [lumina_cuda.cu:3514](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3514), [lumina_plasma.c:6585](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6585) ↔ [lumina_cuda.cu:3566](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3566).

4. **PASS — 실패 route 난수 소비**

   BF channel 진입 후 CDF 난수를 검증 전에 항상 한 번 소비하고, route 실패 여부와 무관하게 energy-split 난수를 추가로 소비합니다. CPU/GPU 순서가 동일합니다.

   근거: [lumina_plasma.c:6587](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6587), [lumina_transport.c:573](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:573), [lumina_transport.c:577](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:577), [lumina_cuda.cu:3576](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3576), [lumina_cuda.cu:5811](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5811).

5. **PASS — fallback 카운터 실배선**

   route별 fallback 표식이 CPU/GPU로 전달되고, 실제 fallback target이 MA로 활성화된 경우에만 증가합니다. CPU 종단 출력과 GPU reset/get/종단 출력까지 연결되어 있습니다.

   근거: [lumina_plasma.c:6767](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6767), [lumina_transport.c:582](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:582), [lumina_cuda.cu:5818](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5818), [lumina_cuda.cu:2364](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2364), [lumina_cuda.cu:2377](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2377), [lumina_main.c:863](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_main.c:863).

6. **PASS — 기존 PASS 항목 비퇴행**

   event gate 단독으로 target map을 적재하며, multi-target 확률, upper-ground fallback, unmappable target의 thermal-pool fail-close 및 strict CDF 선택이 유지됩니다.

   근거: [lumina_atomic.c:985](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:985), [lumina_plasma.c:6712](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6712), [lumina_plasma.c:7248](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7248), [lumina_transport.c:593](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:593), [lumina_cuda.cu:5826](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5826).

7. **PASS — OFF byte-불변**

   OFF에서는 event route를 할당·업로드하지 않고 기존 dominant-absorber 표와 기존 CPU/GPU 분기로 돌아갑니다. 추가 CDF/split RNG 및 fallback 진단 출력도 도달하지 않습니다.

   근거: [lumina_plasma.c:6473](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6473), [lumina_plasma.c:6900](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6900), [lumina_transport.c:611](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:611), [lumina_cuda.cu:342](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:342), [lumina_cuda.cu:5777](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5777), [lumina_cuda.cu:5833](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5833).

요청대로 구현 보고서는 읽지 않았으며 수정·실행·git 없이 소스 정적 현물만 판정했습니다.