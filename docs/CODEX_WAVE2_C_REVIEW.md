소스 현물 기준 종합 판정은 **FAIL**입니다. A 보고서는 읽지 않았고 수정·실행·git 조작도 하지 않았습니다.

### ① ARTIS `rpkt.cc:405–445` 항별 대조 — FAIL

- **CDF 형태: 부분 PASS.** ARTIS는 BF 총 opacity에 난수 1회를 곱하고 첫 누적합 `>` 위치를 선택합니다([rpkt.cc:414](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:414), [rpkt.cc:418](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:418), [rpkt.cc:420](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:420)). CPU/GPU도 같은 `threshold=U*total`, 누적 `>`, 마지막-route fallback 구조입니다([lumina_plasma.c:6545](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6545), [lumina_plasma.c:6577](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6577), [lumina_cuda.cu:3501](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3501), [lumina_cuda.cu:3535](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3535)).

- **실제 주파수 CDF: FAIL.** ARTIS는 이벤트의 실제 `pkt.nu_cmf`를 사용합니다([rpkt.cc:371](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:371)). Lumina는 `nu`로 bin만 정한 뒤 CDF를 bin 중심 `nu_bin`에서 다시 구성합니다([lumina_plasma.c:6548](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6548), [lumina_plasma.c:6556](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6556), [lumina_cuda.cu:3507](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3507), [lumina_cuda.cu:3515](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3515)). Edge 근방에서는 실제 `nu<nu_edge`인데 bin 중심이 edge 위여서 불가능한 route가 선택되거나 반대 상황이 생길 수 있습니다. 뒤의 `p_ion>1` clamp는 이 불일치를 숨길 뿐 ARTIS 등가가 아닙니다([lumina_cuda.cu:5746](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5746)).

- **target 보존: PASS(데이터가 유효할 때).** CSR target이 route에 저장되고([lumina_plasma.c:6734](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6734)), GPU에 그대로 업로드되어([lumina_cuda.cu:367](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:367)) 선택 결과로 반환되고([lumina_cuda.cu:3543](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3543)) macro-atom 활성화에 사용됩니다([lumina_cuda.cu:5749](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5749)). 맵 부재 시 upper-ground fallback이므로 ARTIS의 정확한 phixs target 보존은 아닙니다([lumina_plasma.c:6736](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6736)).

- **에너지 분기 부호: PASS.** ARTIS와 동일하게 `U < nu_edge/nu`이면 MA, 보수확률이면 k-packet입니다([rpkt.cc:435](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:435), [rpkt.cc:442](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:442), [lumina_cuda.cu:5749](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5749), [lumina_cuda.cu:5751](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5751)).

- **난수 순서: 정상 route에서는 PASS.** ARTIS는 continuum-channel → CDF → `nu_edge/nu` 순입니다([rpkt.cc:382](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:382), [rpkt.cc:418](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:418), [rpkt.cc:435](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:435)). GPU도 [lumina_cuda.cu:5714](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5714) → [lumina_cuda.cu:3504](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3504) → [lumina_cuda.cu:5749](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5749)입니다. 다만 bin-center 불일치로 route가 실패하면 split 난수를 소비하지 않아 ARTIS 스트림과 달라집니다([lumina_cuda.cu:5755](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5755)).

- CPU는 ARTIS의 명시적 ff→k-packet 분기([rpkt.cc:402](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:402))도 없이 `chi_e+chi_bf`만 구성합니다([lumina_transport.c:538](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:538)).

### ② GPU 커널–CPU 동등성 — FAIL

Helper의 CDF 수식은 거의 동일하지만 이벤트 주파수 시점이 다릅니다.

- CPU: 이동 전 `comov_nu` 계산([lumina_transport.c:526](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:526)) → 패킷 이동으로 `r,mu` 변경([lumina_transport.c:561](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:561)) → 이전 `comov_nu`로 CDF/split([lumina_transport.c:570](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:570), [lumina_transport.c:573](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:573)).

- GPU: 이동 후 `r,mu`를 반영해 이벤트 위치에서 다시 계산합니다([lumina_cuda.cu:5590](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5590), [lumina_cuda.cu:5723](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5723)).

따라서 CPU는 stale `comov_nu`, GPU는 endpoint `comov_nu`를 사용합니다.

추가로 kinetic k-pool 조건도 CPU는 CDF 포인터만 검사하지만([lumina_transport.c:585](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:585)), GPU는 `d_kpacket_enabled`까지 요구합니다([lumina_cuda.cu:5775](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5775)).

### ③ OFF byte-불변 적대 수색 — 엄격 기준 FAIL

계산·RNG 경로만 보면 네 게이트 모두 구조적으로 PASS입니다.

- EVENT OFF: route table 미생성([lumina_plasma.c:6681](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6681)), legacy argmax 복귀([lumina_plasma.c:7333](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7333)), GPU legacy lookup([lumina_cuda.cu:5762](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5762)).
- MULTI_EDGE OFF: canonical `0`이 legacy alias보다 우선합니다([lumina_plasma.c:6461](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6461)); allocation·추가 난수는 ON branch 안입니다([lumina_cuda.cu:234](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:234), [lumina_cuda.cu:5030](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5030)).
- MA_J_UNCLAMP OFF: effective 값이 기존 cap/floor와 동일합니다([lumina_plasma.c:3478](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3478)).
- MA_NO_LINE_THERM OFF: 기존 LTHERM 조건과 동일하게 환원됩니다([lumina_cuda.cu:6808](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6808)).

하지만 stdout까지 “byte-불변”에 포함하면 FAIL입니다. 보존 현물은 EVENT OFF 여부와 무관하게 기존 D6 문구를 출력했지만([impl_withParityAA/orig/lumina_cuda.cu:6107](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:6107)), 현재는 OFF에서 다른 residual 문구를 출력합니다([lumina_cuda.cu:6445](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6445)). 실행 금지 조건상 산출 파일의 실측 bitwise 동일성은 확인하지 않았습니다.

### ④ D6 배너 정직성 — FAIL

`ENABLED/PARTIAL` 판단은 EVENT와 KPACKET만 봅니다([lumina_cuda.cu:6432](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6432)). 실제 기능의 필수조건인 `LUMINA_BF_OPACITY`는 별도로 결정되고([lumina_cuda.cu:6725](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6725)), OFF이면 BF 객체·route·GPU 배열이 전혀 초기화되지 않습니다([lumina_cuda.cu:7205](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7205)). 따라서:

`EVENT=1, KPACKET=1, BF_OPACITY=0` → 배너는 D6 `ENABLED`, 실제 D6는 inert입니다.

`PARTIAL`도 KPACKET 부재만 설명하고 BF opacity 부재는 누락합니다([lumina_cuda.cu:6439](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6439)).

### ⑤ 계보 5분류 — 4 PASS / 1 FAIL

1. **구 D6 = 배너 부채:** PASS. 보존 현물은 B4를 ground-only residual이라 인정하면서([impl_withParityAA/orig/lumina_cuda.cu:6095](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:6095)) D6를 구현된 것처럼 출력했습니다([impl_withParityAA/orig/lumina_cuda.cu:6107](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:6107)).

2. **CONTINUUM_EVENT = 신규 selector + 기존 target 데이터 재사용:** PASS. 기존 argmax 현물([impl_withParityAA/orig/lumina_cuda.cu:5450](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:5450))에서 현재 route table/CDF로 새로 확장됐습니다([lumina_plasma.c:6673](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6673), [lumina_cuda.cu:3487](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3487)).

3. **MULTI_EDGE = 순수 alias/rename:** FAIL. 기존 `LUMINA_KPKT_FB_MULTI` 구현이 이미 존재했습니다([impl_withParityAA/orig/lumina_cuda.cu:217](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:217), [impl_withParityAA/orig/lumina_cuda.cu:4727](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:4727)). 현재 canonical/legacy가 동일 accessor를 쓰는 것은 맞지만([lumina_plasma.c:6458](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6458)), 기존 BF-activated site의 flat thermal tail([impl_withParityAA/orig/lumina_cuda.cu:5569](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:5569))까지 shared Milne sampler로 변경했습니다([lumina_cuda.cu:5844](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5844), [lumina_cuda.cu:5886](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5886)). 정확한 분류는 “legacy canonicalization + 동작 확장”입니다.

4. **MA_J_UNCLAMP = 기존 cap/floor의 소비점 override:** PASS. 기존 계보([impl_withParityY/orig/lumina_plasma.c:2882](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityY/orig/lumina_plasma.c:2882))를 effective factor 0으로 우회합니다([lumina_plasma.c:3465](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3465)).

5. **MA_NO_LINE_THERM = 기존 LTHERM의 강제-off override:** PASS. 기존 parity kill 조건([impl_withParityAA/orig/lumina_cuda.cu:6461](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/impl_withParityAA/orig/lumina_cuda.cu:6461))에 독립적인 `!fix_no_ltherm`을 추가한 계보입니다([lumina_cuda.cu:6811](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6811)).