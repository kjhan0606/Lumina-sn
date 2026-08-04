C 리뷰의 지정 FAIL을 해소했습니다.

- 실제 사건 `nu_cmf`에서 CPU/GPU route 총합과 CDF를 재구성합니다. Kramers σ와 stim 항은 실제 주파수를 직접 사용하고, CMFGEN bin-average σ는 log-ν 보간합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6535), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3510)
- `nu_edge > nu_cmf` route가 원천 제외되어 CPU/GPU `p_ion` clamp를 함께 제거했습니다.
- CPU도 이동 후 endpoint Doppler/`nu_cmf`를 사용하며, 실제 k-packet enable 상태까지 검사합니다: [lumina_transport.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:565)
- route 실패 시에도 CDF → split 난수를 ARTIS 순서로 소비합니다: [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5802)
- phixs target 부재 route를 표시하고, 실제 upper-ground 대체 MA 활성화를 CPU/GPU 카운터로 출력합니다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6708), [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2178)

CDF 형태, mapped target, 분기 부호, 정상 route 난수 순서는 유지했습니다.

검증 결과:

| 셀 | eligible SHA-256 | full CSV SHA-256 |
|---|---|---|
| s0 | `beaac19b…678cfa` | `4789f13c…1952` |
| s8 | `54f9fafa…d05f` | `a4f1a146…38e2` |
| s43 | `b971a038…80804` | `c48d2619…72aa` |

- unset 대 명시적 OFF: 세 셀 모두 `cmp=0`
- CPU CDF probe: edge 적대 사례 각 10,000회 PASS, 선택률 `0.296625` 대 기대 `0.295737`, `z=1.230`
- 실패 route CDF 난수 소비: PASS
- `make -B bench_frozen_oracle`: PASS
- `make -B cuda`: PASS, 기존 경고만 발생
- GPU 실행 및 git 명령: 미실행

상세 기록: [CODEX_WAVE2_1_A_REPAIR.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE2_1_A_REPAIR.md)

참고로 일반 CPU `make lumina`는 기존 Makefile의 C11 `M_PI`/GPU stub 링크 문제로 여전히 실패하지만, 변경 CPU 소스 syntax check와 oracle/probe 빌드는 통과했습니다.