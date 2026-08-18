# withParityAA — N3 통일 게이트 `LUMINA_JBAR_DAMP_UNIFY` 구현 (2026-07-30, Opus) — **[운전석(Fable) diff 검수 통과 — 생산 인증=parity50(OFF-ident), 판정런=parity51(=1)]**

**운전석 검수 스탬프 (01시대)**: ①diff 전체에서 삭제행 = 1행(구 `jbar_damp=… }` — 실행문은 :8025에 문자 그대로 재보존, `}` 이동뿐) 직접 확인 ②EMA 루프 본문(:8050-8061) pre-gate와 문자 동일 — arm=1은 "f=1.0 강제로 루프 조건 false화"이지 루프 수정이 아님(최소 1지점 원칙 충족) ③arm=2는 jblue_prev 독립 버퍼·정규화 직후 배치 ④W/X/Y/Z 바이너리 md5 불변, AA=md5 7364d701/sha256 2d0cfec5, 클론 사본 byte-identical. 스펙-코드 충돌 0.

정본 설계 = docs/N3_JBAR_DAMP_UNIFY_DESIGN.md (축자 구현). 기준 트리 = withParityZ 상태 src/ 적층. 이 게이트 외 변경 0, 커밋 0, GPU 미사용. 빌드 make cuda exit 0, 신규 경고 0(기준선과 경고 diff CLEAN — 기지 g_fgemm_nulo 3건 동일). 증거 산출물 = `impl_withParityAA/`(orig 사본·diff 131행 3헝크·build.log·md5·branch 하니스+SELFTEST 표).

## 구현 요지
- 헬퍼 `lumina_jbar_damp_unify()` cuda.cu:835-849 (getenv 1회 latch; unset/0=OFF, 1=raw-통일, 2=EMA-통일, 그 외 수치=fail-closed OFF+stderr 경고, 비수치 문자열=atoi 0 관례).
- **arm=1 (권고, ARTIS-충실)**: EMA init 분기 내에서 `jbar_damp=1.0` 강제(:8032-8047) → 유일 EMA 지점 스킵 → jbar=raw=jblue 전 소비자 자동 통일. `LUMINA_COEVOLVE_JBAR_DAMP`(런처 0.5)를 명시 override, 배너에 원값 인용.
- **arm=2 (예비, ARTIS 근거 없음 명기)**: jblue 정규화 직후 동형 EMA(:8087-8133, jblue_prev 독립 버퍼, f=JBAR_DAMP; f∉(0,1)이면 NO-OP 경고).
- C6 계기(:8592-8711) 무접촉 — arm=1 배너가 자기검증법(resolve_ema≡resolve_raw) 안내만.

## 3분기 경로표 (요지)
OFF: EMA 실행(현행 분열 유지). =1: EMA 스킵 → C1 행렬·C2 MA·C3 IUP-JBLUE 전부 raw 1세대. =2: jblue에도 EMA → 전부 EMA 1세대.

## OFF-동일성 (블록-수준 실증 + 구조 논증)
- 구조: 3헝크 전부 게이트 분기 내 부수효과; OFF 추가 실행 = getenv 1회뿐(RNG 0·할당 0·배열 쓰기 0·stdout 0).
- 실측 branch 하니스(축자 추출 블록, 케이스별 별도 프로세스): OFF 4케이스 pre/post md5 IDENTICAL; DAMP=0.5+UNIFY=1 → 무감쇠 기준선 md5와 일치(raw 통일), =2 → 별도 md5(EMA 통일); max|jbar−jblue| = 2.142(OFF, 분열 보존)/0(=1)/0(=2). 배너 OFF 0회·armed 정확 1회·unknown 값 stderr 경고 발화.
- 한계(명시): 블록-수준 증명 — 최종 강제 = parity50 OFF-arm 산출물 byte-identity.

## 변경 전수
src/lumina_cuda.cu 단일 파일 3헝크(+107 −1): H1 :816-850 헬퍼, H2 :8026-8049 arm1 후크, H3 :8087-8133 arm2. 소스 md5: cuda.cu 52fe20b5(편집 전 7ace6d31), plasma.c/lumina.h=Z 상태 유지.

## 처분
- **parity50**(러너1, parity48 뒤): AA OFF-identity — config=parity46/48 복제, 유일 diff=바이너리 Z→AA. 기대=물리 산출물 parity48과 byte-identical.
- **parity51**(러너2, parity49 뒤): 판정런 B-arm = parity44 env + AA + `UNIFY=1`. 사전등록 = 설계 §5(W: 배너 1회+resolve_ema≡resolve_raw <1e-10 아니면 VOID / I-1: JBLUE-ANCHOR thin 버킷 |log-mean| 급감 / D-1: Si III·S III 들뜬준위 인구가 raw-솔브 쪽 이동 / M: b4·s8 이온분율·jbl_verdict·진동 계측 / HS-1: 마지막 3이터 변화 평균이 A-arm의 2배 초과·비수축 → 불안정 판정, 튜닝 금지, C-arm 회부 / HS-2: NaN·음수 pops·FORMAL-CONS 위반 / HS-3: W 실패=VOID). A-arm 대조=parity44 stdout 이터 시계열.
- 미해결 인계: HS-1 위험(고정점 반복의 무감쇠 lagged-Λ), N10(문턱 폴백 세대 혼식), N2=Y6 관할.
