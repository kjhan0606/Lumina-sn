# 외부 원자데이터 다운로드 보고 (2026-07-29 밤, Opus) — **[확정: fable 적대 재검증 CONFIRMED 9 / CORRECTED 1 / REFUTED 0, BLOCKING 없음 — 2026-07-29 밤]**

**fable 스탬프** (`FABLE_VERIFICATION.md` + fable_checks*.py/csv, 전부 원시 바이트 독립 재계산): 헤드라인 확증 — 피크 원본 실재·클립 정확 2점(가수 보존 raw-토큰 확인)·237엔트리 **전수** 대조(217 identical/19 절단 19/19 규칙 일치/1 변조/no-match 0)·2점=78.8% 운반·인증 로그 3값 정확 재현.
**유일 실질 정정 [C8-CORRECTED]**: "Ca V p-파일 field3=전 블록 상수 0.246793"은 과일반화 — 실측: 싱글릿 164블록만 0.246793, 트리플릿 223+퀸텟 55=278블록은 0.0(지상 ³Pᵉ 포함). 함정의 실질 결론(f3≠에너지, e-파일 사용)은 오히려 강화.
**미세 주석 3** (판정 불변): ①3위값 위치 1.955588 Ry(본문 1.955590은 전사 오류) ②OP 문턱 714.13 Å(본문 714.15) ③Δlnν 중앙 6.691e-5는 NORAD E열 기준(CMFGEN x열 6.6935e-5), 79.2× 불변.

발주 근거: B16 (user 승인 "B16 외부데이터 다운로드 해요"). 수납처: `data/atomic/external/` 전용 (기존 수정파일 14건 무접촉 확인).

## 1. NORAD Ni II — 완전 확보
`data/atomic/external/norad_ni2/` (126 MB), 2026-07-29 13:59:12Z–13:59:26Z, `curl -sL`, bytes as served. URL `https://norad.astronomy.osu.edu/ni2/<name>`.

| file | bytes | sha256 |
|---|---|---|
| ni2.px.gd.txt | 56019 | 9ad2c6421d5da2788efccd2a4e1d96cfbaa55fed95ae0642b054945973cce328 |
| ni2.px.txt | 53156229 | b2a9d72ac01521267a257e8589ddd397342e443a64a37e3ce696650d8197247f |
| ni2.ptpx.txt | 77416568 | 1c5f6aa66c97536c5e3370fc514be22e757f71e88ae9fab7cc054405f90304ca |
| ni2.en.ls.txt | 46237 | 5e17090471fbdad947d0574fc7ccd9f62aef73366bd4ec3a5f2a81f81070689d |
| ni2.rrc.txt | 548605 | 07627d787477a645bca28563b51f368d1082483a1ed9ccdadc8a47daa612cd27 |

- **왕복 증명**: ni2.rrc.txt = 저장소 `data/atomic/dr_norad/raw_ni2.rrc.txt`와 byte-identical (`cmp` clean) — 다운로드 경로 충실성 입증.
- **인용 분리 (헤더 verbatim)**: px.gd/px = Bautista 1999 A&AS 137,529 (TOTAL σ); ptpx/en.ls/rrc = Nahar & Bautista 2001 ApJS 137,201 (PARTIAL σ, energies, RRC). 병합 금지.
- **단위 (헤더)**: col1=Ry, col2=Mb. BE는 px/px.gd 음수, ptpx 양수. **실측 함정**: ptpx의 데이터행 수는 두 번째 정수(`nr`) — `nr`로 걸어야 `0 0 0 0` 터미네이터가 EOF 정확 착지 (533 blocks, 3,095,334 pairs).

## 2. Ni II 지상 감사 — 피크는 원본 실재, "값 몇 개 수정"은 문자 그대로 2점
CMFGEN `NICK/II/19apr23/phot_data_A` entry 0 = `3d9_2De`, type 20, 2166 pts ↔ NORAD `ni2.px.gd.txt` SLpi (2,2,0,1), BE −1.27605 Ry, ntot 2166. CMFGEN x = E_Ry/|BE| (max|Δx|=5.0e-8).

- **(a) 최대 σ**: NORAD 지상 최대 = **1.0290e+07 Mb = 1.0290e-11 cm²** @1.906326 Ry. CMFGEN의 1.3610e+06 Mb = **NORAD 3위값 verbatim** @1.955590 Ry. CMFGEN 파일 전체 최대 = 6.6560e+06 Mb (`3d8(3F)6s_2Fe`) = NORAD와 bit-identical. **판정: 임포트/클리핑 아티팩트 아님 — 원본 실재.**
- **(b) 격자**: N=2166 동일. Δlnν 평균 4.4755e-4, 중앙값 6.691e-5 → LUMINA(5.3e-3) 대비 **11.8×/79.2× 촘촘** (커버리지 보고 재현). CMFGEN은 OP 문턱(1.27605 Ry, 714.15 Å)을 NIST 18.169 eV/682.4 Å로 재스케일, 비 1.046499.
- **(c) ∫σdν [1.5e14, 3e16] Hz**: trapz(자체 노드) CMFGEN 5.3750 vs NORAD 25.8941(재스케일)/24.7435(원래대로); external_anchors.py 방식 CMFGEN **5.3941**(인증 로그 I_grid 정확 재현, α_gnd 3.0631e-14 정확 재현) vs NORAD **25.491** → α_gnd 1.4495e-13 (**×4.73**). ※발주문의 "5.445"는 로그값 아님 — `external_anchors_log.txt:9` = 5.3941e+00 [정정].
- **(d) 클립 = 정확히 2점, 둘 다 지상, raw-line 수준 검증**:
  ```
  NORAD 1.815627E+00 2.778E+06  →  CMFGEN 1.4228494 2.7780E+04  (÷100)
  NORAD 1.906326E+00 1.029E+07  →  CMFGEN 1.4939274 1.0290E+06  (÷10)
  ```
  가수 보존·지수 축소. 둘 다 단일-노드 스파이크 (idx 494: 이웃 9.9e3/8.2e3 Mb; idx 616: 이웃 4.5e5/2.4e5 Mb). 이 2점 복원 시 I_grid 25.492/α 1.4495e-13 — **2166점 중 2점이 원본 지상 ∫σdν의 79% 운반.**
- **(e) 전수 분류 (237 type-20 엔트리)**: 237/237이 NORAD `ni2.px.txt` 블록과 대응; **217 bit-identical 전장; 19 꼬리절단**(절단 인덱스 = NORAD E열이 증가를 멈추는 첫 인덱스와 정확 일치, 비단조 꼬리 73–148점 낙하, 보존점은 전부 bit-identical); **1 (지상)만 값 변조**. 스팟체크 4엔트리 0 differing points. 주의: 매칭은 σ 지문 기반, 25엔트리는 중복 블록 탓 후보 >1.
- ni2.px.gd.txt는 ni2.px.txt 내 ntot=2166 블록과 σ 동일(중복 추출본). ptpx 지상 ²Dᵉ ns=1: 8534 pts, max 5.8e5 Mb (partial — 채널합 없이 total과 비교 불가).

## 3. TOPbase Ca III/IV/V — 완전 확보 (raw-file 경로)
`data/atomic/external/topbase_ca/` (3.2 MB), 14:06:59Z–14:07:20Z. `https://cdsweb.u-strasbg.fr/topbase/{p,e,f}/{p,e,f}20.{18,17,16}.gz` (9파일, sha256=PROVENANCE.txt, gzip mtime 전부 2002-09-17). CGI 스크립트 성공 레시피 기록(`com=dt` 직접; 필드=정수범위 필수); CGI 산출 = p20.18.gz 블록과 동일 쌍 확인.

- **단위 확립 (가정 아님)**: 에너지=Ry (폼 라벨+CGI `E(RYD)`); `SLPI=(2S+1)*100+L*10+P`. σ 단위는 TOPbase 미표기 → **왕복으로 확립**: Ca III 지상 E_th 3.68369 Ry에서 p20.18 선형보간 **9.6121** vs CMFGEN `CA/III/10apr99/phot_smooth.dat`(Megabarns, "Topbase93") **9.6080 Mb → 비 1.0004**. Ca IV 1.484, Ca V 0.875 (CMFGEN은 3000 km/s 가우스 스무딩이라 느슨) — 3건 모두 Mb 확인.
- **함정 2건 기록**: (i) p-파일 3번째 레코드는 준위 에너지가 아님 — **Ca V는 전 블록 0.246793 상수**; 에너지는 e-파일/CGI 사용 (Ca V 지상 −6.23865 Ry). (ii) 지상 격자는 문턱보다 정확히 z²/100만큼 아래서 시작 (0.089995/0.159996/0.250002, n*=10); 여기준위는 오프셋 비상수 (Ca III −0.021361…+0.090004 Ry).
- **내용**: Ca III 178준위/19,881쌍 (지상 ¹Sᵉ, 3.68369 Ry=247.38 Å, NP 122, Δlnν 중앙 2.00e-2 — **LUMINA보다 성김**); Ca IV 322/109,674 (4.99190 Ry, NP 455, 2.39e-4, 22× 촘촘); Ca V 442/278,060 (6.23865 Ry, NP 643, 1.90e-4, 28×). σ_max: 7.589e2/5.358e3/2.434e4 Mb.

## 4. 미확보
- NORAD ni2.fls.txt/f.forbid.txt/omrx.txt = HTTP 404 (광이온 데이터 아님).
- TOPbase Ca III/IV/V는 결손 없음. 서버 README/format 부재(404) — 형식은 실증+CGI 헤더로 확립.
- 채택 판단 없음 (에이전트 소관 외).

## 처분 (fable 관문 후)
- fable CONFIRM 시: 원장 기재 + B16 표 갱신. **채택/보정/튜닝 금지** — 국면 규칙+「틀린 값 조용히 기재」 규칙. Ni II σ 채택 여부는 별도 결정(앵커 B + 출판 대조 후).
- CMFGEN의 2점 클립(의도적 수기 편집 양상)은 잣대-충실성 축의 신규 사실 — K6/빈티지 대장과 연결.
