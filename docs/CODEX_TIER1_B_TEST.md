빌드·MD5 검사는 통과했지만, 후속 `strings`와 CSV 독립 합산은 실행 환경 장애로 완료하지 못했습니다. 미확인 항목을 통과로 과장하지 않습니다.

### 1. 클린 재빌드

```sh
make clean
# exit 0

make cuda
# exit 0
```

- 신규 바이너리: `lumina_cuda`
- 크기: `3,720,288 bytes`
- MD5: `2f3ab89ddcf78b779f75f25ca196c182`
- ELF 64-bit, dynamically linked, not stripped
- 경고: `src/lumina_nlte_gemm.cu:75`의 미사용 변수 경고가 아키텍처별로 반복됐으나 링크 성공
- 소스 수정 없음
- GPU 실행·큐 제출 없음

### 2. 기존 parity 바이너리 MD5

전·후 자동 비교 결과:

```text
PARITY_MD5_COMPARE_EXIT=0
COUNT_BEFORE=26
COUNT_AFTER=26
```

| 바이너리 | MD5 전=후 |
|---|---|
| withParity | `0b5787dd0e1454250cf176cb6c6442f1` |
| withParityAA | `7364d70113e7b699545211031cac14fc` |
| withParityD | `0d7b5cd041e0d729309d666b580fbeaa` |
| withParityE | `d38a9e26cb8b4158f8d5b5e5610230f2` |
| withParityF | `27b8983290f2b965024b628dfe1fe264` |
| withParityG | `3b868714416f2c4e4196d0f2917e296f` |
| withParityH | `796a6471bf711eb58604c731e4533485` |
| withParityI | `5c9ed9f3e9461908103e003417acd1c6` |
| withParityJ | `ba72684d41668763987d1b73d76c884a` |
| withParityK | `1c77078e7080809ee862802addd2c26f` |
| withParityL | `20c25ab0119e5a6be78fa2d3f0d061bb` |
| withParityL2 | `43c61ad8705cd65bc50da83f1131998f` |
| withParityM | `6d98194e33aa008ac9e26e01dbb44fad` |
| withParityO | `cc3f1913cd78022b1f1d714a563fb104` |
| withParityP | `f28ed7b189adac661b5c0e2686d6dc7f` |
| withParityQ | `d32ceba9a9841d983f6fed04d7132f68` |
| withParityR | `39426c5d1fcc81b9ba2ff3be768742fd` |
| withParityS | `63e4dd32392ddd9834c60d85fae84d89` |
| withParityT | `1f1672d15014ba3faa23b50da704c6ff` |
| withParityT2 | `f9b2523685a39f78723985651cbc8a73` |
| withParityU | `8ab3a151fe037fe3dda21e29f1fbefd8` |
| withParityV | `1c4cd1994dbf42d96d457b1a3523f560` |
| withParityW | `498b737649ff4b02c26c793abdfba6ce` |
| withParityX | `f18c53a1044e3aede268c2ca9577c14e` |
| withParityY | `3477fa7ff679e28728993e4005aacdce` |
| withParityZ | `b02d1da55b3fc6506946edc855df58d1` |

### 3. 문자열 검사 상태

소스에는 다음 수리 내용이 확인됩니다.

- 새 D6 문구: [lumina_cuda.cu:6172](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6172)
- `[FB-EDGE]` 포맷: [lumina_cuda.cu:2247](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:2247)
- pure-CMFGEN 반환 직전 FB-EDGE 호출: [lumina_cuda.cu:8825](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8825)
- `L_total_in` 포맷과 산식: [lumina_plasma.c:16913](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16913)
- 구 허위 문구는 현행 `src/`에서 사라졌고, 역사 바이너리에서는 다음으로 특정됐습니다.

```text
D6 bf event nu_edge/nu ioniz-vs-kinetic split + stim-recomb corr
```

다만 신규 ELF 대상 실제 `strings` 명령은 실행하지 못했으므로 이 항목은 아직 미통과 상태입니다.

### 4. deposition 판정

코드 의미는 결정적입니다.

```text
L_dep = Σ(heating_rate [erg/s/cm³] × shell_volume [cm³])
L_total_in = L_inj + L_dep
```

따라서 후보들의 관계는:

```text
7.787639e42 + 3.092361e42 = 1.0880000e43 erg/s
```

즉:

- `7.787639e42 erg/s` = deposition 합 `L_dep`
- 약 `3.092361e42 erg/s` = `L_inj`
- `1.088e43 erg/s` = `L_inj + L_dep`, 즉 `L_total_in`

하지만 CSV 헤더·컬럼을 직접 읽고 독립 합산하는 명령은 실행하지 못했으므로, 요청하신 “CSV 독립 합산 검증” 자체는 미완료입니다.

차단 원인은 빌드 후 명령 실행 래퍼가 사라진 환경 오류입니다.

```text
Unable to spawn .../codex-linux-sandbox: ENOENT
```

환경이 재연결되면 남은 것은 신규 ELF `strings` 4건과 CSV 합산 명령뿐입니다.