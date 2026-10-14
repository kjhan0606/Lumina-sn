# O 생산 소스와 FREE_T 가속 계보 보존

2026-10-14, Codex. 사용자 승인: 프로젝트 평가 뒤 「오케이. 자네 생각대로 진행해」.

이 브랜치는 `cfbb008`에서 갈라진 `provenance/sprint-free-t-20261014`다.
실험에 쓰인 소스를 Git에 보존하기 위한 것으로, 기존 생산 소스의 물리적
정당성이나 R2 성공을 새로 판정하는 변경이 아니다. 기본 작업 브랜치로의
병합, 생산 소스 수정, 바이너리 재빌드는 수행하지 않았다.

## 첫 스냅샷: O의 생산 소스

- 원본: `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source`
- 목록과 기존 SHA256: `o_base_source.sha256`.
  원 기록은 `sprint_20261012/accel/base_source.sha256`이다.
- 목록에 있는 186개 파일을 상대 경로 그대로 복사하고 각각 `cmp`로
  원본과 바이트 일치를 확인했다. 목록 밖 파일은 `cfbb008` 상태를 유지한다.
- 저장소와 원본의 차이는 `src/` 14개 파일(7,695줄 추가, 174줄 삭제),
  기존 추적 Makefile·스크립트·시험을 포함하면 26개 파일(8,737줄 추가,
  284줄 삭제)다. 이외에 원본 목록의 시험 파일 5개도 새로 추적했다.
- O 바이너리: `coll_presc_20261009/OMEGA/run1/input/lumina_cuda`.
  기존 실행 기록 SHA256:
  `8ef03fb37b3e0de986f9c9ba15f0b7bbf8f98800d62ab18fb666ec518914f5d8`.

## 검증 범위

이번 복사는 파일 조회·편집·바이트 대조와 Git 기록만 수행했다.
위 SHA256는 기존 증거의 인용이며 이번 로그인 노드에서 재계산하지 않았다.
이 스냅샷에서 빌드·시험을 다시 실행하지 않았으므로 재빌드 동등성을
검증했다고 주장하지 않는다. 계산과 빌드는 Slurm에서 수행해야 한다.

## 두 번째 스냅샷: 결합 Newton

- O 기준 커밋: `703058f`.
- 원본: `/gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel/source`.
- `accel_manifest_paths.txt`의 189개 파일을 복사하고 각각 `cmp`로 원본과
  바이트 일치를 확인했다. 빌드 시점의 기존 SHA256 목록은
  `accel_source_sha_at_build.txt`에 보존했다.
- 정본 빌드: `accel/cuda_build3/lumina_cuda`.
  SHA256 `377e9171d239832e1d307a7547f9aad20f29057e6af6d8115ae4435399712ffd`.
- 원래 빌드 스크립트와 컴파일러 버전, 최종 CN 직접시험 출력(14/14)을
  이 디렉터리에 보존했다. A2-10 시험은 기존 `radeq_selftest_421342`의
  `baseline_pass=1 negative_expected_fail=16 total=17` 기록이다.
- 원래 빌드 스크립트의 절대 경로는 당시 생산 위치를 가리킨다.
  이 Git 스냅샷을 재빌드하려면 아래 재현용 러너를 사용한다.

```bash
# 반드시 계산 노드에서 실행. OUT은 아직 존재하지 않는 /gpfs 또는 /scratch 경로.
sbatch provenance/sprint_20261012/rebuild_accel.slurm "$PWD" /gpfs/kjhan/lumina/cmfgen_map/sprint_20261012/accel/git_rebuild_NEW
```

재현용 러너는 기존과 같은 컴파일 명령을 사용하고 빌드 전후에 기록된
소스 SHA256를 검사한다. 새 바이너리 SHA256는 별도 기록한다.
같은 소스·명령이라도 기존 바이너리와 바이트 동일하다고 미리 가정하지 않는다.
이번 정리에서는 재빌드를 제출하지 않았다.

R2 job `422361`은 기존 `accel/cuda_build3/lumina_cuda`를 사용하므로 이
브랜치 작업과 독립적으로 같은 바이너리·덱으로 실행된다. R2 이후 판독
job `422362`는 기존 판독기의 잠정 수치 산출이며 과학적 수용 판정이 아니다.
