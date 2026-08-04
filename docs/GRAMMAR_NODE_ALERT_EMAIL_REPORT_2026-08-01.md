# grammar 클러스터 불량 노드 알림 메일 발송 보고서

- 상태: **UNRESOLVED — 미발송**
- 수신: 김호영 선생님 `<inros@kias.re.kr>`
- 발신 계정: `kjhan`
- 제목: `[grammar] 노드 이상 보고: grammar072 (및 078/080 /home 마운트)`

## 메일 본문

김호영 선생님, 안녕하세요.

2026-08-01 12:00~16:10 KST에 일부 grammar 노드의 잡이 sbatch 배정 후 0초 내 FAILED되고, slurm 출력 파일도 생성되지 않았습니다.

grammar072는 전 경로와 출력이 /gpfs인 잡 398907도 즉시 실패하여, /gpfs 기록 자체가 안 되는 전면 이상으로 의심됩니다.

grammar078·080은 /home(NFS) 출력 잡에서 같은 증상이 발생해 /home 마운트 불량이 의심됩니다. grammar078의 전 경로 /gpfs 잡 398758은 2분 20초 만에 정상 완주했습니다.

증거 잡: 398907(072), 398745·398746(078), 398753(080), 398758(078 /gpfs 정상).

정상 대조로 grammar011의 /home 읽기·쓰기 프로브 잡 398747·398752는 모두 정상입니다.

해당 노드들의 /gpfs 및 /home 마운트 상태를 확인 부탁드립니다.

감사합니다.

김주한

## 발송 확인

- 가용 명령 확인: `which mail mailx sendmail`
- 확인 명령 exit code: `3`
- 확인 결과: `mail`, `mailx`, `sendmail` 모두 PATH에서 찾을 수 없음
- 발송 명령: 미실행(가용한 메일 명령 없음)
- `sendmail` 폴백: 미실행(`sendmail` 실행 파일 없음)
- 최종 결과: **UNRESOLVED — 메일을 발송하지 않음**
