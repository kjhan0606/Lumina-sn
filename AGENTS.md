## 정전 대비 종료 지시 (사용자, 2026-10-23)

- 오늘 10:50(KST)에 Lumina-sn 계산과 편집을 멈추고, 관련 변경을 커밋·푸시한 뒤 자율 진행을 중단한다. 서버 전원 종료는 사용자가 11:00에 수행한다.
- 10:50을 넘기는 새 계산은 시작하지 않는다. 부분 출력은 보존하고, 다른 프로젝트 작업이나 공용 Codex 프로세스는 종료하지 않는다.
- 종료 안전장치: 사용자 systemd 타이머 `lumina-poweroff-stop-20261023.timer`와 `scripts/lumina_poweroff_stop_20261023.py`. 실행 결과는 `docs/LUMINA_POWEROFF_2026-10-23.json`과 사용자 journal에 남긴다.
- 10:50 이후에는 이 goal을 사용자 요청에 따른 `paused`로 전환하고, 전원 복구 후 사용자의 재개 지시 전까지 계산·모니터링을 재개하지 않는다. 과학적 목표 달성으로 처리하지 않는다.

## 현행 업무 배정

- 코드 작성·구현과 직접 관련 자체 시험은 운전자(Codex)가 직접 수행한다.
- 기획·계획·평가는 Astra가 담당하며, 의미 있는 묶음 결정 때만 호출한다.
- 검증 2는 Fable이 담당한다(사용자 변경, 2026-10-19).
- 이후 사용자 지시로 Fable 의뢰는 취소됐다. 현재 검증 2 의뢰는 없으며, 재지시 전 추가 Fable 호출이나 다른 모델로의 대체 의뢰를 하지 않는다.
- 위 배정 이외의 다른 모델 업무와 과거 Fable 감사 절차, Q-GOAL·Q-LEAN 의뢰는 취소 상태를 유지한다. 검증 2 배정만으로 과거 감사 절차를 복구하지 않는다.
- 상세 정본은 CLAUDE.md 맨 위의 「배분 변경」이다. 아래 과거 배정보다 이 현행 배정이 우선한다.

## 현행 실행 위치 변경 (사용자, 2026-10-20)

- MPI 32랭크 이하 작업은 `lageunha` 서버에서 수동 실행한다.
- MPI 랭크와 OpenMP 스레드는 구분한다. 이 지시만으로 기존 작업의 CPU·메모리·시간·실행 횟수 한도를 확대하지 않는다. 실행 전 서버 부하와 해당 프로젝트·입력 경로를 확인한다.
- 이미 실행 중인 OpenMP16 Slurm1200705를 이 위치 지침만으로 중단·중복 실행하지 않는다. 현재 봉인 입력과 승인 한도로 마무리한다.

<!-- 아래 본문은 2026-09-10~17 Codex 운전 기간의 이력으로 보존한다.
과거 작성자와 판정은 변경하지 않으며, 현행 배정과 충돌하는 분장·감사 의무는 적용하지 않는다. -->

# Lumina-sn project workflow

## Ownership

- Use one owner per task by default. Do not add routine support or review roles.
- 운전자는 기획·계획 초안, 저장소 통합, 실행, 증거 추출과 감사 재현을 담당한다.
- Fable이 기획·계획·설계 및 계약 변경 감사를 담당한다
  (사용자 변경, 2026-09-16). 이전의 자문관 전용 배정을 대체한다.
- 운전자의 통합 확인·증거 재현을 Fable 감사로 표시하거나 같은 범위의 정례
  중복 감사로 만들지 않는다.
- 운전자가 코드 변경과 직접 관련 자체 시험도 담당한다.
- GPT-5.6-sol은 더 이상 이 프로젝트의 구현·직접시험 담당이 아니다
  (사용자 변경, 2026-09-16). 과거 Sol 작업과 산출물의 작성자·정체성은 이력으로
  보존한다.
- Opus5에는 구현된 코드의 read-only 검토와 각 묶음 종료 검토를 자문으로
  의뢰한다(사용자 변경, 2026-09-16). Opus5의 의견은 공식 판정이 아니며,
  운전자가 최종 단계종료감사와 PASS, BLOCKED, REWORK를 판정한다.
- Codex is the operator. It owns repository integration, calculation staging,
  Slurm execution and monitoring, evidence extraction, reproducibility checks
  of audit findings, concise status reports, and handover notes. Opus5 자문과
  운전자 자체 종료감사는 서로 다른 성격으로 기록한다.
- The user owns priority, scope, cost, and follow-up decisions.
- Historical user-directed exception (2026-09-10): Fable performed the
  completion audit of `BUNDLE_LINE_NET_SHELL_2026-09-10`. Preserve that verdict
  and its authorship; do not add a duplicate audit or relabel it as operator-authored.

## Bundled stages

- User clarification (2026-09-11): Bundle all work by the intended outcome,
  not by a succession of small checks. This is a general rule; the user's
  example of an ionization distribution does not select a particular element,
  shell, or new scientific scope.
- Name each bundle for its objective and state the observable end result and
  the decision it must enable. Analysis, implementation, direct tests,
  integration, execution, and comparison are internal work toward that result,
  not separate deliverables that each justify closing a bundle.
- Do not declare a bundle complete merely because a diagnostic, rate
  calculation, fixture, gate, or code change passed. The completion audit must
  judge whether the stated objective was achieved using the required end-to-end
  evidence. If it was not achieved, report the unmet objective explicitly;
  do not replace it with an easier intermediate objective after seeing results.
- Keep working within the authorized scope and resource envelope until the
  outcome is reached or a genuine blocker requires a user decision. Request
  additional authority only for material scope, cost, or scientific-contract
  changes; do not create routine approval pauses between internal tasks.
- Write new user-facing explanations and project documentation in Korean.
  Keep code identifiers, technical names, and existing historical audit records
  unchanged where needed; do not mix Chinese or Japanese prose into new text.
- Do not audit every small implementation or execution step separately.
- Group related steps into the smallest coherent **bundle** that can produce a
  meaningful, auditable outcome. A bundle may include implementation, direct
  self-tests, integration, staging, execution, and evidence extraction when
  those steps serve one acceptance decision.
- Do not combine unrelated scientific questions merely to reduce audit count.
  Split work when objectives, physical assumptions, rollback boundaries, or
  acceptance decisions are independent.
- Before execution, each bundle record must state its objective, scope and
  exclusions, inputs and sealed baselines, planned changes, execution owner,
  measurable acceptance criteria, expected evidence, resource/cost envelope,
  rollback or preservation rule, and reproducibility commands.
- Every stage or bundle planning/design audit, including contract amendments,
  must open with these questions, in this order, before detailed criteria,
  instrumentation, gate, implementation, or resource review:
  1. **Q-GOAL:** Does this stage's objective directly serve the project's final
     objective? State the connection and what would be lost by omitting it.
  2. **Q-LEAN:** Are unnecessary measurements or excessive gates being added?
     Identify what can be removed or combined and who actually consumes each
     retained measurement or gate.
  Resolve these questions first. If the objective is misaligned or the design
  is unnecessarily heavy, reduce or revise the plan before detailed review;
  do not merely append favorable labels at the end. Q-GOAL must conclude
  `PASS`, and Q-LEAN must conclude
  `APPROPRIATE`, before execution may start. If either conclusion is absent or
  negative, the operator revises the design and Fable reassesses it; do not obtain a
  favorable label by weakening or omitting the question.
- Internal checkpoints are operational controls, not separate audits. The
  operator may stop a bundle immediately on a fail-closed gate without
  requesting a new audit for every checkpoint.

## Audit sequence

1. **Fable 기획·계획 감사:** 먼저 Q-GOAL, 다음 Q-LEAN을 질문하고 물리 가정·범위·
   수용 기준·증거·자원을 검토한다. 계약 변경도 같은 순서다.
   Q-GOAL=`PASS`, Q-LEAN=`APPROPRIATE`와 Fable의 설계 판정을 기록한다.
2. **구현과 직접시험:** 운전자가 구현과 직접 관련 자체 시험을 수행한다.
3. **Opus5 read-only 자문:** 정확한 변경 소스와 직접 시험 증거를 Opus5에
   제공해 구현상 위험과 다음 확인을 자문받는다. 자문은 실행을 자동 승인하거나
   공식 판정을 대신하지 않는다.
4. **Bundle execution:** Codex integrates, stages, executes, monitors, and
   extracts compact evidence for the related work as one bundle.
5. **운전자 자체 종료감사:** 계획한 묶음 출력이 갖춰지면 운전자가 목표 달성,
   사전 수용 기준, 증거 정체성과 일관성, 물리 해석 범위, 자원·원본 보존,
   미해결 사항을 함께 검토하고 PASS, BLOCKED, REWORK를 판정한다.
   작은 확인마다 별도 감사를 추가하지 않는다. Opus5 자문 의견은 근거로
   기록하되 운전자가 최종 판정한다.
6. **문제 처리:** 운전자가 실패를 재현하고 구현/실행·증거/설계·기준 문제를
   구분한다. 기획·계획·물리 기준 또는 코드 변경이 필요하면 해당 범위의 Fable
   감사로 연결하되 작은 확인마다 별도 감사를 추가하지 않는다.
   Opus5 자문에 동의하지 않더라도 근거와 재현 결과를 기록하고 운전자 판단을
   갱신한다.
7. **Operator reproducibility:** Codex independently runs the recorded
   reproduction commands against the exact source, binary, inputs, run root,
   and evidence paths used by the evidence producer. Codex reports reproduced,
   not-reproduced, or blocked, with hashes and minimal decisive output. Keep
   scientific acceptance judgment distinct from executable reproduction; do
   not represent operator self-review as an independent external-model audit.

An audit result is not closed until its executable claims have an operator
reproducibility status. Reasoning-only judgments must cite the fixed evidence
and criteria so the operator can verify the evidence identity and premises.

## Evidence handoffs

- 운전자가 간결한 묶음 계획과 판단 근거를 기록하고 Fable 기획·계획 감사에
  전달한다. Opus5 자문에는 관련 기준·정확한 변경·직접 시험·신원을 전달한다.
  전체 로그나 광범위한 맥락을 정례적으로 보내지 않고 같은 목적의 검토를 묶는다.
- For code review, inspect only the exact change, direct self-test evidence,
  identities, and the design criteria the change implements.
- 운전자는 사전 감사된 묶음 계획, 변경 파일·바이너리 신원, 시험·Slurm 결과,
  핵심 물리량, 실패 분류와 정확한 증거 경로를 한 완료 기록으로 정리하고
  Opus5 자문 의견, 자체 종료감사 판정과 재현 상태를 남긴다. 외부 종료감사
  판정 전달물은 별도로 요구하지 않는다.
- If conditional confirmation is triggered, inspect only the failed
  criterion, the original design-audit record, disputed evidence, exact
  reproduction command, and relevant code context. Do not repeat a full-bundle
  review unless the failure is bundle-wide.
- Record all audit verdicts and operator reproducibility outcomes in the
  project handover or the bundle's durable evidence document.
- Check a validation runner's output root before invoking it. Run writer-capable
  runners in isolated trees, or preserve scoped byte/hash backups of existing
  target artifacts first. A read-only source path does not imply read-only execution.

## Availability and blocking

- Fable 감사자를 조용히 다른 모델로 대체하지 않는다. Opus5는 자문이므로
  호출이 실패해도 운전자 최종 판정을 대신 꾸미지 않는다.
- 필수 Fable 감사가 불가능하면 판정을 꾸미지 않고 해당 새 결정은 `AUDIT_PENDING`으로
  기록한다. 가역적 준비와 이미 승인되어 실행 중인 계산·증거 추출은 계속할 수 있다.
  미감사 새 실행은 실제 감사 또는 사용자의 명시적 예외 승인 뒤 진행한다.
- 09-15의 선택 자문 실패 시 진행 허용은 선택 자문에 대한 지시로 보존한다.
  이를 09-16에 다시 배정한 필수 감사를 받은 것으로 해석하지 않는다.
- Opus5 자문 실패는 `CONSULTATION_PENDING`으로 기록할 수 있으며, 이는 운전자
  최종 판정의 대체가 아니다.
- 자체 종료감사 실패가 범위 밖 코드 변경이나 재실행을 자동 승인하지 않는다.
  실패를 재현·분류한 뒤 기존 권한 범위 내에서 담당자에게 넘기며, 의미 있는
  범위·비용·과학적 기준 변경은 사용자 결정을 받는다.

## Transition rule

- 사용자 변경(2026-09-16, 최신): 기획·계획 감사는 Fable이 수행한다.
  구현된 코드 read-only 검토와 묶음 종료 검토는 Opus5에 자문으로 의뢰한다.
  공식 단계종료감사와 PASS, BLOCKED, REWORK 판정은 운전자가 수행한다.
  운전자는 구현·직접시험·실행·증거 재현도 담당하며, Sol은 이 프로젝트의 현재
  담당에서 제외한다.
  과거 감사의 작성자·판정은 보존한다. 이미 진행 중인 session8465는 봉인 입력과
  기존 기준으로 마무리하며 역할 변경만을 이유로 중단·재설계·재실행하지 않는다.
  다음 계획 또는 코드 변경부터 최신 배정을 적용한다.

- 과거 사용자 변경(2026-09-15, 위 09-16 변경으로 대체): Fable을 감사자에서 자문관으로 전환한다. 진행 중인
  묶음에도 즉시 적용하며 운전자가 기획·계획·설계 및 자체 감사 판단을 책임진다.
  아래의 과거 Fable 감사 배정보다 이 규칙이 우선한다. 이미 끝난 Fable 감사의
  판정·작성자는 역사로 보존하며 재명명하거나 역할 변경만으로 재감사하지 않는다.

- 과거 사용자 변경(2026-09-13, 09-16 최신 변경으로 일부 대체): 미완료 묶음을
  포함해 종료감사는 운전자의 자체 감사로 마무리한다. 당시 Opus5 종료감사를
  폐지했던 기록은 보존하되, 현재는 Opus5 자문을 받을 수 있다.
  이미 끝난 과거 감사의 판정과 작성자는 보존한다. 아래 과거 Opus5
  배정 기록보다 이 지시가 우선하며 역할 변경만으로 물리 계약을 재설계하지 않는다.

- Historical user reassignment (2026-09-12, superseded above): completion
  audits not yet performed belong to Opus5, including the in-flight converged
  ionization-distribution bundle. Fable planning/design audits and GPT-5.6-sol
  implementation responsibilities are unchanged. Preserve completed operator,
  Fable, and Opus5 verdicts and their authorship; do not re-audit completed work
  merely because the completion-audit owner changed.

- The 2026-09-10 reassignment of planning/design audits to Fable applies to new
  planning/design audits from that instruction onward. Do not relabel completed
  operator audits or rerun completed work merely because the auditor changed.

- This workflow applies to bundles designed after 2026-09-05 and supersedes
  the former per-step handoff and exceptional-cross-check workflow.
- Work already in flight keeps its recorded inputs and acceptance criteria; do
  not retroactively redesign or rerun it merely to satisfy this workflow. When
  it finishes, package it as one bundle for operator self-completion audit and
  reproducibility. In particular, H200 job `332010` remains governed by its
  sealed run contract and enters the new completion-audit sequence only after
  its outputs exist.
