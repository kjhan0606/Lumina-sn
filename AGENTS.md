# Lumina-sn project workflow

## Ownership

- Use one owner per task by default. Do not add routine support or review roles.
- Codex as operator owns project planning, scientific acceptance criteria,
  bundle design, the pre-execution design audit, code review for bundle code
  changes, and any confirmation audit triggered by a problem in the Opus5
  completion audit. Fable has no assigned project role.
- GPT-5.6-sol owns code changes and the directly relevant self-tests. It does
  not perform bundle design, code review, completion audit, or confirmation
  audit.
- Opus5 owns the completion audit after a bundle has finished.
- Codex is the operator. It owns repository integration, calculation staging,
  Slurm execution and monitoring, evidence extraction, reproducibility checks
  of audit findings, the audits and reviews assigned to the operator above,
  execution of the Opus5 completion-audit workflow, concise status reports,
  and handover notes. Codex invokes and operates the Opus5 audit; Opus5 remains
  the author of its verdict.
- The user owns priority, scope, cost, and follow-up decisions.

## Bundled stages

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
  negative, the operator revises and re-audits the design; do not obtain a
  favorable label by weakening or omitting the question.
- Internal checkpoints are operational controls, not separate audits. The
  operator may stop a bundle immediately on a fail-closed gate without
  requesting a new audit for every checkpoint.

## Audit sequence

1. **Operator design audit:** Before a new bundle starts, Codex as operator first
   asks and answers Q-GOAL, then Q-LEAN. Only after resolving these opening
   questions does it review the operator-authored bundle design, physical
   assumptions, exclusions, acceptance criteria, evidence plan, and resource
   envelope. This order also applies to every stage planning and contract
   amendment audit. Execution
   starts only after Q-GOAL=`PASS`, Q-LEAN=`APPROPRIATE`, and an overall design
   PASS.
2. **Implementation and code review:** GPT-5.6-sol implements and runs directly
   relevant self-tests. If the bundle changes code, the operator reviews the
   exact change and direct self-test evidence before staging execution.
3. **Bundle execution:** Codex integrates, stages, executes, monitors, and
   extracts compact evidence for the related work as one bundle.
4. **Operator-run Opus5 completion audit:** After all planned bundle outputs
   exist, Codex invokes and conducts the completion-audit workflow with Opus5.
   Opus5 audits the complete evidence against the pre-audited criteria and
   returns PASS, BLOCKED, or REWORK; Codex records the invocation, evidence
   identity, and verdict without substituting its own judgment.
5. **Conditional operator confirmation audit:** If the Opus5 audit fails,
   reports an inconsistency, or exposes a material problem, the operator audits
   the disputed finding against the original design and classifies it as an
   implementation defect, execution/evidence defect, design/criterion defect,
   or auditor error. Do not perform this audit after an ordinary Opus5 PASS.
6. **Operator reproducibility:** Codex independently runs the recorded
   reproduction commands against the exact source, binary, inputs, run root,
   and evidence paths used by the auditor. Codex reports reproduced,
   not-reproduced, or blocked, with hashes and minimal decisive output. Codex
   does not replace the scientific judgment of Opus5.

An audit result is not closed until its executable claims have an operator
reproducibility status. Reasoning-only judgments must cite the fixed evidence
and criteria so the operator can verify the evidence identity and premises.

## Evidence handoffs

- The operator drafts and audits the compact bundle card before execution. Do
  not send complete logs or broad repository context when objective, criteria,
  hashes, exact paths, and selected decisive lines are sufficient.
- For code review, inspect only the exact change, direct self-test evidence,
  identities, and the design criteria the change implements.
- Codex passes Opus5 one completion package per bundle containing the
  pre-audited bundle card, changed-file and binary identities, test and Slurm
  outcomes, key physical metrics, failure classification, and exact evidence
  paths, and records that operator-run audit in the durable handoff.
- If conditional confirmation is triggered, inspect only the failed
  criterion, the original design-audit record, disputed evidence, exact
  reproduction command, and relevant code context. Do not repeat a full-bundle
  review unless the failure is bundle-wide.
- Record all audit verdicts and operator reproducibility outcomes in the
  project handover or the bundle's durable evidence document.

## Availability and blocking

- Do not silently substitute one model for Opus5 or GPT-5.6-sol.
- If a required auditor is unavailable, preserve completed work and mark the
  bundle `AUDIT_PENDING`; do not claim the missing audit verdict. Continue only
  with reversible preparation that does not cross the unaudited decision gate,
  unless the user explicitly overrides it.
- A failed Opus5 audit does not automatically authorize code changes or a
  rerun. Complete the operator's conditional confirmation audit and
  reproduction first, then return the classified finding to the appropriate
  owner or the user.

## Transition rule

- This workflow applies to bundles designed after 2026-09-05 and supersedes
  the former per-step handoff and exceptional-cross-check workflow.
- Work already in flight keeps its recorded inputs and acceptance criteria; do
  not retroactively redesign or rerun it merely to satisfy this workflow. When
  it finishes, package it as one bundle for Opus5 completion audit and operator
  reproducibility. In particular, H200 job `332010` remains governed by its
  sealed run contract and enters the new completion-audit sequence only after
  its outputs exist.
