아래 수리는 A2-10 계약을 완화하지 않고 발행 위상만 바로잡는다. R8 실패는 committed `(T_e,t)`를 보존한 채 `R7_MATERIAL_UPDATE_BLOCKED`를 출력하고 즉시 상위 driver까지 실패를 전파한다.

## 1. 발주서

### 반복 타임라인

[설계]

```text
현재 immutable material M_m
  → radiation transport
  → radiation_field_commit(r)
  → radiation_field_read_view(r)
  → canonical line_jbar_view(r)
  → a208 opacity publication(o=r)
  → a209 emissivity publication(e=r)
  → A2-10 transaction(t→t+1)
  → compute_plasma_state / NLTE / tau / BF 등 물질 갱신
  → 다음 M_(m+1)
```

성공 조건은 다음 동세대 튜플이다.

```text
radfield.generation
  == line_view.generation
  == cpu_opacity.generation_committed
  == cpu_emissivity.committed_emissivity_generation
```

또한 a208/a209가 기록한 population과 `T_e_generation`은 발행 직전 committed material과 같아야 한다.

### MC lane

[실측] [lumina_main.c:529](/tmp/claude-10396/codex_r7/src/lumina_main.c:529)에서 MC field commit, [lumina_main.c:530](/tmp/claude-10396/codex_r7/src/lumina_main.c:530)에서 continuum view, [lumina_main.c:531](/tmp/claude-10396/codex_r7/src/lumina_main.c:531)에서 line view를 갱신한다.

[실측] 기존에는 그 뒤 `solve_radiation_field` 및 A2-10·population mutation을 거친 후 [lumina_main.c:666](/tmp/claude-10396/codex_r7/src/lumina_main.c:666)의 a208과 [lumina_main.c:678](/tmp/claude-10396/codex_r7/src/lumina_main.c:678)의 a209를 호출했다.

[설계] line view refresh 블록 직후에 공통 `lumina_r7_publish_and_solve_te()`를 호출한다. Gamma deposition처럼 A2-10 입력인 계산은 transport/commit 전에 현 committed material에서 미리 계산한다. `solve_radiation_field`, `compute_plasma_state`, NLTE, BF 갱신은 A2-10 성공 뒤로 남긴다.

반복 0도 `o=e=r=1`, `t:1→2`가 되어야 하므로 R7 phase는 반복 0부터 실행한다. A2-10이 활성화되어 성공한 반복 0은 바로 첫 물질 갱신까지 진행한다.

### pure-CMFGEN lane

[실측] [lumina_cmfgen.c:5162](/tmp/claude-10396/codex_r7/src/lumina_cmfgen.c:5162)의 a208은 formal solve와 field commit보다 앞이다. [lumina_cmfgen.c:5202](/tmp/claude-10396/codex_r7/src/lumina_cmfgen.c:5202)에서 commit/view가 끝나지만 a209 호출은 없다.

[설계]

- commit 전 a208 호출을 제거한다.
- `cmfgen_commit_jnu()` 직후 공통 R7 phase를 호출한다.
- 이 함수가 a208, a209, A2-10을 순서대로 수행한다.
- 기존 호출부의 외부 `T_e_generation++`를 제거한다.

### a209 입력 완비성

[실측] [lumina_plasma.c:8229](/tmp/claude-10396/codex_r7/src/lumina_plasma.c:8229)에 따르면 a209는 다음을 요구한다.

- BF grid와 `bf->eta_bf`
- committed a208 opacity
- canonical radiation-field generation
- canonical line-view generation
- committed population
- 현재 committed `T_e`
- fresh tau와 status-bearing line source

MC lane은 dual-view commit이 continuum과 line view를 함께 발행하므로 입력이 존재한다.

pure lane은 [lumina_cmfgen.c:3399](/tmp/claude-10396/codex_r7/src/lumina_cmfgen.c:3399)의 `cmfgen_commit_jnu()`가 continuum field만 commit한다. `line_n=0`이므로 [radiation_field.c:655](/tmp/claude-10396/codex_r7/src/radiation_field.c:655)에서 line generation이 0으로 남는다.

따라서 현 트리에서는 R7 정위상 호출 뒤 다음이 정상 결과다.

```text
a208: o=1, radiation_generation=1
a209: rc=3, blocked_stale_line 증가
driver: R7_PUBLICATION_BLOCKED 표면 종료
```

R7 자체의 실패가 아니라 R6 deterministic canonical line-J̄ 미착륙 경계다. R6가 `cmfgen_commit_jnu()`의 동일 atomic commit에 q-set/profile이 일치하는 line block을 넣어야 a209와 A2-10까지 진행한다.

### R8 세대 소유권

[실측] [radeq_publication.c:21](/tmp/claude-10396/codex_r7/src/radeq_publication.c:21)의 transaction은 성공 시 `committed_te_generation=gen`과 새 `T_e`를 함께 commit한다. 실패 시 공개 `T_e`를 쓰지 않는다.

[설계]

- `compute_radiative_equilibrium_te()`가 성공 publication의 generation을 `plasma->T_e_generation`에 반영하는 단일 wrapper 소유자가 된다.
- 두 driver의 수동 `T_e_generation++`를 삭제한다.
- 실패 시 호출 전·후 `T_e` manifest와 generation을 대조한다.
- 보존이 확인되어도 계속 진행하지 않는다. `material_update=BLOCKED action=TERMINATE`를 출력하고 종료한다.

## 2. 완전한 패치

아래 unified diff에는 생략 표기가 없으며 그대로 적용할 수 있다.

```diff
diff --git a/src/lumina.h b/src/lumina.h
--- a/src/lumina.h
+++ b/src/lumina.h
@@ -1249,6 +1249,14 @@ int a209_publish_cpu_emissivity(OpacityState *opacity, const BFOpacity *bf,
                                 const AtomicData *atom,
                                 const PlasmaState *plasma,
                                 const NLTEConfig *nlte, double epoch);
+int lumina_r7_publish_and_solve_te(OpacityState *opacity,
+                                   const BFOpacity *bf,
+                                   AtomicData *atom,
+                                   PlasmaState *plasma,
+                                   NLTEConfig *nlte,
+                                   GammaDeposition *gamma_dep,
+                                   double epoch, int n_shells,
+                                   int solve_te,
+                                   const char *lane, int iter);
 /* Wave-1 bf repair gates. All are default OFF and shared with CUDA helpers so
  * host/device producer selection cannot disagree. */
 int lumina_fix_bf_stim_recomb_enabled(void);
diff --git a/src/lumina_plasma.c b/src/lumina_plasma.c
--- a/src/lumina_plasma.c
+++ b/src/lumina_plasma.c
@@ -8261,6 +8261,158 @@ int a209_publish_cpu_emissivity(OpacityState *opacity,const BFOpacity *bf,
  ctr->shells_published+=ns;ctr->cells_published+=n;return 0;
 }
 
+static const char *r7_a210_block_reason(const A210Counters *before,
+                                        const A210Counters *after)
+{
+    if (after->no_bracket > before->no_bracket)
+        return "RADEQ_NO_BRACKET";
+    if (after->no_root > before->no_root)
+        return "RADEQ_NO_ROOT";
+    if (after->nonconverged > before->nonconverged)
+        return "RADEQ_NOT_CONVERGED";
+    if (after->charge_nonconverged > before->charge_nonconverged)
+        return "RADEQ_CHARGE_NOT_CONVERGED";
+    if (after->blocked_stale > before->blocked_stale)
+        return "RADEQ_STALE_INPUT";
+    if (after->blocked_missing_term > before->blocked_missing_term)
+        return "RADEQ_TERM_MISSING";
+    if (after->blocked_schema > before->blocked_schema)
+        return "RADEQ_TERM_SCHEMA";
+    if (after->blocked_sign > before->blocked_sign)
+        return "RADEQ_SIGN_MISMATCH";
+    if (after->te_manifest_mismatch > before->te_manifest_mismatch)
+        return "RADEQ_TE_MANIFEST_MISMATCH";
+    if (after->te_context_mismatch > before->te_context_mismatch)
+        return "RADEQ_TE_CONTEXT_MISMATCH";
+    if (after->nonfinite_failures > before->nonfinite_failures)
+        return "RADEQ_NONFINITE";
+    return "RADEQ_UNQUALIFIED_TE";
+}
+
+int lumina_r7_publish_and_solve_te(OpacityState *opacity,
+                                   const BFOpacity *bf,
+                                   AtomicData *atom,
+                                   PlasmaState *plasma,
+                                   NLTEConfig *nlte,
+                                   GammaDeposition *gamma_dep,
+                                   double epoch, int n_shells,
+                                   int solve_te,
+                                   const char *lane, int iter)
+{
+    if (!lane) lane = "UNKNOWN";
+    if (!opacity || !bf || !atom || !plasma || !nlte ||
+        !plasma->T_e || n_shells <= 0) {
+        fprintf(stderr,
+                "[R7][FATAL] event=R7_INVALID_PHASE_INPUT "
+                "lane=%s iter=%d\n", lane, iter);
+        return 5;
+    }
+
+    uint64_t r = nlte->radfield_view.generation;
+    uint64_t m = atom->population_committed_generation;
+    uint64_t t = plasma->T_e_generation;
+
+    fprintf(stderr,
+            "[R7][PHASE] lane=%s iter=%d phase=view "
+            "rad_status=%d r=%llu line_status=%d line_r=%llu "
+            "population_m=%llu te_t=%llu\n",
+            lane, iter, (int)nlte->radfield_view_status,
+            (unsigned long long)r, (int)nlte->line_view_status,
+            (unsigned long long)nlte->line_view.generation,
+            (unsigned long long)m, (unsigned long long)t);
+
+    if (nlte->radfield_view_status != RADIATION_FIELD_VIEW_OK || r == 0) {
+        a208_counters()->blocked_stale++;
+        fprintf(stderr,
+                "[A2-08][BLOCKED] event=R7_PUBLICATION_BLOCKED "
+                "lane=%s iter=%d reason=STALE_RADIATION_VIEW "
+                "rad_status=%d r=%llu\n",
+                lane, iter, (int)nlte->radfield_view_status,
+                (unsigned long long)r);
+        return 3;
+    }
+
+    int rc = a208_publish_cpu_opacity(opacity, bf, atom, plasma, nlte,
+                                      epoch);
+    if (rc != 0) {
+        A208Counters *c = a208_counters();
+        fprintf(stderr,
+                "[A2-08][BLOCKED] event=R7_PUBLICATION_BLOCKED "
+                "lane=%s iter=%d rc=%d blocked_stale=%llu "
+                "partial_publish=%llu\n",
+                lane, iter, rc,
+                (unsigned long long)c->blocked_stale,
+                (unsigned long long)c->partial_publish_attempts);
+        return rc;
+    }
+
+    const CpuOpacityPublication *op = &opacity->cpu_opacity;
+    if (op->generation_committed != r ||
+        op->radiation_generation != r ||
+        op->population_generation != m ||
+        op->te_generation != t ||
+        op->line_jbar_generation != nlte->line_view.generation) {
+        fprintf(stderr,
+                "[A2-08][FATAL] event=R7_GENERATION_MISMATCH "
+                "lane=%s iter=%d r=%llu o=%llu op_rad=%llu "
+                "op_line=%llu op_pop=%llu op_te=%llu\n",
+                lane, iter, (unsigned long long)r,
+                (unsigned long long)op->generation_committed,
+                (unsigned long long)op->radiation_generation,
+                (unsigned long long)op->line_jbar_generation,
+                (unsigned long long)op->population_generation,
+                (unsigned long long)op->te_generation);
+        return 5;
+    }
+
+    fprintf(stderr,
+            "[R7][PHASE] lane=%s iter=%d phase=a208 r=%llu o=%llu\n",
+            lane, iter, (unsigned long long)r,
+            (unsigned long long)op->generation_committed);
+
+    rc = a209_publish_cpu_emissivity(opacity, bf, atom, plasma, nlte,
+                                     epoch);
+    if (rc != 0) {
+        A209Counters *c = a209_counters();
+        fprintf(stderr,
+                "[A2-09][BLOCKED] event=R7_PUBLICATION_BLOCKED "
+                "lane=%s iter=%d rc=%d r=%llu o=%llu "
+                "blocked_stale_rf=%llu blocked_stale_line=%llu "
+                "blocked_stale_pop=%llu blocked_stale_opacity=%llu\n",
+                lane, iter, rc, (unsigned long long)r,
+                (unsigned long long)op->generation_committed,
+                (unsigned long long)c->blocked_stale_rf,
+                (unsigned long long)c->blocked_stale_line,
+                (unsigned long long)c->blocked_stale_pop,
+                (unsigned long long)c->blocked_stale_opacity);
+        return rc;
+    }
+
+    const CpuEmissivityPublication *em = &opacity->cpu_emissivity;
+    if (nlte->line_view_status != LINE_JBAR_VIEW_OK ||
+        nlte->line_view.generation != r ||
+        em->committed_emissivity_generation != r ||
+        em->opacity_generation != r ||
+        em->radfield_generation != r ||
+        em->line_view_generation != r ||
+        em->population_generation != m ||
+        em->te_generation != t) {
+        fprintf(stderr,
+                "[A2-09][FATAL] event=R7_GENERATION_MISMATCH "
+                "lane=%s iter=%d r=%llu line_status=%d line_r=%llu "
+                "e=%llu em_o=%llu em_rad=%llu em_line=%llu "
+                "em_pop=%llu em_te=%llu\n",
+                lane, iter, (unsigned long long)r,
+                (int)nlte->line_view_status,
+                (unsigned long long)nlte->line_view.generation,
+                (unsigned long long)em->committed_emissivity_generation,
+                (unsigned long long)em->opacity_generation,
+                (unsigned long long)em->radfield_generation,
+                (unsigned long long)em->line_view_generation,
+                (unsigned long long)em->population_generation,
+                (unsigned long long)em->te_generation);
+        return 5;
+    }
+
+    fprintf(stderr,
+            "[R7][PHASE] lane=%s iter=%d phase=a209 "
+            "r=%llu o=%llu e=%llu\n",
+            lane, iter, (unsigned long long)r,
+            (unsigned long long)op->generation_committed,
+            (unsigned long long)em->committed_emissivity_generation);
+
+    if (!solve_te) {
+        fprintf(stderr,
+                "[R7][PHASE] lane=%s iter=%d phase=A2-10 "
+                "action=NOT_REQUESTED te_generation=%llu\n",
+                lane, iter, (unsigned long long)t);
+        return 0;
+    }
+
+    if (t == UINT64_MAX) {
+        fprintf(stderr,
+                "[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED "
+                "lane=%s iter=%d reason=TE_GENERATION_OVERFLOW "
+                "te_generation=%llu material_update=BLOCKED "
+                "action=TERMINATE\n",
+                lane, iter, (unsigned long long)t);
+        return 5;
+    }
+
+    char te_before[65], te_after[65];
+    if (population_te_manifest_sha256(plasma->T_e, (size_t)n_shells,
+                                      te_before) != POP_OK) {
+        fprintf(stderr,
+                "[A2-10][FATAL] lane=%s iter=%d "
+                "reason=TE_MANIFEST_SNAPSHOT_FAILED\n", lane, iter);
+        return 5;
+    }
+
+    A210Counters before = *a210_counters();
+    fprintf(stderr,
+            "[A2-10][PRE] lane=%s iter=%d te_gen=%llu "
+            "rad=%llu line=%llu opacity=%llu emissivity=%llu "
+            "population=%llu\n",
+            lane, iter, (unsigned long long)t,
+            (unsigned long long)r,
+            (unsigned long long)nlte->line_view.generation,
+            (unsigned long long)op->generation_committed,
+            (unsigned long long)em->committed_emissivity_generation,
+            (unsigned long long)m);
+
+    int qualified = compute_radiative_equilibrium_te(
+        plasma, gamma_dep, nlte, atom, opacity, epoch, n_shells);
+    A210Counters after = *a210_counters();
+
+    if (!qualified) {
+        int manifest_ok =
+            population_te_manifest_sha256(plasma->T_e, (size_t)n_shells,
+                                          te_after) == POP_OK;
+        int te_preserved = manifest_ok && strcmp(te_before, te_after) == 0;
+        int generation_preserved = plasma->T_e_generation == t;
+        const char *reason = r7_a210_block_reason(&before, &after);
+
+        fprintf(stderr,
+                "[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED "
+                "lane=%s iter=%d reason=%s "
+                "te_generation_before=%llu te_generation_after=%llu "
+                "te_manifest_preserved=%d generation_preserved=%d "
+                "material_update=BLOCKED action=TERMINATE "
+                "blocked_stale_delta=%llu no_bracket_delta=%llu "
+                "missing_term_delta=%llu schema_delta=%llu\n",
+                lane, iter, reason, (unsigned long long)t,
+                (unsigned long long)plasma->T_e_generation,
+                te_preserved, generation_preserved,
+                (unsigned long long)(after.blocked_stale -
+                                     before.blocked_stale),
+                (unsigned long long)(after.no_bracket -
+                                     before.no_bracket),
+                (unsigned long long)(after.blocked_missing_term -
+                                     before.blocked_missing_term),
+                (unsigned long long)(after.blocked_schema -
+                                     before.blocked_schema));
+
+        if (!te_preserved || !generation_preserved) {
+            fprintf(stderr,
+                    "[A2-10][FATAL] event=R8_PRESERVATION_VIOLATION "
+                    "lane=%s iter=%d\n", lane, iter);
+            return 5;
+        }
+        return 4;
+    }
+
+    if (plasma->T_e_generation != t + 1 ||
+        plasma->te_publication.committed_te_generation !=
+            plasma->T_e_generation) {
+        fprintf(stderr,
+                "[A2-10][FATAL] event=R7_TE_COMMIT_MISMATCH "
+                "lane=%s iter=%d before=%llu plasma=%llu publication=%llu\n",
+                lane, iter, (unsigned long long)t,
+                (unsigned long long)plasma->T_e_generation,
+                (unsigned long long)
+                    plasma->te_publication.committed_te_generation);
+        return 5;
+    }
+
+    fprintf(stderr,
+            "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED "
+            "lane=%s iter=%d phase=A2-10 r=%llu o=%llu e=%llu "
+            "te_generation=%llu->%llu\n",
+            lane, iter, (unsigned long long)r,
+            (unsigned long long)op->generation_committed,
+            (unsigned long long)em->committed_emissivity_generation,
+            (unsigned long long)t,
+            (unsigned long long)plasma->T_e_generation);
+    return 0;
+}
+
 /* Sample Planck frequency using Bjorkman-Wood method (4-random) */
 double sample_planck_frequency(double T, RNG *rng) {
     double kT_h = K_BOLTZMANN * T / H_PLANCK;
@@ -12114,12 +12266,19 @@ int compute_radiative_equilibrium_te(PlasmaState *plasma, GammaDeposition *gamma_dep,
                                      NLTEConfig *nlte, AtomicData *atom,
                                      OpacityState *opacity,
                                      double time_explosion, int n_shells) {
-    /* A2-10 is the only production Te path.  A failed transaction preserves
-     * the previously committed material temperature; there is no scalar-
-     * radiation fallback. */
-    return a210_production_solve(plasma,gamma_dep,nlte,atom,opacity,
-                                 time_explosion,n_shells);
+    if (!plasma || plasma->T_e_generation == UINT64_MAX)
+        return 0;
+
+    uint64_t old_generation = plasma->T_e_generation;
+    int qualified = a210_production_solve(
+        plasma, gamma_dep, nlte, atom, opacity, time_explosion, n_shells);
+
+    if (!qualified)
+        return 0;
+
+    if (plasma->te_publication.committed_te_generation !=
+        old_generation + 1)
+        return 0;
+
+    plasma->T_e_generation =
+        plasma->te_publication.committed_te_generation;
+    return 1;
 }
diff --git a/src/lumina_main.c b/src/lumina_main.c
--- a/src/lumina_main.c
+++ b/src/lumina_main.c
@@ -391,8 +391,24 @@ int main(int argc, char *argv[]) {
     /* ============================================================ */
 
     for (int iter = 0; iter < n_iterations; iter++) { /* Phase 5 - Step 5 */
+        int te_qualified = 0;
+        int material_locked = iter > 0 && nlte_ion_lock_active(iter);
+
         printf("\n--- Iteration %d/%d ---\n", iter + 1, n_iterations); /* Phase 5 - Step 5 */
 
+        /* A2-10 input derived from the currently committed material.  Compute it
+         * before the radiation commit so the post-commit R7 barrier can remain
+         * commit -> view -> a208 -> a209 -> A2-10 with no material mutation. */
+        if (!material_locked && gamma_dep_enabled &&
+            (iter > 0 || radeq_te)) {
+            compute_gamma_deposition(&gamma_dep, &atom_data, &plasma, &geo);
+            printf("  [Gamma] heating_rate[0]=%.2e, [%d]=%.2e erg/s/cm3\n",
+                   gamma_dep.heating_rate[0], geo.n_shells - 1,
+                   gamma_dep.heating_rate[geo.n_shells - 1]);
+        }
+
         /* Phase 5 - Step 5: Reset estimators */
         reset_estimators(est); /* Phase 5 - Step 5 */
         reset_spectrum(spec); /* Phase 5 - Step 5 */
@@ -540,6 +556,19 @@ int main(int argc, char *argv[]) {
                 return EXIT_FAILURE;
             }
         }
+
+        {
+            int r7_rc = lumina_r7_publish_and_solve_te(
+                &opacity, bf_opacity_enabled ? &bf : NULL,
+                &atom_data, &plasma, enable_nlte ? &nlte : NULL,
+                gamma_dep_enabled ? &gamma_dep : NULL,
+                geo.time_explosion, geo.n_shells,
+                radeq_te && !material_locked, "MC", iter);
+            if (r7_rc != 0) {
+                fprintf(stderr,
+                        "[R7][FATAL] lane=MC iter=%d rc=%d\n",
+                        iter, r7_rc);
+                return EXIT_FAILURE;
+            }
+            te_qualified = radeq_te && !material_locked;
+        }
 
         /* Phase 5 - Step 5b: Spectrum binning + L_emitted from actual packets */
         double L_emitted = 0.0;
@@ -586,60 +615,14 @@ int main(int argc, char *argv[]) {
 
         /* Option (8): freeze W/T_rad too once ion-lock activates — true
          * transport-only iteration; plasma state from converged free-NLTE iter. */
-        if (!(iter > 0 && nlte_ion_lock_active(iter))) {
+        if (!material_locked) {
             /* Phase 5 - Step 6: Solve radiation field */
             solve_radiation_field(est, geo.time_explosion, time_simulation, volume,
                                    &opacity, &plasma, config.damping_constant);
         }
 
-        /* Task #072: Recompute tau_sobolev from updated W, T_rad.
-         * Option (8): skip ALL plasma updates once ion-lock activates — freeze
-         * plasma at the converged free-NLTE state, transport packets only. */
-        if (iter > 0 && nlte_ion_lock_active(iter)) {
+        if (material_locked) {
             printf("  [plasma frozen by ion-lock; transport-only iter %d]\n", iter);
-        } else if (iter > 0) {
-            /* Gamma-ray deposition: compute heating/ionization rates */
-            if (gamma_dep_enabled) {
-                compute_gamma_deposition(&gamma_dep, &atom_data, &plasma, &geo);
-                printf("  [Gamma] heating_rate[0]=%.2e, [%d]=%.2e erg/s/cm3\n",
-                       gamma_dep.heating_rate[0], geo.n_shells - 1,
-                       gamma_dep.heating_rate[geo.n_shells - 1]);
-            }
-
-            /* P6: Update per-shell T_e before plasma state.
-             * Both LUMINA_RADEQ_TE and LUMINA_SELF_CONSISTENT_TE now route to the
-             * complete radiative-equilibrium balance (photoionization + Compton +
-             * gamma heating vs. recombination + free-free + collisional bound-bound
-             * + adiabatic cooling). The old Compton-only + f_coll_boost path
-             * (compute_electron_temperature self_consistent branch) is retired: it
-             * omitted photoionization/collisional heating and floor-saturated at
-             * early epochs. No free parameters; under-relaxed via LUMINA_RADEQ_DAMP. */
-            int te_qualified = 0;
-            if (radeq_te) {
-                /* Radiative-equilibrium T_e needs the CURRENT iteration's
-                 * radiation field; normalize J_nu now (re-normalized later in
-                 * the NLTE block, harmlessly — it recomputes from the raw
-                 * estimator). */
-                if (enable_nlte && iter >= nlte_start_iter)
-                    nlte_normalize_j_nu(&nlte, time_simulation, volume, geo.n_shells);
-                te_qualified = compute_radiative_equilibrium_te(&plasma,
-                    gamma_dep_enabled ? &gamma_dep : NULL,
-                    &nlte, &atom_data, &opacity,
-                    geo.time_explosion, geo.n_shells);
-            } else {
-                /* Preserve the committed material temperature.  Radiation has
-                 * no scalar-temperature owner after A2-17. */
-                plasma.T_e_generation = 0;
-            }
-
-            if (te_qualified) {
-                if (plasma.T_e_generation == UINT64_MAX) return EXIT_FAILURE;
-                plasma.T_e_generation++;
-            } else {
-                plasma.T_e_generation = 0;
-            }
+        } else if (iter > 0 || te_qualified) {
             if (compute_plasma_state(&atom_data, &plasma, &opacity,
                                      geo.time_explosion) != 0) {
                 fprintf(stderr, "[A2-07][FATAL] population transaction failed at iter=%d\n",
@@ -663,26 +646,6 @@ int main(int argc, char *argv[]) {
 
             }
 
-            if (a208_publish_cpu_opacity(&opacity,
-                    bf_opacity_enabled ? &bf : NULL,&atom_data,&plasma,
-                    enable_nlte ? &nlte : NULL,
-                    geo.time_explosion) != 0) {
-                fprintf(stderr,"[A2-08][FATAL] signed opacity publication failed iter=%d\n",iter);
-                return EXIT_FAILURE;
-            }
-
-            /* A2-09 publishes only from the checked A2-05..08 generations.
-             * Truth/coverage insufficiency is a surfaced BLOCKED state, not a
-             * license to retain an old eta/CDF or synthesize Planck emission. */
-            {
-                int emiss_rc=a209_publish_cpu_emissivity(&opacity,
-                    bf_opacity_enabled ? &bf : NULL,&atom_data,&plasma,
-                    enable_nlte ? &nlte : NULL,geo.time_explosion);
-                if(emiss_rc!=0)
-                    fprintf(stderr,"[A2-09][BLOCKED] publication rc=%d iter=%d\n",emiss_rc,iter);
-                else if(te_qualified){plasma.te_publication.population_generation=atom_data.population_committed_generation;plasma.te_publication.opacity_generation=opacity.cpu_opacity.generation_committed;plasma.te_publication.emissivity_generation=opacity.cpu_emissivity.committed_emissivity_generation;}
-            }
-
             /* Dynamic transition probability recomputation */
             if (enable_transprob_update && iter >= config.hold_iterations) {
                 compute_transition_probabilities(&atom_data, &plasma, &opacity,
diff --git a/src/lumina_cmfgen.c b/src/lumina_cmfgen.c
--- a/src/lumina_cmfgen.c
+++ b/src/lumina_cmfgen.c
@@ -5156,24 +5156,12 @@ int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
     double t_exp = geo->time_explosion;
     for (int iter = 0; iter < n_iter; ++iter) {
         if (nlte) nlte->current_iter = iter;
+        if (gamma)
+            compute_gamma_deposition(gamma, atom, plasma, geo);
 
         /* refresh bf opacity for current ionization/T_e */
         if (bf) compute_bf_opacity(bf, atom, plasma, cs.n_shells);
-        if (a208_publish_cpu_opacity(opac,bf,atom,plasma,nlte,t_exp)!=0) {
-            /* ★침묵 금지(2026-08-07).  L1-1 로 물질 사슬이 선 뒤 결정론 팔이 여기서
-             * 죽었는데 메시지가 없어 "deterministic path failed" 만 남았다.
-             * 이 지점은 배선도의 R7(발행 위상) 소관이다 — a208/a209 는 field commit
-             * 직후·T_e 호출 전에 발행돼야 A2-10 동세대 삼중항이 성립한다. */
-            A208Counters *c8 = a208_counters();
-            fprintf(stderr,
-                    "[A2-08][FATAL] CPU opacity publication failed at iter=%d "
-                    "(blocked_stale=%llu attempted=%llu committed=%llu)\n",
-                    iter,
-                    (unsigned long long)c8->blocked_stale,
-                    (unsigned long long)c8->replay_line_blocks_attempted,
-                    (unsigned long long)c8->replay_line_blocks_committed);
-            cmfgen_free(&cs);return 5;
-        }
 
         cmfgen_assemble(&cs, geo, opac, bf, plasma);
         if (cmf_lineres) {
@@ -5198,6 +5186,15 @@ int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
                                cs.n_shells, cs.n_bins);
         if (cs.diag && iter == n_iter - 1)
             cmfgen_validate(&cs, geo, plasma);
+
+        radeq_set_line_re_source(cs.chi_line_re, cs.chi_abs, cs.chi_tot,
+                                 cs.S_fixed, cs.J, cs.nu, cs.dnu,
+                                 cs.lambda_star, plasma->T_e,
+                                 cs.chi_line, cs.chi_line_cls,
+                                 cs.n_shells, cs.n_bins);
+
         a208_counters()->replay_line_blocks_attempted++;
         if (cmfgen_commit_jnu(&cs, nlte, geo, (uint64_t)(iter + 1)) != 0) {
             fprintf(stderr, "[RADIATION-FIELD][FATAL] pure-CMFGEN commit failed iter=%d\n",
@@ -5206,6 +5203,18 @@ int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
             return -1;
         }
         a208_counters()->replay_line_blocks_committed++;
+
+        {
+            int r7_rc = lumina_r7_publish_and_solve_te(
+                opac, bf, atom, plasma, nlte, gamma,
+                t_exp, cs.n_shells, 1, "DET", iter);
+            if (r7_rc != 0) {
+                fprintf(stderr,
+                        "[R7][FATAL] lane=DET iter=%d rc=%d\n",
+                        iter, r7_rc);
+                cmfgen_free(&cs);
+                return r7_rc;
+            }
+        }
 
         /* P7 PRODUCER (LUMINA_CMF_LINERES_JBAR=1): fine-grid line-resolved J_bar_l
          * over the UV pump window. Fills opac->jbar_line_det; the plasma bb-rate
@@ -5309,63 +5318,6 @@ int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
                 memcpy(cs.J, Jsave, NS*NB*sizeof(double));     /* restore full J */
             }
         }
-        /* Option-2 integral RE: register the CMFGEN line opacity/source for the
-         * RADEQ/Newton T_e solve (LUMINA_RADEQ_LINE_RE=1). */
-        /* RE/Newton line channel: chi_line_th (physical thermal share) except
-         * in transfer-only eps_uv mode, where the closure keeps FULL chi_line
-         * (cooling-only form in radeq_line_re; codex ruling). */
-        radeq_set_line_re_source(cs.chi_line_re, cs.chi_abs, cs.chi_tot,
-                                 cs.S_fixed, cs.J, cs.nu, cs.dnu,
-                                 cs.lambda_star, plasma->T_e,
-                                 cs.chi_line, cs.chi_line_cls,
-                                 cs.n_shells, cs.n_bins);
-
-        /* ★Codex 가설 확정/기각용 계측(2026-08-07).  A2-10 입구가 요구하는 동세대
-         * 삼중항(opacity·emissivity·radiation)이 이 시점에 무엇인지 그대로 찍는다.
-         * 가설: pure lane 에 a209(emissivity) 발행이 없어 emissivity committed=0 이고
-         * 그 때문에 A2-10 이 blocked_stale 로 자격을 주지 않는다(원인=R7 발행 위상). */
-        {
-            A210Counters *ct10 = a210_counters();
-            fprintf(stderr,
-                "[A2-10][PRE] iter=%d te_gen=%llu | radfield: status=%d gen=%llu | "
-                "line: status=%d gen=%llu | opacity: req=%llu com=%llu rad=%llu pop=%llu | "
-                "emissivity: com=%llu | A2-10 blocked_stale=%llu missing_term=%llu schema=%llu\n",
-                iter, (unsigned long long)plasma->T_e_generation,
-                nlte ? (int)nlte->radfield_view_status : -1,
-                (unsigned long long)(nlte ? nlte->radfield_view.generation : 0),
-                nlte ? (int)nlte->line_view_status : -1,
-                (unsigned long long)(nlte ? nlte->line_view.generation : 0),
-                (unsigned long long)opac->cpu_opacity.generation_required,
-                (unsigned long long)opac->cpu_opacity.generation_committed,
-                (unsigned long long)opac->cpu_opacity.radiation_generation,
-                (unsigned long long)opac->cpu_opacity.population_generation,
-                (unsigned long long)opac->cpu_emissivity.committed_emissivity_generation,
-                (unsigned long long)ct10->blocked_stale,
-                (unsigned long long)ct10->blocked_missing_term,
-                (unsigned long long)ct10->blocked_schema);
-        }
-
-        /* downstream solvers reused unchanged */
-        int te_qualified = compute_radiative_equilibrium_te(
-            plasma, gamma, nlte, atom, opac, t_exp, cs.n_shells);
-        if (!te_qualified) plasma->T_e_generation = 0;
-        if (!te_qualified) {
-            fprintf(stderr, "[CMFGEN][FATAL] radiative-equilibrium T_e not qualified "
-                            "iter=%d (te_generation=%llu)\n",
-                    iter, (unsigned long long)plasma->T_e_generation);
-            return -1;
-        }
-        if (plasma->T_e_generation == UINT64_MAX) {
-            fprintf(stderr, "[CMFGEN][FATAL] T_e generation overflow\n");
-            return -1;
-        }
-        plasma->T_e_generation++;
         if (compute_plasma_state(atom, plasma, opac, t_exp) != 0) {
             fprintf(stderr, "[A2-07][FATAL] CMF population transaction failed iter=%d\n",
                     iter);
```

## 3. 검증 사전등록

### 기대 변경집합

#### R7 + R6가 모두 준비된 성공 경로

MC와 DET 양쪽에서 반복마다 다음 순서가 보여야 한다.

```text
[R7][PHASE] lane=DET iter=0 phase=view ... r=1 line_r=1
[R7][PHASE] lane=DET iter=0 phase=a208 r=1 o=1
[R7][PHASE] lane=DET iter=0 phase=a209 r=1 o=1 e=1
[A2-10][PRE] lane=DET iter=0 te_gen=1 rad=1 line=1 opacity=1 emissivity=1 population=1
[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED ... te_generation=1->2
```

MC lane도 `lane=MC`로 동일하다. 기존 다음 증상은 사라져야 한다.

```text
opacity: com=1 rad=0
emissivity: com=0
```

성공 시 기대값은 다음과 같다.

```text
opacity: com=1 rad=1
emissivity: com=1
T_e_generation: 1 -> 2
```

#### 현 R6 미착륙 스냅샷의 기대 결과

pure lane은 A2-10까지 통과하지 않는 것이 정상이다.

```text
[R7][PHASE] lane=DET iter=0 phase=view rad_status=0 r=1 line_status=-1 line_r=0
[R7][PHASE] lane=DET iter=0 phase=a208 r=1 o=1
[A2-09][BLOCKED] event=R7_PUBLICATION_BLOCKED ... rc=3 ... blocked_stale_line=1
[R7][FATAL] lane=DET iter=0 rc=3
[CMFGEN][FATAL] deterministic path failed
```

이는 R7 실패가 아니라 “R7이 a209의 실제 선행조건을 처음 올바른 위치에서 드러낸 결과”다. R6 성공 기준은 canonical line view가 `line_status=LINE_JBAR_VIEW_OK`, `line_r=r`로 바뀌는 것이다.

### 음성 대조

1. no-bracket 주입 — 조건 5/R8

테스트 빌드에서 모든 셸의 A2-10 하·상단 residual 부호를 같게 만든다. 출하 노브나 덱 변경으로 만들지 않는다.

기대:

```text
[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED
reason=RADEQ_NO_BRACKET
te_generation_before=t
te_generation_after=t
te_manifest_preserved=1
generation_preserved=1
material_update=BLOCKED
action=TERMINATE
no_bracket_delta=1
```

그 뒤 A2-07, NLTE, BF, tau 물질 갱신 로그가 한 줄이라도 나오면 FAIL이다. `T_e_generation=0` 또는 다음 반복 진입도 FAIL이다.

2. a209를 A2-10 뒤로 되돌리는 위상 회귀

R6가 준비된 테스트 빌드에서 helper 안의 a209 호출만 A2-10 뒤로 옮긴다.

기대:

```text
[A2-10][BLOCKED] ... reason=RADEQ_STALE_INPUT
blocked_stale_delta=1
te_manifest_preserved=1
generation_preserved=1
material_update=BLOCKED action=TERMINATE
```

첫 반복이면 emissivity 미발행, 이후 반복이면 `em.radfield_generation != current r` 때문에 차단되어야 한다. A2-10 성공은 계약 회귀다.

3. pure canonical line commit 생략

현 기준선 그대로 `cmfgen_commit_jnu()`에 line block을 싣지 않는다.

기대:

```text
[A2-09][BLOCKED] ... blocked_stale_line=1
[R7][FATAL] lane=DET ...
```

a209가 e를 commit하거나 A2-10으로 진행하면 FAIL이다.

4. radiation generation off-by-one

테스트 빌드에서 a208 직전 active view generation 또는 opacity required generation을 한 세대 어긋나게 한다.

기대:

```text
[A2-08][FATAL] event=R7_GENERATION_MISMATCH
r=N o=N-1
```

A2-10 진입은 없어야 한다.

5. a209 opacity/population stamp 불일치

a208 뒤 테스트 하니스에서 `cpu_opacity.population_generation`을 1 증가시키거나, a209 후보의 population stamp를 현재 `m`과 다르게 한다.

기대:

```text
[A2-09][FATAL] event=R7_GENERATION_MISMATCH
```

오래된 emissivity/CDF 유지 후 속행하면 FAIL이다.

### 단 경계

- 이 단의 성공: commit/view 뒤 a208·a209 호출 위치, pure a209 신설, `o=e=r` 검사, A2-10의 `t→t+1` 단일 소유, 실패 시 `(T_e,t)` 보존과 표면 종료.
- R6 소관: DET canonical line-J̄ 생산, q-set/profile 결박, `line_status=OK`, `line_generation=r`.
- 따라서 현 `line: status=-1 gen=0`에서 R7 단독 실행이 A2-09에서 멈추는 것은 사전등록된 기대 결과다.

### Fable 조건 1 사전등록

R7은 radiation owner의 초기 generation이나 R5 barrier를 변경하지 않는다. 다만 `o=e=현재 state-owner view의 r`을 새로 강제하므로 R5 전에 다음 초기화 규칙을 등록한다.

[설계]

```text
bootstrap:
    DET owner commits r=1
    MC owner has no consumable r=1 payload

first two-arm epoch:
    MC owner generation baseline := DET committed generation(1)
    baseline은 payload commit이 아니며 view/comparison 소비 금지
    DET requested = DET computed + 1 = 2
    MC  requested = MC  baseline + 1 = 2
    양쪽 r=2 실제 commit이 끝난 뒤에만 barrier 통과
    그때 baseline-only 표지를 해제

later epochs:
    each requested = its own computed + 1
    barrier requires MC.r == DET.r
```

MC r=1을 가짜 field commit으로 꾸미거나 DET payload를 MC 측정치처럼 복사하면 안 된다.

음성 대조는 MC baseline을 0 또는 2로 고의 설정해 첫 barrier에서 다음을 요구한다.

```text
TWOARM_GENERATION_MISMATCH
MC requested != DET requested
material update not entered
```

현재 작업공간에는 Makefile/CMake 및 전체 include 집합이 없어 여기서는 빌드 실행 대신 호출·타입·generation 경로에 대한 정적 검토만 수행했다.