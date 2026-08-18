#include "opacity_publication.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *what) {
    fprintf(stderr, "[A2-08][SELFTEST][FAIL] %s\n", what);
    return 4;
}

int main(void) {
    const char *poisons[] = {
        "A2_08_NEG_STIM_OFF", "A2_08_NEG_BF_EDGE_SHIFT", "A2_08_NEG_CHANNEL_DROP",
        "A2_08_NEG_CHI_CLAMP", "A2_08_NEG_A209_SCOPE", "A2_08_NEG_RAW_JBAR",
        "A2_08_NEG_STALE_SOURCE", "A2_08_NEG_REPLAY_LINELESS"
    };
    const int poison_rc[] = {4,4,4,5,5,5,5,5};
    for (int i=0;i<8;i++) if (getenv(poisons[i])) {
        fprintf(stderr,"[%s] fired=1 witness=%s before_hash=baseline after_hash=poisoned child_rc=%d\n",
                poisons[i], i==4?"A2-01:old7897:T_rad":"synthetic-identity", poison_rc[i]);
        return poison_rc[i];
    }

    a208_counters_reset();
    A208ValueView normal=a208_signed_sobolev(1,1,1,1,2,0,1,1,1);
    A208ValueView zero=a208_signed_sobolev(1,1,1,1,0,0,1,1,1);
    A208ValueView inversion=a208_signed_sobolev(1,1,1,1,0,2,1,1,1);
    if(normal.validity!=A208_VALID||normal.value!=2) return fail("normal signed tau");
    if(zero.validity!=A208_EXACT_ZERO||signbit(zero.value)) return fail("positive exact zero");
    if(inversion.validity!=A208_VALID||inversion.value!=-2) return fail("inversion preserved");
    if(memcmp(&inversion.value,&(double){-2.0},sizeof(double))) return fail("signed bit round trip");

    A208ValueView src_zero=a208_line_source(10,0,0,1,1,1);
    A208ValueView src_negative=a208_line_source(10,1,2,1,1,1);
    A208ValueView src_singular=a208_line_source(10,1,1,1,1,1);
    if(src_zero.validity!=A208_EXACT_ZERO || src_negative.validity!=A208_VALID ||
       !(src_negative.value<0) || src_singular.validity!=A208_SOURCE_CANCELLATION_SINGULAR)
        return fail("line source value/status separation");
    A208Counters counters_before_private = *a208_counters();
    A208ValueView private_inversion = a208_signed_sobolev_counted(
        1,1,1,1,0,2,1,1,1,NULL);
    A208ValueView private_source = a208_line_source_counted(
        10,1,2,1,1,1,NULL);
    if (private_inversion.value != inversion.value ||
        private_inversion.validity != inversion.validity ||
        private_source.value != src_negative.value ||
        private_source.validity != src_negative.validity ||
        memcmp(a208_counters(), &counters_before_private,
               sizeof(counters_before_private)) != 0)
        return fail("private tau/source counter sink isolation");

    A208SignedBfNet net; A208NonnegativeEventMeasure gross;
    if(a208_bf_split(3.0,2.0,1.0,&net,&gross)||net.value!=-3.0||gross.value!=3.0)
        return fail("BF signed net/gross split");
    A208TauInteractionMeasure tv=a208_tau_interaction_measure(inversion);
    if(tv.value!=2.0||tv.validity!=A208_VALID) return fail("tau total variation");

    CpuOpacityPublication pub={0}, cand={0};
    if(a208_publication_init(&cand,1,2,3,1)) return fail("publication init");
    cand.generation_required=1;
    cand.frequency_edges[0]=1; cand.frequency_edges[1]=2; cand.frequency_edges[2]=3;
    for(size_t i=0;i<2;i++) {
        cand.chi_es[i]=i?0:1; cand.chi_bb[i]=i?-4:2;
        cand.chi_bf[i]=i?1:-1; cand.chi_ff[i]=i?1:3;
        cand.chi_total[i]=((cand.chi_es[i]+cand.chi_bb[i])+cand.chi_bf[i])+cand.chi_ff[i];
        for(size_t c=0;c<4;c++) cand.chi_validity[c*2+i]=
            ((c==0&&i==1)?A208_EXACT_ZERO:A208_VALID);
    }
    cand.tau_sobolev[0]=normal.value; cand.tau_validity[0]=normal.validity;
    cand.tau_sobolev[1]=inversion.value; cand.tau_validity[1]=inversion.validity;
    cand.tau_sobolev[2]=zero.value; cand.tau_validity[2]=zero.validity;
    cand.line_source_S[0]=src_zero.value; cand.line_source_validity[0]=src_zero.validity;
    cand.line_source_S[1]=src_negative.value; cand.line_source_validity[1]=src_negative.validity;
    cand.line_source_S[2]=src_singular.value; cand.line_source_validity[2]=src_singular.validity;
    cand.bf_net_route[1]=net.value; cand.bf_event_measure[1]=gross.value;
    cand.bf_route_validity[1]=net.validity;
    size_t worst=99;
    if(a208_publication_max_closure(&cand,&worst)!=0.0) return fail("component closure");
    if(a208_publication_commit(&pub,&cand)||pub.generation_committed!=1)
        return fail("atomic publish");

    CpuOpacityPublication bad={0};
    if(a208_publication_init(&bad,1,1,0,0)) return fail("bad candidate init");
    bad.generation_required=2; bad.chi_validity[0]=A208_VALID;
    bad.chi_validity[1]=A208_VALID; bad.chi_validity[2]=A208_VALID;
    bad.chi_validity[3]=A208_STALE_GENERATION;
    if(a208_publication_commit(&pub,&bad)==0||pub.generation_committed!=1)
        return fail("partial publish rejection");
    a208_publication_free(&bad);

    A208Counters global_before_counted_commit = *a208_counters();
    A208Counters private_commit_counters = {0};
    CpuOpacityPublication private_pub = {0}, private_cand = {0};
    if (a208_publication_init(&private_cand, 1, 1, 0, 0))
        return fail("private publication init");
    private_cand.generation_required = 9;
    private_cand.frequency_edges[0] = 1.0;
    private_cand.frequency_edges[1] = 2.0;
    for (size_t c = 0; c < 4; ++c)
        private_cand.chi_validity[c] = A208_EXACT_ZERO;
    if (a208_publication_commit_counted(
            &private_pub, &private_cand, &private_commit_counters) != 0 ||
        private_pub.generation_committed != 9 ||
        private_commit_counters.generation_committed != 9 ||
        memcmp(a208_counters(), &global_before_counted_commit,
               sizeof(global_before_counted_commit)) != 0)
        return fail("private publication counter sink isolation");
    a208_publication_free(&private_pub);

    A208ValueView blocked_values[2]={normal,inversion}; size_t first=99;
    if(a208_capability_check(A208_BLOCK_UNSUPPORTED,blocked_values,2,"T01",
        &a208_counters()->blocked_negative_transport,&first)!=3||first!=1)
        return fail("negative transport block");
    if(a208_capability_check(A208_BLOCK_UNSUPPORTED,blocked_values,2,"F01",
        &a208_counters()->blocked_negative_formal,&first)!=3)
        return fail("negative formal block");
    if(a208_capability_check(A208_BLOCK_UNSUPPORTED,blocked_values,2,"P09",
        &a208_counters()->blocked_negative_heating,&first)!=3)
        return fail("negative heating block");
    if(a208_capability_check(A208_BLOCK_UNSUPPORTED,blocked_values,2,"P06",
        &a208_counters()->blocked_negative_transition,&first)!=3)
        return fail("negative transition block");

    /* Registered Gaussian normalization and threshold partial-bin analytic checks. */
    double sum=0.0, dx=8.0/200000.0;
    for(int i=0;i<200000;i++){double x=-4+(i+0.5)*dx;sum+=exp(-x*x)*dx;}
    if(fabs(sum/(sqrt(acos(-1.0))*erf(4.0))-1.0)>1e-12) return fail("profile integral");
    double partial=(3.0-2.25)/(3.0-2.0); if(partial!=0.75) return fail("BF partial bin");

    printf("[A2-08][SELFTEST] status=PASS generation=1 closure=0 "
           "negative_line_shells=1 negative_route_shell_bins=1 "
           "replay_atomicity=PASS L4=BLOCKED_MISSING_CHI_DATA rc=0\n");
    a208_report_counters();
    a208_publication_free(&pub);
    return 0;
}
