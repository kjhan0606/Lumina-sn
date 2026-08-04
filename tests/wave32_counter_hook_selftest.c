#include "lumina.h"

#include <stdio.h>

void nlte_ew_note_save_restore_call(void);
void nlte_ew_note_per_ion_pin_call(void);
void nlte_ew_note_topstage_IV_call(void);
void nlte_ew_runtime_counts_snapshot(unsigned long out[3]);

int main(void) {
    unsigned long before[3], after[3];
    nlte_ew_runtime_counts_snapshot(before);
    nlte_ew_note_save_restore_call();
    nlte_ew_note_per_ion_pin_call();
    nlte_ew_note_topstage_IV_call();
    nlte_ew_runtime_counts_snapshot(after);
    printf("save_restore=%lu per_ion_pin=%lu topstage_IV=%lu\n",
           after[0] - before[0], after[1] - before[1],
           after[2] - before[2]);
    return 0;
}
