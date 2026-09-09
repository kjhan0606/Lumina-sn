#ifndef LUMINA_CMFGEN_FINAL_WRITER_H
#define LUMINA_CMFGEN_FINAL_WRITER_H

#include "lumina.h"

int cmfgen_final_writer(const char *directory, const PlasmaState *plasma,
                        const AtomicData *atom, const NLTEConfig *nlte,
                        int n_shells);

#endif
