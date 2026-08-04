#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int exercise(int written_shells, int requested_shells,
                    int *nonzero, int *implicit_zero, int *pointer_stalls) {
    char line[8192];
    int used = snprintf(line, sizeof line, "28");
    for (int s = 0; s < written_shells; s++)
        used += snprintf(line + used, sizeof line - (size_t)used, ",%d", s + 1);
    char *p = line;
    (void)strtol(p, &p, 10);
    *nonzero = *implicit_zero = *pointer_stalls = 0;
    for (int s = 0; s < requested_shells; s++) {
        if (*p == ',') p++;
        char *before = p;
        double value = strtod(p, &p); /* exact lumina_atomic.c:835-838 pattern */
        if (value != 0.0) (*nonzero)++;
        else (*implicit_zero)++;
        if (p == before) (*pointer_stalls)++;
    }
    return 0;
}

int main(void) {
    int nonzero, zero, stalls;
    exercise(30, 50, &nonzero, &zero, &stalls);
    printf("SHORT_ROW_CONFIRM nonzero=%d implicit_zero=%d pointer_stalls=%d\n",
           nonzero, zero, stalls);
    if (nonzero != 30 || zero != 20 || stalls != 20) return 1;
    exercise(50, 50, &nonzero, &zero, &stalls);
    printf("FULL_ROW_CONTROL nonzero=%d implicit_zero=%d pointer_stalls=%d\n",
           nonzero, zero, stalls);
    return (nonzero == 50 && zero == 0 && stalls == 0) ? 0 : 1;
}
