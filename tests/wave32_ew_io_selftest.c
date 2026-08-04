#include <stdio.h>

int nlte_ew_test_dump_io(const char *path);

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int good_rc = nlte_ew_test_dump_io(argv[1]);
    int full_rc = nlte_ew_test_dump_io("/dev/full");
    printf("good_artifact_rc=%d dev_full_write_close_rc=%d\n",
           good_rc, full_rc);
    return good_rc == 0 && full_rc != 0 ? 0 : 1;
}
