/* 비균일 물질 온도를 경계 온도와 분리하고 손상된 입력을 거부한다. */
#define _POSIX_C_SOURCE 200809L
#include "lumina.h"
#include <unistd.h>

static int fixture(const char *dir, const char *text) {
    char path[4096]; snprintf(path,sizeof(path),"%s/electron_temperatures.csv",dir);
    FILE *fp=fopen(path,"w");
    if (!fp) return -1;
    int rc=fputs(text,fp)<0;
    return fclose(fp)||rc ? -1:0;
}

int main(int argc, char **argv) {
    if (argc!=2) return 2;
    double *te=NULL;
    if (load_material_temperature_profile(argv[1],2,9000,&te) ||
        te[0]!=9000 || te[1]!=9000) return 1;
    free(te); te=NULL;
    if (fixture(argv[1],"shell_id,T_e\n0,23021\n1,16497.6392\n") ||
        load_material_temperature_profile(argv[1],2,9000,&te) ||
        te[0]!=23021 || te[1]!=16497.6392) return 1;
    free(te); te=NULL;
    const char *bad[]={
        "shell_id,T_e\n0,23021\n",
        "shell_id,T_e\n0,23021\n0,16497\n",
        "shell_id,T_e\n0,23021\n1,nan\n",
        "shell_id,T_e\n0,23021\n1,-2\n",
        "shell_id,T_e\n0,23021\n1,16497garbage\n",
        "shell_id,T_e\n0,23021\n1,16497\n2,17000\n",
        "shell_id,T_rad\n0,23021\n1,16497\n",
        "shell_id,T_e\n0,23021\n1,1e10000\n"
    };
    for (size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++)
        if (fixture(argv[1],bad[i]) ||
            load_material_temperature_profile(argv[1],2,9000,&te)==0 || te) return 1;
    puts("[MATERIAL-TEMPERATURE][PASS] nonuniform seed preserved; legacy unchanged; 8 corruptions rejected");
    return 0;
}
