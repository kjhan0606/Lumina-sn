#include "cmf_toy_model.h"
int main(int argc,char **argv){
    if(argc!=4&&argc!=5)return 2;
    char *args[]={argv[0],"--cmf-toy-initial",argv[1],argv[2],
                  "input/NUC_DECAY_DATA_ni56_co56_fe56",argv[3],"20000","0",
                  "input/arnaud_rothenflug_23jun15",argc==5?argv[4]:NULL};
    return cmf_toy_initial_main(argc==5?10:8,args);
}
