#include "cmf_toy_model.h"
int main(int argc,char **argv){
    if(argc!=4)return 2;
    char *args[]={argv[0],"--cmf-toy-initial",argv[1],argv[2],
                  "input/NUC_DECAY_DATA_ni56_co56_fe56",argv[3],"20000","0"};
    return cmf_toy_initial_main(8,args);
}
