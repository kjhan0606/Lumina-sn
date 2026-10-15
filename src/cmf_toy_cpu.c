/* GPU 없이 기존 toy06 물리 소유자를 실행하는 전용 CLI. 별도 물리식은 없다. */
#include "cmf_toy_model.h"
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv){
    if(argc>1&&!strcmp(argv[1],"--cmf-toy-bootstrap"))return cmf_toy_bootstrap_main(argc,argv);
    if(argc>1&&!strcmp(argv[1],"--cmf-toy-initial"))return cmf_toy_initial_main(argc,argv);
    if(argc>1&&!strcmp(argv[1],"--cmf-toy-step"))return cmf_toy_step_main(argc,argv);
    fprintf(stderr,"usage: %s --cmf-toy-{bootstrap,initial,step} ...\n",argv[0]);return 2;
}
