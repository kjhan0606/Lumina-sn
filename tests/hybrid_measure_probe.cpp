#include "cmf_hybrid_measure.hpp"
#include <limits>
extern "C" int hybrid_policy_probe(int total,int code,int damage,int *values){
    if(!values)return 1;
    lumina_linear_compact::Map map={{.5,1},{.5,2},{.5,-30},0.};
    values[0]=lumina_hybrid_measure::primary_budget(total);
    values[1]=lumina_hybrid_measure::secondary_budget(total);
    values[2]=lumina_hybrid_measure::retryable(code);
    values[3]=lumina_hybrid_measure::mark(map,code);
    if(damage==1)map.source.mantissa=.5;
    else if(damage==2)map.tau_error=std::numeric_limits<double>::quiet_NaN();
    else if(damage==3)map.tau_error=-3.5;
    values[4]=lumina_hybrid_measure::pending(map);return 0;
}
