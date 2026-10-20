#include "cmf_relativistic_characteristic.h"
#include <math.h>

CMFRelStatus cmf_rel_characteristic_coefficients(
    double radius_cm,double mu,double beta,double dbeta_dr_cm_inverse,
    CMFRelCharacteristic *result)
{
    if(!result||!isfinite(radius_cm)||radius_cm<=0||!isfinite(mu)||fabs(mu)>1||
       !isfinite(beta)||fabs(beta)>=1||!isfinite(dbeta_dr_cm_inverse))return CMF_REL_INVALID_INPUT;
    double gamma=1.0/sqrt((1.0-beta)*(1.0+beta));
    double gamma2=gamma*gamma,angular=1.0-mu*mu,radial=mu+beta;
    CMFRelCharacteristic value;
    value.gamma=gamma;
    value.dr_ds=gamma*radial;
    value.dmu_ds=gamma*angular*((1.0+beta*mu)/radius_cm-
                    gamma2*radial*dbeta_dr_cm_inverse);
    value.b_cm_inverse=gamma*(angular*beta/radius_cm+
                    gamma2*mu*radial*dbeta_dr_cm_inverse);
    if(!isfinite(value.gamma)||!isfinite(value.dr_ds)||!isfinite(value.dmu_ds)||
       !isfinite(value.b_cm_inverse))return CMF_REL_NONFINITE;
    *result=value;return CMF_REL_OK;
}

CMFRelStatus cmf_rel_frequency_raw_terms(
    double chi_cm_inverse,double eta,double nu_dnu,double b_cm_inverse,
    double intensity_blue,CMFRelFrequencyTerms *result)
{
    if(!result||!isfinite(chi_cm_inverse)||!isfinite(eta)||eta<0||
       !isfinite(nu_dnu)||nu_dnu<0||!isfinite(b_cm_inverse)||
       !isfinite(intensity_blue)||intensity_blue<0)return CMF_REL_INVALID_INPUT;
    if(nu_dnu>0&&b_cm_inverse<0)return CMF_REL_NEEDS_RED_UPWIND;
    CMFRelFrequencyTerms value;
    value.blue_coupling_cm_inverse=nu_dnu*b_cm_inverse;
    value.chi_effective_cm_inverse=chi_cm_inverse+(nu_dnu+3.0)*b_cm_inverse;
    value.eta_effective=eta+value.blue_coupling_cm_inverse*intensity_blue;
    if(!isfinite(value.blue_coupling_cm_inverse)||!isfinite(value.chi_effective_cm_inverse)||
       !isfinite(value.eta_effective))return CMF_REL_NONFINITE;
    *result=value;return CMF_REL_OK;
}

CMFRelStatus cmf_rel_homologous_segment(
    double impact_cm,double from_z_cm,double to_z_cm,double epoch_seconds,
    CMFRelHomologousSegment *result)
{
    if(!result||!isfinite(impact_cm)||impact_cm<0||!isfinite(from_z_cm)||
       !isfinite(to_z_cm)||to_z_cm<from_z_cm||!isfinite(epoch_seconds)||epoch_seconds<=0)
        return CMF_REL_INVALID_INPUT;
    double A=2.99792458e10*epoch_seconds;
    double r0=hypot(impact_cm,from_z_cm),r1=hypot(impact_cm,to_z_cm);
    if(!isfinite(A)||!isfinite(r0)||!isfinite(r1)||r0<=0||r1<=0||r0>=A||r1>=A||
       (impact_cm==0&&from_z_cm<0&&to_z_cm>0))return CMF_REL_INVALID_INPUT;
    double b0=r0/A,b1=r1/A,p=impact_cm/A;
    double g0=sqrt((1-b0)*(1+b0)),g1=sqrt((1-b1)*(1+b1));
    double Q=A*sqrt((1-p)*(1+p));
    double x0=from_z_cm/Q,x1=to_z_cm/Q;
    double v0=A*g0/Q,v1=A*g1/Q,dz=to_z_cm-from_z_cm;
    double half_sine=(dz/Q)/sqrt(2*(1+v0*v1-x0*x1));
    CMFRelHomologousSegment value;
    value.comoving_length_cm=A*(2*asin(half_sine))-dz*((from_z_cm+to_z_cm)/A)/(g0+g1);
    value.delta_log_nu=log1p(dz/(A-to_z_cm))-
        .5*log1p((dz/A)*((from_z_cm+to_z_cm)/A)/((1-b1)*(1+b1)));
    value.mu_from=(from_z_cm/r0-b0)/(1-from_z_cm/A);
    value.mu_to=(to_z_cm/r1-b1)/(1-to_z_cm/A);
    if(!isfinite(value.comoving_length_cm)||value.comoving_length_cm<0||
       !isfinite(value.delta_log_nu)||!isfinite(value.mu_from)||!isfinite(value.mu_to))
        return CMF_REL_NONFINITE;
    *result=value;return CMF_REL_OK;
}
