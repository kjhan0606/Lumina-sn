#define _GNU_SOURCE
#include "../src/lumina_cmfgen.c"

#include <float.h>

typedef struct {
    Geometry geo;
    double *ri, *ro, *vi, *vo;
} TestGeometry;

static int failures;
#define REQUIRE(x, msg) do { if (!(x)) { \
    fprintf(stderr,"[SPHERICAL-TEST][FAIL] %s\n",msg); failures++; } \
} while (0)

static int geometry_from_edges(TestGeometry *g,const double *edge,int ns)
{
    memset(g,0,sizeof(*g));
    g->ri=(double*)calloc((size_t)ns,sizeof(double));
    g->ro=(double*)calloc((size_t)ns,sizeof(double));
    g->vi=(double*)calloc((size_t)ns,sizeof(double));
    g->vo=(double*)calloc((size_t)ns,sizeof(double));
    if(!g->ri||!g->ro||!g->vi||!g->vo)return -1;
    g->geo.n_shells=ns;g->geo.r_inner=g->ri;g->geo.r_outer=g->ro;
    g->geo.v_inner=g->vi;g->geo.v_outer=g->vo;
    for(int s=0;s<ns;s++){g->ri[s]=edge[s];g->ro[s]=edge[s+1];}
    return 0;
}

static int geometry_from_shell_csv(TestGeometry *g,const char *path)
{
    FILE *fp=fopen(path,"r");char line[8192];double edge[51];
    if(!fp||!fgets(line,sizeof(line),fp))return -1;
    for(int s=0;s<50;s++){
        int id;double ri,ro,vi,vo,te,ne;
        if(!fgets(line,sizeof(line),fp)||
           sscanf(line,"%d,%lf,%lf,%lf,%lf,%lf,%lf",&id,&ri,&ro,&vi,&vo,&te,&ne)!=7||id!=s){
            fclose(fp);return -1;
        }
        if(s==0)edge[0]=ri;else if(ri!=edge[s]){fclose(fp);return -1;}
        edge[s+1]=ro;
    }
    if(fclose(fp)!=0)return -1;
    return geometry_from_edges(g,edge,50);
}

static void geometry_free(TestGeometry *g)
{
    free(g->ri);free(g->ro);free(g->vi);free(g->vo);memset(g,0,sizeof(*g));
}

static int ref_gauss(int n,long double *x,long double *w)
{
    const long double pi=3.141592653589793238462643383279502884L;
    for(int i=0;i<(n+1)/2;i++){
        long double z=cosl(pi*((long double)i+0.75L)/((long double)n+0.5L));int ok=0;
        for(int it=0;it<80;it++){
            long double p1=1,p2=0;
            for(int j=1;j<=n;j++){long double p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}
            long double pp=n*(z*p1-p2)/(z*z-1),next=z-p1/pp;
            if(fabsl(next-z)<=8*LDBL_EPSILON){z=next;ok=1;break;}z=next;
        }
        if(!ok||z<0||z>=1)return -1;
        long double p1=1,p2=0;
        for(int j=1;j<=n;j++){long double p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}
        long double pp=n*(z*p1-p2)/(z*z-1),wi=2/((1-z*z)*pp*pp);
        x[i]=-z;x[n-1-i]=z;w[i]=wi;w[n-1-i]=wi;
    }
    return 0;
}

static long double chord_I(const Geometry *g,int obs,long double mu,
                           long double chi,long double source,long double Bin)
{
    long double rin=g->r_inner[0],rout=g->r_outer[g->n_shells-1];
    long double r=0.5L*((long double)g->r_inner[obs]+g->r_outer[obs]);
    long double p=r*sqrtl(fmaxl(0,(1-mu)*(1+mu))),z=r*mu;
    int core=mu>0&&p<rin;
    long double z0=core?sqrtl(fmaxl(0,(rin-p)*(rin+p))):
                         -sqrtl(fmaxl(0,(rout-p)*(rout+p)));
    long double ex=chi==0?1:expl(-chi*(z-z0));
    return source+((core?Bin:0)-source)*ex;
}

static long double chord_J(const Geometry *g,int obs,long double chi,
                           long double source,long double Bin,int nq,
                           const long double *x,const long double *w)
{
    long double rin=g->r_inner[0];
    long double r=0.5L*((long double)g->r_inner[obs]+g->r_outer[obs]);
    long double limb=sqrtl(fmaxl(0,(1-rin/r)*(1+rin/r)));
    long double edge[4]={-1,0,limb,1},sum=0;
    for(int k=0;k<3;k++){
        long double mid=0.5L*(edge[k]+edge[k+1]),half=0.5L*(edge[k+1]-edge[k]);
        for(int q=0;q<nq;q++)sum+=0.5L*half*w[q]*chord_I(g,obs,mid+half*x[q],chi,source,Bin);
    }
    return sum;
}

static double rel(double a,double b){return fabs(a-b)/fmax(fabs(b),DBL_MIN);}
static double dilution(double rin,double r){double x=rin/r;return 0.5*x*x/(1+sqrt((1-x)*(1+x)));}

static int forward_march(const CMFGENState *cs,const CMFSphericalGeometry *g,
                         int b,const double *S,double Bin,double *J)
{
    int ns=cs->n_shells,nb=cs->n_bins;memset(J,0,(size_t)ns*sizeof(double));
    for(size_t ir=0;ir<g->n_rays;ir++){
        const CMFSphericalRay *ray=&g->rays[ir];double I=ray->core_boundary?Bin:0;
        for(int k=0;k<ray->n_segments;k++){
            const CMFSphericalSegment *sg=&g->segments[ray->first_segment+(size_t)k];
            double tau=cs->chi_tot[(size_t)sg->shell*nb+b]*sg->ds;
            double ex=tau>745?0:exp(-tau),psi=tau>745?1:-expm1(-tau);
            I=I*ex+S[sg->shell]*psi;
        }
        J[ray->observer_shell]+=ray->weight*I;
    }
    return 0;
}

static int solve_dense(double *a,double *b,int n)
{
    for(int k=0;k<n;k++){
        int p=k;for(int i=k+1;i<n;i++)if(fabs(a[(size_t)i*n+k])>fabs(a[(size_t)p*n+k]))p=i;
        if(!(fabs(a[(size_t)p*n+k])>DBL_MIN))return -1;
        if(p!=k){for(int j=0;j<n;j++){double t=a[(size_t)k*n+j];a[(size_t)k*n+j]=a[(size_t)p*n+j];a[(size_t)p*n+j]=t;}double t=b[k];b[k]=b[p];b[p]=t;}
        for(int i=k+1;i<n;i++){double m=a[(size_t)i*n+k]/a[(size_t)k*n+k];
            for(int j=k+1;j<n;j++)a[(size_t)i*n+j]-=m*a[(size_t)k*n+j];b[i]-=m*b[k];}
    }
    for(int i=n-1;i>=0;i--){for(int j=i+1;j<n;j++)b[i]-=a[(size_t)i*n+j]*b[j];b[i]/=a[(size_t)i*n+i];}
    return 0;
}

static int vacuum_test(const char *name,const Geometry *g,FILE *fp,double *maxerr)
{
    CMFGENState cs;if(cmfgen_init(&cs,g)!=0)return -1;int ns=g->n_shells;
    double *S=(double*)calloc((size_t)ns,sizeof(double)),*J=(double*)malloc((size_t)ns*sizeof(double));
    if(!S||!J||cmfgen_formal_bin_order(&cs,g,0,S,1,CMFGEN_SPHERICAL_GAUSS_ORDER,J,NULL,NULL,NULL)!=0)return -1;
    for(int s=0;s<ns;s++){double r=0.5*(g->r_inner[s]+g->r_outer[s]),W=dilution(g->r_inner[0],r),e=rel(J[s],W);
        if(e>*maxerr)*maxerr=e;fprintf(fp,"%s,%d,%.17g,%.17g,%.17g,%.17g\n",name,s,r,J[s],W,e);}
    free(S);free(J);cmfgen_free(&cs);return 0;
}

static int absorption_test(const Geometry *g,FILE *fp,double *maxerr,double *maxref)
{
    const double taus[]={0,0.1,1,10,100};const int orders[]={4,8,16,32};int ns=g->n_shells;
    long double *x256=malloc(256*sizeof(*x256)),*w256=malloc(256*sizeof(*w256));
    long double *x512=malloc(512*sizeof(*x512)),*w512=malloc(512*sizeof(*w512));
    if(!x256||!w256||!x512||!w512||ref_gauss(256,x256,w256)||ref_gauss(512,x512,w512))return -1;
    CMFGENState cs;if(cmfgen_init(&cs,g)!=0)return -1;
    double *S=malloc((size_t)ns*sizeof(double)),*J=malloc((size_t)ns*sizeof(double));
    double span=g->r_outer[ns-1]-g->r_inner[0];for(int s=0;s<ns;s++)S[s]=0.3;
    for(size_t t=0;t<sizeof(taus)/sizeof(*taus);t++){
        double chi=taus[t]/span;for(int s=0;s<ns;s++)cs.chi_tot[(size_t)s*cs.n_bins]=chi;
        for(int s=0;s<ns;s++){double a=(double)chord_J(g,s,chi,0.3L,1,256,x256,w256),b=(double)chord_J(g,s,chi,0.3L,1,512,x512,w512);double e=rel(a,b);if(e>*maxref)*maxref=e;}
        for(size_t o=0;o<sizeof(orders)/sizeof(*orders);o++){
            if(cmfgen_formal_bin_order(&cs,g,0,S,1,orders[o],J,NULL,NULL,NULL))return -1;
            for(int s=0;s<ns;s++){double ref=(double)chord_J(g,s,chi,0.3L,1,512,x512,w512),e=rel(J[s],ref);
                if(orders[o]==CMFGEN_SPHERICAL_GAUSS_ORDER&&e>*maxerr)*maxerr=e;
                fprintf(fp,"%d,%.17g,%d,%.17g,%.17g,%.17g\n",s,taus[t],orders[o],J[s],ref,e);}
        }
    }
    free(S);free(J);cmfgen_free(&cs);free(x256);free(w256);free(x512);free(w512);return 0;
}

static int response_thick_test(const Geometry *g,FILE *fp,double *maxresp,double *maxlambda)
{
    CMFGENState cs;if(cmfgen_init(&cs,g))return -1;int ns=g->n_shells;double span=g->r_outer[ns-1]-g->r_inner[0];
    for(int s=0;s<ns;s++)cs.chi_tot[(size_t)s*cs.n_bins]=(0.2+0.7*s)/span;
    CMFSphericalGeometry sg={0};double *C=malloc((size_t)ns*sizeof(double)),*L=malloc((size_t)ns*ns*sizeof(double));
    double *S=calloc((size_t)ns,sizeof(double)),*J=malloc((size_t)ns*sizeof(double));
    if(!C||!L||!S||!J||spherical_geometry_build(&cs,g,CMFGEN_SPHERICAL_GAUSS_ORDER,&sg)||spherical_build_operator(&cs,&sg,0,0,C,L))return -1;
    for(int col=0;col<ns;col++){memset(S,0,(size_t)ns*sizeof(double));S[col]=1;forward_march(&cs,&sg,0,S,0,J);
        for(int row=0;row<ns;row++){double e=fabs(J[row]-L[(size_t)row*ns+col]);if(e>*maxresp)*maxresp=e;
            fprintf(fp,"unit,%d,%d,%.17g,%.17g,%.17g\n",row,col,J[row],L[(size_t)row*ns+col],e);}}
    for(int s=0;s<ns;s++)cs.chi_tot[(size_t)s*cs.n_bins]=1e6/span;
    if(spherical_build_operator(&cs,&sg,0,0,C,L))return -1;
    for(int row=0;row<ns;row++)for(int col=0;col<ns;col++){
        double v=L[(size_t)row*ns+col],violation=fmax(0,v-1);if(violation>*maxlambda)*maxlambda=violation;
        fprintf(fp,"thick,%d,%d,%.17g,%.17g,%.17g\n",row,col,v,v,violation);
        REQUIRE(v>=0&&v<=1,"thick Lambda outside [0,1]");
    }
    spherical_geometry_free(&sg);free(C);free(L);free(S);free(J);cmfgen_free(&cs);return 0;
}

static int ali_test(const Geometry *g,FILE *fp,double *maxdiag,double *maxtri)
{
    const double albedo[]={0,0.5,0.99};int ns=g->n_shells;double span=g->r_outer[ns-1]-g->r_inner[0];
    for(size_t ia=0;ia<sizeof(albedo)/sizeof(*albedo);ia++){
        CMFGENState cs;if(cmfgen_init(&cs,g))return -1;double Bin=cm_planck(cs.nu[0],10000);
        for(int s=0;s<ns;s++)for(int b=0;b<cs.n_bins;b++){size_t q=(size_t)s*cs.n_bins+b;
            cs.chi_tot[q]=1/span;cs.chi_es[q]=albedo[ia]/span;cs.S_fixed[q]=(1-albedo[ia])*0.3*cm_planck(cs.nu[b],10000);}
        CMFSphericalGeometry sg={0};double C[16],L[256],A[256],exact[16];
        if(ns>16||spherical_geometry_build(&cs,g,CMFGEN_SPHERICAL_GAUSS_ORDER,&sg)||spherical_build_operator(&cs,&sg,0,Bin,C,L))return -1;
        for(int i=0;i<ns;i++){exact[i]=C[i];for(int j=0;j<ns;j++){exact[i]+=L[(size_t)i*ns+j]*cs.S_fixed[(size_t)j*cs.n_bins];A[(size_t)i*ns+j]=(i==j?1:0)-L[(size_t)i*ns+j]*albedo[ia];}}
        if(solve_dense(A,exact,ns))return -1;spherical_geometry_free(&sg);
        for(int mode=0;mode<2;mode++){
            memset(cs.J,0,(size_t)ns*cs.n_bins*sizeof(double));
            if(mode){setenv("LUMINA_CMFGEN_LAMBDA_TRI","1",1);setenv("LUMINA_CMFGEN_ALI_TOL","1e-14",1);setenv("LUMINA_CMFGEN_ALI_MINIT","1",1);}else{unsetenv("LUMINA_CMFGEN_LAMBDA_TRI");unsetenv("LUMINA_CMFGEN_ALI_TOL");unsetenv("LUMINA_CMFGEN_ALI_MINIT");}
            if(cmfgen_solve_J(&cs,g,10000,1000))return -1;double num=0,den=0,resid=0;
            for(int i=0;i<ns;i++){double v=cs.J[(size_t)i*cs.n_bins];num+=fabs(v-exact[i]);den+=fabs(exact[i]);double r=v-C[i];
                for(int j=0;j<ns;j++)r-=L[(size_t)i*ns+j]*(cs.S_fixed[(size_t)j*cs.n_bins]+albedo[ia]*cs.J[(size_t)j*cs.n_bins]);resid=fmax(resid,fabs(r)/fmax(fabs(v),DBL_MIN));}
            double e=num/den;if(mode){if(e>*maxtri)*maxtri=e;}else if(e>*maxdiag)*maxdiag=e;
            fprintf(fp,"%.17g,%s,%d,%.17g,%.17g\n",albedo[ia],mode?"tri":"diag",1000,e,resid);
        }
        cmfgen_free(&cs);
    }
    unsetenv("LUMINA_CMFGEN_LAMBDA_TRI");unsetenv("LUMINA_CMFGEN_ALI_TOL");unsetenv("LUMINA_CMFGEN_ALI_MINIT");return 0;
}

static int center_index(const Geometry *g,double r){for(int s=0;s<g->n_shells;s++)if(0.5*(g->r_inner[s]+g->r_outer[s])==r)return s;return -1;}
static int spatial_test(const Geometry *a,const Geometry *b,FILE *fp,double *maxerr)
{
    CMFGENState ca,cb;if(cmfgen_init(&ca,a)||cmfgen_init(&cb,b))return -1;double Sa[16],Sb[32],Ja[16],Jb[32],span=a->r_outer[a->n_shells-1]-a->r_inner[0];
    long double *x=malloc(512*sizeof(*x)),*w=malloc(512*sizeof(*w));
    if(!x||!w||ref_gauss(512,x,w))return -1;
    for(int s=0;s<a->n_shells;s++){Sa[s]=0.3;ca.chi_tot[(size_t)s*ca.n_bins]=10/span;}
    for(int s=0;s<b->n_shells;s++){Sb[s]=0.3;cb.chi_tot[(size_t)s*cb.n_bins]=10/span;}
    if(cmfgen_formal_bin_order(&ca,a,0,Sa,1,CMFGEN_SPHERICAL_GAUSS_ORDER,Ja,NULL,NULL,NULL)||cmfgen_formal_bin_order(&cb,b,0,Sb,1,CMFGEN_SPHERICAL_GAUSS_ORDER,Jb,NULL,NULL,NULL))return -1;
    for(int s=0;s<a->n_shells;s++){double r=0.5*(a->r_inner[s]+a->r_outer[s]);int k=center_index(b,r);if(k<0)return -1;
        double ref=(double)chord_J(a,s,10/span,0.3L,1,512,x,w);
        double ecf=rel(Ja[s],Jb[k]),ec=rel(Ja[s],ref),ef=rel(Jb[k],ref);
        *maxerr=fmax(*maxerr,fmax(ecf,fmax(ec,ef)));
        fprintf(fp,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",r,Ja[s],Jb[k],ref,ecf,ec,ef);}
    free(x);free(w);cmfgen_free(&ca);cmfgen_free(&cb);return 0;
}

static FILE *open_out(const char *dir,const char *name){char p[4096];snprintf(p,sizeof(p),"%s/%s",dir,name);return fopen(p,"wx");}
int main(int argc,char **argv)
{
    if(argc!=3){fprintf(stderr,"usage: %s ACTUAL_SHELL_CSV OUTDIR\n",argv[0]);return 2;}
    unsetenv("LUMINA_INNER_BB_SCALE");unsetenv("LUMINA_CMFGEN_LAMBDA_TRI");
    const double ce[]={1e14,1.3e14,1.8e14,2.5e14,3.5e14};
    const double ae[]={1e14,2e14,3e14,4e14};
    const double be[]={1e14,1.25e14,1.75e14,2e14,2.25e14,2.75e14,3e14,3.25e14,3.75e14,4e14};
    TestGeometry compact={0},actual={0},coarse={0},fine={0};
    if(geometry_from_edges(&compact,ce,4)||geometry_from_shell_csv(&actual,argv[1])||geometry_from_edges(&coarse,ae,3)||geometry_from_edges(&fine,be,9))return 2;
    FILE *v=open_out(argv[2],"vacuum.csv"),*a=open_out(argv[2],"absorption.csv"),*r=open_out(argv[2],"response.csv"),*al=open_out(argv[2],"ali.csv"),*sp=open_out(argv[2],"spatial.csv");if(!v||!a||!r||!al||!sp)return 73;
    fprintf(v,"geometry,shell,r_cm,J,W,relative_error\n");fprintf(a,"shell,radial_tau,order,J,reference_GL512,relative_error\n");fprintf(r,"case,observer,source,forward,Lambda,absolute_error\n");fprintf(al,"albedo,mode,iterations,relative_L1,matrix_residual\n");fprintf(sp,"r_cm,J_coarse,J_refined,J_chord_GL512,coarse_vs_refined_rel,coarse_vs_chord_rel,refined_vs_chord_rel\n");
    double mv=0,ma=0,mq=0,mr=0,ml=0,md=0,mt=0,ms=0;
    REQUIRE(vacuum_test("compact4",&compact.geo,v,&mv)==0,"compact vacuum");REQUIRE(vacuum_test("actual50",&actual.geo,v,&mv)==0,"actual vacuum");
    REQUIRE(absorption_test(&compact.geo,a,&ma,&mq)==0,"absorption chord");REQUIRE(response_thick_test(&compact.geo,r,&mr,&ml)==0,"response/thick Lambda");
    REQUIRE(ali_test(&compact.geo,al,&md,&mt)==0,"scattering ALI");REQUIRE(spatial_test(&coarse.geo,&fine.geo,sp,&ms)==0,"spatial split");
    fclose(v);fclose(a);fclose(r);fclose(al);fclose(sp);
    REQUIRE(mv<=1e-10,"vacuum W tolerance");REQUIRE(ma<=1e-4,"absorption tolerance");REQUIRE(mq<=1e-7,"reference convergence");REQUIRE(mr<=1e-11,"unit response tolerance");REQUIRE(ml==0,"thick Lambda strict range");REQUIRE(md<=1e-8,"diag ALI tolerance");REQUIRE(mt<=1e-8,"tri ALI tolerance");REQUIRE(ms<=1e-4,"spatial tolerance");
    CMFGENState invalid;REQUIRE(cmfgen_init(&invalid,&compact.geo)==0,"invalid fixture init");uint64_t lineage=invalid.stage32_lambda_generation;REQUIRE(cmfgen_solve_J(&invalid,&compact.geo,10000,0)!=0&&invalid.stage32_lambda_generation==lineage,"ALI0 fail closed");cmfgen_free(&invalid);
    FILE *sum=open_out(argv[2],"summary.txt");if(!sum)return 73;fprintf(sum,"overall=%s\ndefault_order=%d\nmax_vacuum_relative=%.17g\nmax_absorption_relative=%.17g\nmax_reference_q256_q512_relative=%.17g\nmax_forward_response_absolute=%.17g\nmax_thick_lambda_excess=%.17g\nmax_diag_ALI_relative_L1=%.17g\nmax_tri_ALI_relative_L1=%.17g\nmax_spatial_relative=%.17g\nraw_source=0.3\nraw_inner_boundary=1\nraw_radial_tau=0,0.1,1,10,100\n",failures?"FAIL":"PASS",CMFGEN_SPHERICAL_GAUSS_ORDER,mv,ma,mq,mr,ml,md,mt,ms);fclose(sum);
    printf("[SPHERICAL-TEST] %s vacuum=%.3e absorption=%.3e qref=%.3e response=%.3e thick=%.3e diag=%.3e tri=%.3e spatial=%.3e\n",failures?"FAIL":"PASS",mv,ma,mq,mr,ml,md,mt,ms);
    geometry_free(&compact);geometry_free(&actual);geometry_free(&coarse);geometry_free(&fine);return failures?1:0;
}
