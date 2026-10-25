/* 선택 빌드 전용 지수-이차 모멘트 실험 후보. 물리/공간 근사는 바꾸지 않는다.
 * 2026-10-21 실제 native13003 비용/완료 기준 미달: 생산 채택 기각.
 * 수치 대조/재현용으로만 보존하며 기본 경로를 이 엔진으로 바꾸지 않는다.
 * 정본 증거: docs/HANDOVER_2026-10-21.md의 실제 엔진 판정 기록. */
#ifndef CMF_EXPQUAD_ENGINE_H
#define CMF_EXPQUAD_ENGINE_H

static _Thread_local uint64_t exq_attempts,exq_accepted;
void cmf_history_moments_engine_usage(uint64_t *attempts,uint64_t *accepted,int reset)
{
    if(reset)exq_attempts=exq_accepted=0;
    if(attempts)*attempts=exq_attempts;
    if(accepted)*accepted=exq_accepted;
}

typedef struct {long double b,endpoint,m[132],e[132];int count;} EXQLinear;
static int exq_linear(EXQLinear *c,int order)
{
    if(order>131||c->b<0||!isfinite(c->b))return 1;
    const long double eps=LDBL_EPSILON;
    if(c->count==0)c->endpoint=expl(-c->b);
    for(int n=c->count;n<=order;n++){
        long double value,error,b=c->b;
        if(b==0){value=1.L/(n+1);error=4*eps*value;}
        else if(n==0&&b>=1){value=(1-c->endpoint)/b;error=16*eps*value;}
        else if(n>0&&b>=n+1){
            long double endpoint=c->endpoint,first=n*c->m[n-1];
            value=(first-endpoint)/b;
            error=(n*c->e[n-1]+16*eps*(fabsl(first)+endpoint))/b+4*eps*fabsl(value);
        }else{
            long double term=1.L/(n+1),sum=term,tail=0.;int k;
            for(k=1;k<512;k++){
                term*=b/(n+k+1.L);sum+=term;
                long double ratio=b/(n+k+2.L);
                if(ratio<1){tail=term*ratio/(1-ratio);if(tail<=8*eps*sum)break;}
            }
            if(k==512)return 1;
            long double scale=c->endpoint;value=scale*sum;
            error=scale*(tail+16*(k+1)*eps*sum);
        }
        if(!(value>=0)||!isfinite(value)||!isfinite(error))return 1;
        c->m[n]=value;c->e[n]=error;c->count=n+1;
    }
    return 0;
}

static int exq_linear_weights(EXQLinear *c,int p,long double w[3],long double e[3])
{
    if(exq_linear(c,p+2))return 1;
    long double a=c->m[p],b=c->m[p+1],d=c->m[p+2],eps=LDBL_EPSILON;
    w[0]=d;w[1]=b-d;w[2]=a-2*b+d;
    e[0]=c->e[p+2];e[1]=c->e[p+1]+c->e[p+2]+8*eps*(b+d);
    e[2]=c->e[p]+2*c->e[p+1]+c->e[p+2]+16*eps*(a+2*b+d);
    return w[0]>0&&w[1]>0&&w[2]>0 ? 0 : 1;
}

/* e^(-bu) 주위의 곡률 급수. a>=0의 Lagrange 나머지는 e^0로,
 * a<0은 e^(|a|u²)<=e^(|a|u)로 가중 적분해 제한한다. */
static int exq_curvature(long double a,long double b,long double tolerance,long double out[3],long double error[3])
{
    if(b<0||b+(a<0?a:0)<0)return 1;
    EXQLinear base,bound;base.b=b;base.count=0;bound.b=b+(a<0?a:0);bound.count=0;
    long double sum[3]={0,0,0},absolute[3]={0,0,0},acc_error[3]={0,0,0};
    long double coefficient=1,previous=HUGE_VALL;
    for(int n=0;n<=32;n++){
        long double w[3],e[3],r[3],re[3];
        if(exq_linear_weights(&base,2*n,w,e))return 1;
        for(int k=0;k<3;k++){
            long double term=coefficient*w[k];sum[k]+=term;absolute[k]+=fabsl(term);
            acc_error[k]+=fabsl(coefficient)*e[k]+32*(n+1)*LDBL_EPSILON*fabsl(term);
        }
        long double next=coefficient*(-a)/(n+1.L);
        if(exq_linear_weights(a>=0?&base:&bound,2*n+2,r,re))return 1;
        int met=1;long double maximum=0;
        for(int k=0;k<3;k++){
            error[k]=acc_error[k]+fabsl(next)*(r[k]+re[k])+32*LDBL_EPSILON*absolute[k];
            if(!(sum[k]>0)||error[k]>tolerance*sum[k])met=0;
            long double relative=fabsl(next)*(r[k]+re[k])/(sum[k]>0?sum[k]:1);
            if(relative>maximum)maximum=relative;
        }
        if(met){for(int k=0;k<3;k++)out[k]=sum[k];return 0;}
        if(n>=4&&maximum>=previous)return 1;
        previous=maximum;coefficient=next;
        if(!isfinite(coefficient))return 1;
    }
    return 1;
}

/* erfcx의 큰 양수 영역은 Stieltjes 점근급수의 다음 항으로 나머지를 제한한다. */
static int exq_erfcx(long double x,long double *value,long double *error)
{
    const long double root_pi=1.77245385090551602729816748334114518L;
    if(x<0||!isfinite(x))return 1;
    if(x<24){*value=expl(x*x)*erfcl(x);*error=64*LDBL_EPSILON*fabsl(*value);return 0;}
    long double term=1,sum=1,previous=1;
    for(int n=1;n<64;n++){
        term*=-(2*n-1.L)/(2*x*x);
        if(fabsl(term)>=previous)return 1;
        sum+=term;long double next=fabsl(term)*(2*n+1.L)/(2*x*x);
        if(next<=8*LDBL_EPSILON*fabsl(sum)){
            *value=sum/(root_pi*x);*error=(next+64*n*LDBL_EPSILON*fabsl(sum))/(root_pi*x);return 0;
        }
        previous=fabsl(term);
    }
    return 1;
}

static int exq_dawson(long double x,long double *value,long double *error)
{
    long double sign=x<0?-1:1;x=fabsl(x);long double y=x*x;
    if(x==0){*value=*error=0;return 0;}
    if(x>=12){
        long double w[3],e[3];
        if(!exq_curvature(-y,2*y,4e-17L,w,e)){
            *value=sign*x*(w[0]+2*w[1]+w[2]);*error=x*(e[0]+2*e[1]+e[2])+32*LDBL_EPSILON*fabsl(*value);return 0;
        }
    }
    if(x>=24)return 1;
    long double term=1,sum=1,tail=0;int n;
    for(n=1;n<2048;n++){
        term*=y/n*(2*n-1.L)/(2*n+1.L);sum+=term;
        long double ratio=y/(n+1.L);
        if(ratio<1){tail=term*ratio/(1-ratio);if(tail<=8*LDBL_EPSILON*sum)break;}
    }
    if(n==2048)return 1;
    long double scale=x*expl(-y);*value=sign*scale*sum;
    *error=scale*(tail+32*(n+1)*LDBL_EPSILON*sum);return 0;
}

static int exq_closed(long double a,long double b,long double out[3])
{
    if(a==0)return 1;
    const long double root_pi=1.77245385090551602729816748334114518L;
    long double eps=128*LDBL_EPSILON,t=expl(-a-b),root=sqrtl(fabsl(a));
    long double m0,m1,m2,e0,e1,e2;
    if(a>0&&b<0&&b+2*a>0){
        /* 내부 gain 봉우리의 중심 모멘트는 약한 끝점 모멘트를 차감하지 않는다. */
        long double c=-b/(2*a),z=root*c,y=root*(1-c),peak=b*b/(4*a);
        if(a<8||peak>700)return 1;
        long double left=expl(-z*z),right=expl(-y*y);
        long double v0=root_pi*(erfl(z)+erfl(y))/(2*root),v1=(left-right)/(2*a);
        long double v2=(v0-c*left-(1-c)*right)/(2*a),scale=expl(peak);
        long double terms[3][3]={{c*c*v0,2*c*v1,v2},
            {c*(1-c)*v0,(1-2*c)*v1,-v2},{(1-c)*(1-c)*v0,-2*(1-c)*v1,v2}};
        for(int k=0;k<3;k++){
            long double sum=terms[k][0]+terms[k][1]+terms[k][2];
            long double err=eps*(fabsl(terms[k][0])+fabsl(terms[k][1])+fabsl(terms[k][2])+v0/a);
            if(!(sum>0)||err>4e-13L*sum)return 1;
            out[k]=scale*sum;
        }
        return 0;
    }
    long double first,last,ef,el;
    if(b<0)return 1;
    if(a>0){
        if(exq_erfcx(b/(2*root),&first,&ef)||exq_erfcx((b+2*a)/(2*root),&last,&el))return 1;
        long double scale=root_pi/(2*root);m0=scale*(first-t*last);
        e0=scale*(ef+t*el+eps*(fabsl(first)+fabsl(t*last)));
    }else{
        if(exq_dawson(b/(2*root),&first,&ef)||exq_dawson((b+2*a)/(2*root),&last,&el))return 1;
        m0=(first-t*last)/root;e0=(ef+t*el+eps*(fabsl(first)+fabsl(t*last)))/root;
    }
    m1=(1-t-b*m0)/(2*a);e1=(fabsl(b)*e0+eps*(1+fabsl(t)+fabsl(b*m0)))/(2*fabsl(a));
    m2=(m0-t-b*m1)/(2*a);e2=(e0+fabsl(b)*e1+eps*(fabsl(m0)+fabsl(t)+fabsl(b*m1)))/(2*fabsl(a));
    out[0]=m2;out[1]=m1-m2;out[2]=m0-2*m1+m2;
    long double e[3]={e2,e1+e2+eps*(fabsl(m1)+fabsl(m2)),
        e0+2*e1+e2+eps*(fabsl(m0)+2*fabsl(m1)+fabsl(m2))};
    for(int k=0;k<3;k++)if(!(out[k]>0)||e[k]>4e-13L*out[k]||!isfinite(out[k]))return 1;
    return 0;
}

/* 보통 크기의 지수는 양의 Bernstein 계수 급수로 평가한다. 각 계수와
 * 세 적분은 양의 합이며 미래 항의 기하 상한 및 누적 반올림을 함께 제한한다. */
static int exq_positive_series(long double a,long double b,long double out[3])
{
    long double p[3]={0,-b/2,-a-b},minimum=fminl(0,fminl(p[1],p[2])),q=0;
    for(int j=0;j<3;j++){p[j]-=minimum;if(p[j]>q)q=p[j];}
    if(q>64)return 1;
    long double current[513],next[513],sum[3]={1.L/3,1.L/6,1.L/3};current[0]=1;
    for(int n=1;n<=256;n++){
        int degree=2*n,old=degree-2;long double max_term=0,den=(long double)degree*(degree-1)*n;
        for(int k=0;k<=degree;k++){
            long double value=0;
            if(k<=old)value+=(degree-k)*(degree-k-1.L)*p[0]*current[k];
            if(k>=1&&k-1<=old)value+=2.L*k*(degree-k)*p[1]*current[k-1];
            if(k>=2)value+=(long double)k*(k-1)*p[2]*current[k-2];
            next[k]=value/den;if(next[k]>max_term)max_term=next[k];
            long double d=(degree+1.L)*(degree+2.L)*(degree+3.L);
            sum[0]+=next[k]*(k+1.L)*(k+2.L)/d;
            sum[1]+=next[k]*(k+1.L)*(degree-k+1.L)/d;
            sum[2]+=next[k]*(degree-k+1.L)*(degree-k+2.L)/d;
        }
        if(n+2>q){
            long double tail=max_term*q/(n+1.L)/(1-q/(n+2.L));int met=1;
            for(int k=0;k<3;k++){
                long double base=k==1?1.L/6:1.L/3;
                if(tail*base+(64.L*n*n+64)*LDBL_EPSILON*sum[k]>4e-13L*sum[k])met=0;
            }
            if(met){for(int k=0;k<3;k++)out[k]=expl(minimum)*sum[k];return 0;}
        }
        memcpy(current,next,(size_t)(degree+1)*sizeof(long double));
    }
    return 1;
}

int cmf_history_moments_engine(double q0,double q1,double *out)
{
    exq_attempts++;
    if(!out||!isfinite(q0)||!isfinite(q1)||!isfinite(q0-q1)||LDBL_MANT_DIG<64)return 1;
    long double a=((long double)q0-q1)/2,b=q1,scale=1,w[3],e[3];int reflected=0;
    if(a+b<0){scale=expl(-a-b);b=-q0;reflected=1;}
    if(!isfinite(scale))return 1;
    int status=1;
    if(b>=0&&(fabsl(a)<=.25L||fabsl(a)/((1+b)*(1+b))<.001L))
        status=exq_curvature(a,b,4e-13L,w,e);
    if(status)status=exq_closed(a,b,w);
    if(status)status=exq_curvature(a,b,4e-13L,w,e);
    if(status)status=exq_positive_series(a,b,w);
    if(status)return 1;
    double result[3];
    for(int k=0;k<3;k++){
        long double value=scale*w[reflected?2-k:k];
        result[k]=(double)value;if(!(value>=0)||!isfinite(result[k]))return 1;
    }
    memcpy(out,result,sizeof(result));exq_accepted++;return 0;
}
#endif
