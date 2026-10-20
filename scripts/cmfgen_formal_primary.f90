! 같은 angular RTE의 source 고정점을 공통장으로 쓴다. 물리 플래그를 변경하지 않는다.
module lumina_formal_primary
use set_kind_module
use, intrinsic :: ieee_arithmetic
use mod_ray_mom_store,only:nd_store,jnu_store,hnu_store,knu_store,nnu_store
use mod_space_grid_v2,only:ray,jqw_p,jqw_m
implicit none
logical,save::fp_initialized=.false.,fp_on=.false.,fp_have_h=.false.,fp_ready=.false.,fp_in_call=.false.
real(ldp),save::fp_freq=0,fp_j_residual=huge(1._ldp),fp_h_residual=huge(1._ldp)
real(ldp),save::fp_source_residual=huge(1._ldp),fp_f_change=huge(1._ldp),fp_support_min=0
integer,save::fp_count=0,fp_calls=0
type fp_blue_ray
 real(ldp),allocatable::i_p(:),i_m(:)
end type
type(fp_blue_ray),allocatable,save::fp_blue(:)
real(ldp),allocatable,save::fp_source_j(:),fp_eta_true(:),fp_scattering(:),fp_previous_h(:),fp_previous_f(:)
real(ldp),allocatable,save::fp_lambda_columns(:,:,:),fp_x_history(:,:),fp_p_history(:,:)
! 원래 20회 예산 안에서 얻은 잔차를 버리지 않는다. 추가 formal 호출은 없다.
integer,parameter::fp_memory=19
integer,save::fp_history_count=0,fp_ali_proposals=0,fp_anderson_proposals=0,fp_proposal_rejected=0
contains
logical function fp_enabled()
character(len=16) value
if(.not.fp_initialized)then
 call get_environment_variable('LUMINA_FORMAL_PRIMARY',value)
 if(value/=' '.and.value/='1')stop 85
 fp_on=value=='1';fp_initialized=.true.
end if
fp_enabled=fp_on
end function

subroutine fp_fail(message)
character(len=*) message
integer lu,n
character(len=16) tag
open(newunit=lu,file='FORMAL_PRIMARY_FAILURE.txt',status='new',action='write')
write(lu,'(A)')trim(message)
write(lu,*)fp_freq,fp_count,fp_j_residual,fp_source_residual,fp_h_residual,fp_f_change,fp_support_min
if(allocated(fp_source_j))write(lu,*)fp_source_j
close(lu)
! total eta의 차감 복원 대신 실제 세 source 배열을 직접 보존한다.
if(allocated(fp_source_j).and.allocated(fp_eta_true).and.allocated(fp_scattering))then
 n=size(fp_source_j);tag='FPRIMARYSOURCE1'
 open(newunit=lu,file='FORMAL_PRIMARY_SOURCE.bin',status='new',access='stream',form='unformatted')
 write(lu)tag,n,fp_count,fp_calls,fp_freq,fp_source_j,fp_eta_true,fp_scattering
 close(lu)
end if
call fp_primary_failure()
write(6,'(A)')'FORMAL_PRIMARY_REJECTED: '//trim(message)
stop 95
end subroutine

real(ldp) function fp_relative(a,b)
real(ldp),intent(in)::a(:),b(:)
real(ldp) d
integer i
fp_relative=0
do i=1,size(a)
 d=max(abs(a(i)),abs(b(i)))
 if(d>0)then
  fp_relative=max(fp_relative,abs(a(i)-b(i))/d)
 else if(a(i)/=b(i))then
  fp_relative=huge(1._ldp)
 end if
end do
end function

subroutine fp_start(freq,count,n,j,eta,scattering,coherent,use_rel,use_formal)
integer,intent(in)::count,n
real(ldp),intent(in)::freq,j(n),eta(n),scattering(n)
logical,intent(in)::coherent,use_rel,use_formal
if(.not.fp_enabled())return
if(.not.coherent.or..not.use_rel.or..not.use_formal)call fp_fail('unsupported physical setting')
if(count==0)then
 fp_freq=freq;fp_have_h=.false.;fp_ready=.false.;fp_calls=0;fp_history_count=0
else if(freq/=fp_freq.or.count/=fp_calls)then
 call fp_fail('frequency/iteration history mismatch')
end if
fp_count=count
fp_source_j=j;fp_eta_true=eta;fp_scattering=scattering
if(.not.all(ieee_is_finite(j)).or.any(j<0).or..not.all(ieee_is_finite(eta)).or.any(eta<0))call fp_fail('invalid source input')
if(.not.all(ieee_is_finite(scattering)).or.any(scattering<0))call fp_fail('invalid scattering input')
fp_in_call=.true.
end subroutine

real(ldp) function fp_local_psi(tau)
real(ldp),intent(in)::tau
fp_local_psi=0
if(tau<=0)return
if(tau<1.e-3_ldp)then
 fp_local_psi=tau*(.5_ldp-tau*(1._ldp/6-tau*(1._ldp/24-tau/120)))
else
 fp_local_psi=1-(1-exp(-tau))/tau
end if
end function

subroutine fp_lambda_begin(n,np)
integer,intent(in)::n,np
if(.not.fp_enabled().or..not.fp_in_call)return
if(fp_calls>0)return
if(allocated(fp_lambda_columns))then
 if(size(fp_lambda_columns,1)/=n.or.size(fp_lambda_columns,3)/=np)deallocate(fp_lambda_columns)
end if
if(.not.allocated(fp_lambda_columns))allocate(fp_lambda_columns(n,n,np))
fp_lambda_columns=0
end subroutine

subroutine fp_lambda_ray(ip,nray,n,np,nc,r,chi,scattering,dlog,initialize)
integer,intent(in)::ip,nray,n,np,nc
real(ldp),intent(in)::r(n),chi(nray),scattering(n),dlog
logical,intent(in)::initialize
real(ldp) invnu,cp(nray),cm(nray),tau,psi,ee,e0,w
real(ldp) emission(n,nray),response(n,nray),outward(n)
integer i,k,l
if(.not.fp_enabled().or..not.fp_in_call)return
if(fp_calls>0)return
invnu=0
if(.not.initialize)invnu=1/dlog
cp=chi+(invnu+3)*ray(ip)%b_p
cm=chi+(invnu+3)*ray(ip)%b_m
if(any(cp<=0).or.any(cm<=0))return
! 선형 source의 비국소 angular Lambda를 가속에만 쓴다.
! 실제 formal 적분의 포물선/단조 보간과 최종 잔차는 바꾸지 않는다.
emission=0;l=1
do k=1,nray
 if(ray(ip)%r_ray(k)>=r(1))then
  emission(1,k)=scattering(1)
 else if(ray(ip)%r_ray(k)<=r(n))then
  emission(n,k)=scattering(n)
 else
  do while(ray(ip)%r_ray(k)<r(l+1).and.l<n-1)
   l=l+1
  end do
  w=(ray(ip)%r_ray(k)-r(l+1))/(r(l)-r(l+1))
  emission(l,k)=w*scattering(l);emission(l+1,k)=(1-w)*scattering(l+1)
 end if
end do
response=0
do k=2,nray
 tau=.5_ldp*(cm(k)+cm(k-1))*(ray(ip)%s_m(k)-ray(ip)%s_m(k-1))
 call fp_sc_weights(tau,ee,e0,psi)
 response(:,k)=ee*response(:,k-1)+(e0-psi)*emission(:,k-1)/cm(k-1)+psi*emission(:,k)/cm(k)
end do
do i=1,min(n,np+1-ip)
 k=ray(ip)%lnk(i)
 fp_lambda_columns(i,:,ip)=jqw_m(i,ip)*response(:,k)
end do
outward=0
if(ip>nc)outward=response(:,nray)
do k=nray,1,-1
 if(k<nray)then
  tau=.5_ldp*(cp(k)+cp(k+1))*(ray(ip)%s_p(k)-ray(ip)%s_p(k+1))
  call fp_sc_weights(tau,ee,e0,psi)
  outward=ee*outward+(e0-psi)*emission(:,k+1)/cp(k+1)+psi*emission(:,k)/cp(k)
 end if
 response(:,k)=outward
end do
do i=1,min(n,np+1-ip)
 k=ray(ip)%lnk(i)
 ! 각 thread는 자기 광선의 행렬만 쓴다. 합산 순서는 formal 호출과 무관하다.
 fp_lambda_columns(i,:,ip)=fp_lambda_columns(i,:,ip)+jqw_p(i,ip)*response(:,k)
end do
end subroutine

subroutine fp_sc_weights(tau,ee,e0,psi)
real(ldp),intent(in)::tau
real(ldp),intent(out)::ee,e0,psi
ee=exp(-max(0._ldp,tau));e0=1-ee;psi=fp_local_psi(tau)
if(tau>0.and.tau<1.e-3_ldp)e0=tau*(1-tau*(.5_ldp-tau*(1._ldp/6-tau*(1._ldp/24-tau/120))))
end subroutine

subroutine fp_solve_preconditioner(n,a,b,info)
integer,intent(in)::n
real(ldp),intent(inout)::a(n,n),b(n)
integer,intent(out)::info
real(ldp) row(n),value,factor
integer i,k,p
info=0
do k=1,n
 p=k-1+maxloc(abs(a(k:n,k)),dim=1)
 if(a(p,k)==0.or..not.ieee_is_finite(a(p,k)))then
  info=k;return
 end if
 if(p/=k)then
  row=a(k,:);a(k,:)=a(p,:);a(p,:)=row
  value=b(k);b(k)=b(p);b(p)=value
 end if
 do i=k+1,n
  factor=a(i,k)/a(k,k)
  a(i,k+1:n)=a(i,k+1:n)-factor*a(k,k+1:n)
  b(i)=b(i)-factor*b(k)
 end do
end do
do i=n,1,-1
 b(i)=(b(i)-dot_product(a(i,i+1:n),b(i+1:n)))/a(i,i)
end do
if(.not.all(ieee_is_finite(b)))info=n+1
end subroutine

subroutine fp_next_source(n,j_formal,next_j)
integer,intent(in)::n
real(ldp),intent(in)::j_formal(n)
real(ldp),intent(out)::next_j(n)
real(ldp) correction(n),ali(n),candidate(n),scale(n),col(n),rhs(n),q(n,fp_memory),upper(fp_memory,fp_memory)
real(ldp) dx(n,fp_memory),dp(n,fp_memory),coefficient(fp_memory),before,after,factor
real(ldp) matrix(n,n),raw_correction(n)
integer i,l,m,rank,pass,info
correction=j_formal-fp_source_j
if(allocated(fp_lambda_columns))then
 raw_correction=correction;matrix=-sum(fp_lambda_columns,dim=3)
 do i=1,n
  matrix(i,i)=1+matrix(i,i)
 end do
 call fp_solve_preconditioner(n,matrix,correction,info)
 if(info/=0.or..not.all(ieee_is_finite(correction)))correction=raw_correction
end if
ali=fp_source_j+correction;next_j=j_formal
if(all(ieee_is_finite(ali)).and.all(ali>=0))then
 next_j=ali;fp_ali_proposals=fp_ali_proposals+1
else
 fp_proposal_rejected=fp_proposal_rejected+1
 correction=j_formal-fp_source_j
end if
if(allocated(fp_x_history))then
 if(size(fp_x_history,1)/=n)deallocate(fp_x_history,fp_p_history)
end if
if(.not.allocated(fp_x_history))allocate(fp_x_history(n,fp_memory+1),fp_p_history(n,fp_memory+1))
if(fp_history_count==fp_memory+1)then
 fp_x_history(:,1:fp_memory)=fp_x_history(:,2:fp_memory+1)
 fp_p_history(:,1:fp_memory)=fp_p_history(:,2:fp_memory+1)
 fp_history_count=fp_memory
end if
fp_history_count=fp_history_count+1
fp_x_history(:,fp_history_count)=fp_source_j;fp_p_history(:,fp_history_count)=correction
m=fp_history_count-1
if(m==0)return
scale=max(abs(fp_source_j),abs(j_formal));rhs=0
where(scale>0)rhs=correction/scale
rank=0;upper=0;coefficient=0
do l=m,1,-1
 col=0
 where(scale>0)col=(fp_p_history(:,l+1)-fp_p_history(:,l))/scale
 if(.not.all(ieee_is_finite(col)))cycle
 before=norm2(col)
 if(before==0)cycle
 rank=rank+1
 dx(:,rank)=fp_x_history(:,l+1)-fp_x_history(:,l)
 dp(:,rank)=fp_p_history(:,l+1)-fp_p_history(:,l)
 do pass=1,2
  do i=1,rank-1
   factor=dot_product(q(:,i),col);upper(i,rank)=upper(i,rank)+factor;col=col-factor*q(:,i)
  end do
 end do
 after=norm2(col)
 if(after<=64*epsilon(1._ldp)*before)then
  upper(:,rank)=0;rank=rank-1;cycle
 end if
 upper(rank,rank)=after;q(:,rank)=col/after
 coefficient(rank)=dot_product(q(:,rank),rhs)
end do
if(rank==0)return
do i=rank,1,-1
 coefficient(i)=(coefficient(i)-dot_product(upper(i,i+1:rank),coefficient(i+1:rank)))/upper(i,i)
end do
candidate=fp_source_j+correction
do i=1,rank
 candidate=candidate-(dx(:,i)+dp(:,i))*coefficient(i)
end do
if(all(ieee_is_finite(candidate)).and.all(candidate>=0))then
 next_j=candidate;fp_anderson_proposals=fp_anderson_proposals+1
else
 fp_proposal_rejected=fp_proposal_rejected+1
end if
end subroutine

subroutine fp_check_emissivity(n,freq,eta,scattering)
integer,intent(in)::n
real(ldp),intent(in)::freq,eta(n),scattering(n)
real(ldp) expected(n)
if(.not.fp_enabled().or..not.fp_in_call)return
if(freq/=fp_freq.or.n/=size(fp_source_j))call fp_fail('formal call does not belong to source trial')
expected=fp_eta_true+fp_scattering*fp_source_j
if(.not.all(ieee_is_finite(eta)).or.any(scattering/=fp_scattering))call fp_fail('nonfinite/wrong scattering source')
if(fp_relative(expected,eta)>16*epsilon(1._ldp))call fp_fail('actual emissivity does not match source trial')
end subroutine

subroutine fp_angular_f(n,j,h,k,nn,f)
integer,intent(in)::n
real(ldp),intent(in)::j(n),h(n),k(n),nn(n)
real(ldp),intent(out)::f(n)
integer i
if(.not.all(ieee_is_finite(j)).or..not.all(ieee_is_finite(h)).or..not.all(ieee_is_finite(k)).or..not.all(ieee_is_finite(nn)))call fp_fail('nonfinite angular moments')
do i=1,n
 if(j(i)>0)then
  f(i)=k(i)/j(i)
 else if(j(i)==0.and.h(i)==0.and.k(i)==0.and.nn(i)==0)then
  ! 영장에서는 소비되지 않는 K/J 표현만0으로 둔다. 광장을 floor로 만들지 않는다.
  f(i)=0
 else
  call fp_fail('negative or inconsistent zero angular field')
 end if
end do
end subroutine

subroutine fp_record(n,np,j,f,thick_outer)
integer,intent(in)::n,np
real(ldp),intent(in)::j(n),f(n)
logical,intent(in)::thick_outer
real(ldp) eta_before(n),eta_after(n),scale,a,b,c,s,h,k,nn,low,roundoff
integer i,ip,sign_index,ray_limit
if(.not.fp_enabled())return
if(.not.fp_in_call)call fp_fail('formal result without a source trial')
if(n/=nd_store.or..not.allocated(fp_source_j))call fp_fail('angular/source dimensions mismatch')
if(any(j/=jnu_store).or..not.all(ieee_is_finite(f)))call fp_fail('field identity/nonfinite closure')
ray_limit=np-1
if(thick_outer)ray_limit=np
if(allocated(fp_blue))then
 if(size(fp_blue)/=ray_limit)deallocate(fp_blue)
end if
if(.not.allocated(fp_blue))allocate(fp_blue(ray_limit))
do ip=1,ray_limit
 if(.not.all(ieee_is_finite(ray(ip)%i_p)).or..not.all(ieee_is_finite(ray(ip)%i_m)))call fp_fail('nonfinite ray intensity')
 if(any(ray(ip)%i_p<0).or.any(ray(ip)%i_m<0))call fp_fail('negative ray intensity')
 if(fp_calls==0)then
  fp_blue(ip)%i_p=ray(ip)%i_p_prev;fp_blue(ip)%i_m=ray(ip)%i_m_prev
 else
  if(any(fp_blue(ip)%i_p/=ray(ip)%i_p_prev).or.any(fp_blue(ip)%i_m/=ray(ip)%i_m_prev))call fp_fail('blue ray changed within source iteration')
 end if
end do
fp_j_residual=fp_relative(j,fp_source_j)
eta_before=fp_eta_true+fp_scattering*fp_source_j
eta_after=fp_eta_true+fp_scattering*j
if(.not.all(ieee_is_finite(eta_before)).or..not.all(ieee_is_finite(eta_after)))call fp_fail('nonfinite total source')
fp_source_residual=fp_relative(eta_before,eta_after)
fp_h_residual=huge(1._ldp);fp_f_change=huge(1._ldp)
if(fp_have_h)then
 fp_h_residual=0
 do i=1,n
  scale=abs(j(i))
  if(scale>0)then
   fp_h_residual=max(fp_h_residual,abs(hnu_store(i)-fp_previous_h(i))/scale)
  else if(hnu_store(i)/=fp_previous_h(i))then
   fp_h_residual=huge(1._ldp)
  end if
 end do
 fp_f_change=maxval(abs(f-fp_previous_f))
end if
fp_support_min=huge(1._ldp)
do i=1,n
 if(j(i)==0)then
  low=0
 else
  h=hnu_store(i)/j(i);k=knu_store(i)/j(i);nn=nnu_store(i)/j(i)
  low=huge(1._ldp)
  do sign_index=-1,1,2
   s=real(sign_index,ldp)
   a=1+s*h;b=h+s*k;c=k+s*nn
   low=min(low,.5_ldp*(a+c-sqrt((a-c)**2+4*b*b)))
  end do
 end if
 fp_support_min=min(fp_support_min,low)
end do
! 반올림 판독이며 모멘트를 바꾸지 않는다. 양의 선형각보간의 적분은 PSD다.
roundoff=64*epsilon(1._ldp)*(np+16)
if(fp_support_min< -roundoff)call fp_fail('angular moment support violation')
fp_previous_h=hnu_store;fp_previous_f=f;fp_have_h=.true.;fp_ready=.true.;fp_calls=fp_calls+1;fp_in_call=.false.
end subroutine

subroutine fp_end_iteration(count,old_f_change,f_tolerance,inaccurate)
integer,intent(in)::count
real(ldp),intent(in)::old_f_change,f_tolerance
logical,intent(inout)::inaccurate
if(.not.fp_enabled())return
if(.not.fp_ready.or.count/=fp_calls)call fp_fail('missing actual formal trial')
inaccurate=inaccurate.or.max(fp_j_residual,fp_h_residual,fp_source_residual)>1.e-8_ldp.or.fp_f_change>f_tolerance
if(count>=20.and.inaccurate)call fp_fail('source fixed-point iteration cap')
end subroutine
end module
