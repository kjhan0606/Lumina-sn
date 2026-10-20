! 같은 LINMON 원식의 늦은 반복에만 기울기 후보를 제안한다. 해를 대체하지 않는다.
module lumina_mom_acceleration
use set_kind_module
use, intrinsic :: ieee_arithmetic
implicit none
integer, save :: accel_proposals=0,accel_accepted=0,accel_last_reason=0,accel_last_meta(3)=0
real(ldp), save :: accel_last_residual=0
contains
subroutine linmon_branch(n,r,x,lower,diagonal,upper,branch)
integer,intent(in)::n
real(ldp),intent(in)::r(n),x(n)
real(ldp),intent(out)::lower(n),diagonal(n),upper(n)
integer,intent(out)::branch(n)
integer i
real(ldp) hl,hr,sl,sr,p,s
lower=0;diagonal=0;upper=0;branch=0
! 두 경계행은 이 비선형 기울기를 소비하지 않는다.
do i=2,n-1
 hl=r(i)-r(i-1);hr=r(i+1)-r(i)
 sl=(x(i)-x(i-1))/hl;sr=(x(i+1)-x(i))/hr
 p=(sl*hr+sr*hl)/(hl+hr)
 s=sign(1._ldp,sl)+sign(1._ldp,sr)
 if(s==0)then
  branch(i)=4
 else if(abs(sl)<=abs(sr).and.abs(sl)<=.5_ldp*abs(p))then
  s=s*sign(1._ldp,sl);branch(i)=1
  lower(i)=-s/hl;diagonal(i)=s/hl
 else if(abs(sr)<=.5_ldp*abs(p))then
  s=s*sign(1._ldp,sr);branch(i)=2
  diagonal(i)=-s/hr;upper(i)=s/hr
 else
  s=s*.5_ldp*sign(1._ldp,p);branch(i)=3
  lower(i)=-s*hr/(hl*(hl+hr));upper(i)=s*hl/(hr*(hl+hr))
  diagonal(i)=s*(hr/(hl*(hl+hr))-hl/(hr*(hl+hr)))
 end if
end do
end subroutine

subroutine validate_candidate(n,r,base,alpha,x,branch,hu,hl,hs,eps,epsp,jp,hp,flux_constant,ok,slope,reason,err)
integer,intent(in)::n,branch(n)
real(ldp),intent(in)::r(n),base(4,n),alpha(n),x(n),hu(n),hl(n),hs(n),eps(n),epsp(n),jp(n),hp(n),flux_constant
logical,intent(out)::ok
real(ldp),intent(out)::slope(n),err
integer,intent(out)::reason
real(ldp) a(n),b(n),c(n),derivative(n),value,den,h
integer fresh(n),i
ok=.false.;slope=0;err=huge(1._ldp);reason=10
if(.not.all(ieee_is_finite(x)).or.any(x<=0))return
call linmon_branch(n,r,x,a,b,c,fresh)
reason=11
if(any(fresh/=branch))return
call derivchi(derivative,x,r,n,'LINMON')
err=0
do i=1,n
 value=base(2,i)*x(i)+alpha(i)*r(i)*derivative(i)-base(4,i)
 den=abs(base(2,i)*x(i))+abs(alpha(i)*r(i)*derivative(i))+abs(base(4,i))
 if(i>1)then
  value=value+base(1,i)*x(i-1);den=den+abs(base(1,i)*x(i-1))
 end if
 if(i<n)then
  value=value+base(3,i)*x(i+1);den=den+abs(base(3,i)*x(i+1))
 end if
 if(den<=0.or..not.ieee_is_finite(value).or..not.ieee_is_finite(den))return
 err=max(err,abs(value)/den)
end do
reason=12
if(err>1.e-10_ldp)return
reason=13
do i=1,n-1
 h=hu(i)*x(i+1)-hl(i)*x(i)+hs(i)*hp(i)+epsp(i)*(jp(i)+jp(i+1))-eps(i)*(x(i)+x(i+1))
 if(i==1)h=h+flux_constant
 if(.not.ieee_is_finite(h).or.abs(h)>max(x(i),x(i+1)))return
end do
slope=r*derivative/x
reason=14
if(.not.all(ieee_is_finite(slope)))return
reason=0;ok=.true.
end subroutine

subroutine propose_gradient(n,r,tri,used_slope,alpha,x,hu,hl,hs,eps,epsp,jp,hp,flux_constant,ok,slope)
integer,intent(in)::n
real(ldp),intent(in)::r(n),tri(4,n),used_slope(n),alpha(n),x(n),hu(n),hl(n),hs(n),eps(n),epsp(n),jp(n),hp(n),flux_constant
logical,intent(out)::ok
real(ldp),intent(out)::slope(n)
real(ldp) base(4,n),a(n),b(n),c(n),candidate(n),lower(n),diagonal(n),upper(n)
integer branch(n)
accel_proposals=accel_proposals+1
base=tri;base(2,:)=base(2,:)-alpha*used_slope
call linmon_branch(n,r,x,lower,diagonal,upper,branch)
a=base(1,:)+alpha*r*lower
b=base(2,:)+alpha*r*diagonal
c=base(3,:)+alpha*r*upper
candidate=base(4,:)
! 이 풀이도 운전자의 100회 계산 예산에서 1회를 차감한다.
call thomas(a,b,c,candidate,n,1)
call validate_candidate(n,r,base,alpha,candidate,branch,hu,hl,hs,eps,epsp,jp,hp,flux_constant, &
 ok,slope,accel_last_reason,accel_last_residual)
if(ok)accel_accepted=accel_accepted+1
end subroutine

subroutine lumina_accel_summary()
integer lu
open(newunit=lu,file='MOM_ACCEL_SUMMARY.txt',status='replace',action='write')
write(lu,'(A)')'CANDIDATE_SOLVES_ACCEPTED_LAST_REASON_META_RESIDUAL_NOT_PHYSICS_PASS'
write(lu,*)accel_proposals,accel_accepted,accel_last_reason,accel_last_meta
write(lu,*)accel_last_residual
close(lu)
end subroutine
end module
