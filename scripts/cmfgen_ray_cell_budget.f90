! 광선 이력의 읽기 전용 셀 적분. J/H/blue 및 source 고정점을 변경하지 않는다.
module lumina_ray_cell_budget
use set_kind_module
use mod_space_grid_v2
use lumina_positive_sc,only:sc_control
use lumina_sc_cell_integral
use,intrinsic::ieee_arithmetic
implicit none
private
public::cb_enabled,cb_capture_ray,cb_compute,cb_frequency_end,cb_finish
type material_ray
 real(ldp),allocatable::chi(:),eta(:)
 real(ldp)::freq=0,alpha=0
end type
type(material_ray),save::material(500)
integer,save::counted=0,expected_count=0,original_n=0
real(ldp),allocatable,save::total(:,:),surface_total(:),native_total(:),work_first(:),work_last(:),identity_signed(:),identity_L1(:),original_r(:)
real(ldp),save::first_frequency=0,last_frequency=huge(1._ldp),maximum_endpoint=0,maximum_identity=0
contains

logical function cb_enabled()
character(len=16) value
call get_environment_variable('LUMINA_FORMAL_CELL_BUDGET',value)
if(value/=' '.and.value/='1')error stop 'cell budget mode'
cb_enabled=value=='1'
end function

subroutine cb_capture_ray(ip,n,chi,eta,freq,alpha)
integer,intent(in)::ip,n
real(ldp),intent(in)::chi(n),eta(n),freq,alpha
if(.not.cb_enabled())return
if(ip<1.or.ip>500.or.n/=ray(ip)%nz)error stop 'cell material shape'
if(allocated(material(ip)%chi))then
 if(size(material(ip)%chi)/=n)deallocate(material(ip)%chi,material(ip)%eta)
end if
if(.not.allocated(material(ip)%chi))allocate(material(ip)%chi(n),material(ip)%eta(n))
material(ip)%chi=chi;material(ip)%eta=eta;material(ip)%freq=freq;material(ip)%alpha=alpha
end subroutine

subroutine cb_ray(ip,r,v,nd,nc,freq,order,volume,surface,endpoint_error)
integer,intent(in)::ip,nd,nc,order
real(ldp),intent(in)::r(nd),v(nd),freq
real(ldp),intent(out)::volume(5,nd-1),surface(nd),endpoint_error
real(ldp),allocatable::beta(:),dp(:),dm(:),effective(:),source(:),depth(:)
real(ldp)::fraction,gamma,length,tau,control,values(5),result,actual,scale,midpoint
integer n,k,i,cell,dir,up,down,far,j,first
n=ray(ip)%nz
if(.not.allocated(material(ip)%chi))error stop 'missing actual ray material'
if(material(ip)%freq/=freq)error stop 'stale ray material'
first=ray(ip)%lnk(1)
if(first<1.or.ray(ip)%r_ray(first)/=r(1))error stop 'missing physical outer boundary'
allocate(beta(n),dp(n),dm(n),effective(n),source(n),depth(n-1))
volume=0;surface=0;endpoint_error=0;cell=1;beta=0;dp=0;dm=0
! thick-outer 추가 영역의 source/blue는 보존하되 물리 R(1) 밖의 부피는 세지 않는다.
do k=first,n
 do while(cell<nd-1)
  if(ray(ip)%r_ray(k)>=r(cell+1))exit
  cell=cell+1
 end do
 fraction=(r(cell)-ray(ip)%r_ray(k))/(r(cell)-r(cell+1))
 beta(k)=((1-fraction)*v(cell)+fraction*v(cell+1))/299792.458_ldp
 gamma=1/sqrt(1-beta(k)*beta(k))
 dp(k)=1/(gamma*(1+beta(k)*ray(ip)%mu_p(k)))
 dm(k)=1/(gamma*(1+beta(k)*ray(ip)%mu_m(k)))
end do
do i=1,nd
 j=ray(ip)%lnk(i)
 if(j>0)surface(i)=ray(ip)%i_p(j)/dp(j)**3-ray(ip)%i_m(j)/dm(j)**3
end do
do dir=1,2
 if(dir==1)then
  effective=material(ip)%chi+(material(ip)%alpha+3)*ray(ip)%b_m
  source=(material(ip)%eta+material(ip)%alpha*ray(ip)%b_m*ray(ip)%i_m_prev)/effective
  depth=.5_ldp*(ray(ip)%s_m(2:n)-ray(ip)%s_m(1:n-1))*(effective(1:n-1)+effective(2:n))
 else
  effective=material(ip)%chi+(material(ip)%alpha+3)*ray(ip)%b_p
  source=(material(ip)%eta+material(ip)%alpha*ray(ip)%b_p*ray(ip)%i_p_prev)/effective
  depth=.5_ldp*(ray(ip)%s_p(1:n-1)-ray(ip)%s_p(2:n))*(effective(1:n-1)+effective(2:n))
 end if
 cell=1
 do k=first,n-1
  midpoint=.5_ldp*(ray(ip)%r_ray(k)+ray(ip)%r_ray(k+1))
  do while(cell<nd-1)
   if(midpoint>=r(cell+1))exit
   cell=cell+1
  end do
  tau=depth(k)
  if(dir==1)then
   up=k;down=k+1;far=k+2;length=ray(ip)%s_m(down)-ray(ip)%s_m(up)
   control=.5_ldp*(source(up)+source(down))
   if(down<n)control=sc_control(source(up),source(down),source(far),tau,depth(k+1))
   call sc_physical_cell_budget(ray(ip)%i_m(up),source(up),control,source(down), &
    material(ip)%chi([up,down]),material(ip)%eta([up,down]),ray(ip)%b_m([up,down]), &
    ray(ip)%i_m_prev([up,down]),dm([up,down]),length,material(ip)%alpha,order,values)
   result=sc_cell_intensity_at(ray(ip)%i_m(up),source(up),control,source(down),tau,tau)
   actual=ray(ip)%i_m(down)
   scale=max(ray(ip)%i_m(up),actual,source(up)*min(tau,1._ldp),source(down)*min(tau,1._ldp))
  else
   up=k+1;down=k;far=k-1;length=ray(ip)%s_p(down)-ray(ip)%s_p(up)
   control=.5_ldp*(source(up)+source(down))
   if(down>=3)control=sc_control(source(up),source(down),source(far),tau,depth(k-1))
   call sc_physical_cell_budget(ray(ip)%i_p(up),source(up),control,source(down), &
    material(ip)%chi([up,down]),material(ip)%eta([up,down]),ray(ip)%b_p([up,down]), &
    ray(ip)%i_p_prev([up,down]),dp([up,down]),length,material(ip)%alpha,order,values)
   result=sc_cell_intensity_at(ray(ip)%i_p(up),source(up),control,source(down),tau,tau)
   actual=ray(ip)%i_p(down)
   scale=max(ray(ip)%i_p(up),actual,source(up)*min(tau,1._ldp),source(down)*min(tau,1._ldp))
  end if
  volume(:,cell)=volume(:,cell)+values
  if(scale>0)endpoint_error=max(endpoint_error,abs(result-actual)/scale)
 end do
end do
end subroutine

subroutine cb_compute(r,v,p,nd,nc,np,freq,order,budget,flux,endpoint_error)
integer,intent(in)::nd,nc,np,order
real(ldp),intent(in)::r(nd),v(nd),p(np),freq
real(ldp),intent(out)::budget(5,nd-1),flux(nd),endpoint_error
real(ldp),allocatable::volumes(:,:,:),surfaces(:,:),errors(:)
real(ldp) weight
integer ip
if(.not.cb_enabled())error stop 'cell observer disabled'
if(p(1)/=0.or.p(np)/=r(1).or.any(p(2:np)<=p(1:np-1)))error stop 'cell observer p coverage'
allocate(volumes(5,nd-1,np),surfaces(nd,np),errors(np));volumes=0;surfaces=0;errors=0
!$omp parallel do schedule(dynamic)
do ip=1,np-1
 call cb_ray(ip,r,v,nd,nc,freq,order,volumes(:,:,ip),surfaces(:,ip),errors(ip))
end do
!$omp end parallel do
budget=0;flux=0
! comoving 공간/각 측도: r^2 dr dmu/2 = p dp ds/(2*D^3).
do ip=1,np-1
 weight=(p(ip+1)-p(ip))*(p(ip+1)+p(ip))/8
 budget=budget+weight*(volumes(:,:,ip)+volumes(:,:,ip+1))
 flux=flux+weight*(surfaces(:,ip)+surfaces(:,ip+1))
end do
endpoint_error=maxval(errors)
end subroutine

subroutine cb_frequency_end(index,expected,freq,weight,r,v,p,nd,nc,np,j,h)
integer,intent(in)::index,expected,nd,nc,np
real(ldp),intent(in)::freq,weight,r(nd),v(nd),p(np),j(nd),h(nd)
real(ldp)::budget(5,nd-1),flux(nd),native(nd),delta(nd-1),remainder(nd-1),endpoint_error,scale
if(.not.cb_enabled())return
if(index/=counted+1.or.freq>=last_frequency.or.weight<=0)error stop 'cell frequency coverage'
if(counted==0)then
 original_n=nd;expected_count=expected
 allocate(total(5,nd-1),surface_total(nd),native_total(nd),work_first(nd-1),work_last(nd-1), &
  identity_signed(nd-1),identity_L1(nd-1),original_r(nd))
 total=0;surface_total=0;native_total=0;identity_signed=0;identity_L1=0;original_r=r
end if
if(nd/=original_n.or.expected/=expected_count.or.any(original_r/=r))error stop 'cell shape changed'
call cb_compute(r,v,p,nd,nc,np,freq,8,budget,flux,endpoint_error)
native=r*r/sqrt(1-(v/299792.458_ldp)**2)*(h+(v/299792.458_ldp)*j)
delta=flux(1:nd-1)-flux(2:nd)
remainder=delta-budget(1,:)-budget(3,:)-budget(4,:)-budget(5,:)
scale=max(sum(abs(delta)),sum(abs(budget(1,:))),sum(abs(budget(3,:))),sum(abs(budget(4,:))),sum(abs(budget(5,:))))
if(scale>0)maximum_identity=max(maximum_identity,sum(abs(remainder))/scale)
maximum_endpoint=max(maximum_endpoint,endpoint_error)
total=total+weight*budget;surface_total=surface_total+weight*flux;native_total=native_total+weight*native
identity_signed=identity_signed+weight*remainder;identity_L1=identity_L1+weight*abs(remainder)
if(counted==0)then
 first_frequency=freq;work_first=freq*1.e15_ldp*budget(2,:)
end if
last_frequency=freq;work_last=freq*1.e15_ldp*budget(2,:);counted=counted+1
end subroutine

subroutine cb_finish()
integer lu
character(len=16) tag
if(.not.cb_enabled())return
if(counted==0.or.counted/=expected_count)error stop 'incomplete cell frequency integral'
open(newunit=lu,file='FORMAL_CELL_ENERGY.bin',status='new',access='stream',form='unformatted')
tag='RAYENERGYV1'
write(lu)tag,original_n,counted,expected_count,8
write(lu)original_r,first_frequency,last_frequency,maximum_endpoint,maximum_identity
write(lu)total,surface_total,native_total,work_first,work_last,identity_signed,identity_L1
close(lu)
end subroutine
end module
