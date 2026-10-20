! 실제 ray/blue와 물리 source로 적분한 구형 제조해. whole 또는 앵커가 아니다.
program formal_primary_cell_budget
use lumina_formal_primary
use lumina_formal_diag
use lumina_ray_cell_budget
implicit none
integer n,nc,np,nf,nrows,i,k,count,lu,order
character(len=80) mode,arg
real(ldp),allocatable::r(:),v(:),sigma(:),p(:),chi(:),eta(:),scat(:),j(:),jnew(:),f(:),iplus(:),vdop(:),budget(:,:),flux(:),native_flux(:),nodal_source(:),nodal_work(:)
real(ldp) freq,dlog,inner_h,outer_h,bplanck,error,previous_freq,frequency_weight
logical inaccurate,extended
call get_command_argument(1,mode);call get_command_argument(2,arg);read(arg,*)n
call get_command_argument(3,arg);read(arg,*)nc
call get_command_argument(4,arg);read(arg,*)nf
np=nc+n
if(n<7.or.n>257.or.np>500.or.nf<32.or.nf>4096.or..not.fp_enabled().or..not.cb_enabled())stop 85
allocate(r(n),v(n),sigma(n),p(np),chi(n),eta(n),scat(n),j(n),jnew(n),f(n),iplus(np),vdop(n),budget(5,n-1),flux(n),native_flux(n),nodal_source(n),nodal_work(n))
do i=1,n
 r(i)=1.5_ldp+.5_ldp*cos(acos(-1._ldp)*real(i-1,ldp)/(n-1))
end do
do i=1,nc
 p(i)=sqrt(1-(1-real(i-1,ldp)/nc)**2)
end do
do i=1,n
 p(nc+i)=r(n-i+1)
end do
v=0;sigma=0;chi=1;eta=1;scat=0;vdop=3000;j=0;bplanck=1;extended=mode=='extended'
if(mode=='thermal')then
 chi=1.e6_ldp;eta=chi;bplanck=1.01_ldp
else if(mode=='sharp')then
 eta=1+.2_ldp*tanh((r-1.5_ldp)/.03_ldp)
else if(mode=='scatter')then
 chi=100;eta=1;scat=99;j=1
else if(mode=='expanding')then
 v=.025_ldp*299792.458_ldp*r
else if(mode=='extended')then
 ! native thick-outer는 ESEC의 log 외삽을 사용한다. 실제 모델처럼 양의 산란을 준다.
 v=.003_ldp*299792.458_ldp*r
 scat=.1_ldp;eta=.9_ldp
else if(mode/='absorb'.and.mode/='extended')then
 stop 85
end if
nrows=1
if(mode=='expanding')nrows=nf+1
open(newunit=lu,file='RAY_CELL_BUDGET.bin',status='new',access='stream',form='unformatted')
write(lu)n,nc,np,nrows,r,v,p
do k=1,nrows
 freq=1;dlog=.01_ldp
 if(mode=='expanding')then
  freq=exp(1._ldp-real(k-1,ldp)/nf);dlog=1._ldp/nf
  eta=freq**3;bplanck=freq**3
 end if
 do count=0,19
  call fp_start(freq,count,n,j,eta,scat,.true.,.true.,.true.)
  call cmf_formal_rel_v4(eta+scat*j,chi,scat,v,sigma,r,p,jnew,f,inner_h,outer_h,iplus,freq,dlog,bplanck,0._ldp, &
   'DIFFUSION',extended,vdop,1._ldp,1.5_ldp,'LOGMON',k==1,count==0,nc,np,n)
  call fp_record(n,np,jnew,f,extended)
  j=jnew
  inaccurate=.false.;call fp_end_iteration(count+1,fp_f_change,1.e-4_ldp,inaccurate)
  if(.not.inaccurate)exit
  call fp_next_source(n,jnew,j)
 end do
 native_flux=r*r*gam_rel_store*(hnu_store+v/299792.458_ldp*jnu_store)
 nodal_source=r*r*(eta+scat*fp_source_j-chi*jnu_store)
 nodal_work=r*(v/299792.458_ldp)*gam_rel_store*(jnu_store-knu_store+ &
  gam_rel_store*gam_rel_store*(knu_store+(v/299792.458_ldp)*hnu_store))
 write(lu)freq,native_flux,nodal_source,nodal_work
 do order=4,8,4
  call cb_compute(r,v,p,n,nc,np,freq,order,budget,flux,error)
  write(lu)budget,flux,error
 end do
 frequency_weight=1.e15_ldp
 if(nrows>1)then
  frequency_weight=.5_ldp*freq*(exp(1._ldp/nf)-exp(-1._ldp/nf))*1.e15_ldp
  if(k==1)frequency_weight=.5_ldp*freq*(1-exp(-1._ldp/nf))*1.e15_ldp
  if(k==nrows)frequency_weight=.5_ldp*freq*(exp(1._ldp/nf)-1)*1.e15_ldp
 end if
 call cb_frequency_end(k,nrows,freq,frequency_weight,r,v,p,n,nc,np,jnu_store,hnu_store)
end do
call cb_finish()
close(lu)
write(6,'(A,A,4I8)')'RAY_CELL_MANUFACTURED_NOT_ANCHOR ',trim(mode),n,nc,nrows,fp_calls
end program
