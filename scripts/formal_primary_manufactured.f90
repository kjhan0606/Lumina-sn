! native angular solver의 작은 독립 제조해/두꺼운 산란장 시험.
program formal_primary_manufactured
use lumina_formal_primary
use lumina_formal_diag
implicit none
integer n,nc,np,i,ip,lu,count,nfreq,k,iz
character(len=100) mode,arg,mapping
real(ldp),allocatable::r(:),v(:),sigma(:),p(:),chi(:),eta(:),scat(:),j(:),jnew(:),f(:),iplus(:),vdop(:)
real(ldp),allocatable::xray(:),eray(:),previous(:)
real(ldp) inner_h,outer_h,dbb,bplanck,freq,dlog,exact
logical inaccurate
call get_command_argument(1,mode);call get_command_argument(2,arg);read(arg,*)n
if(n<6.or.n>257.or..not.fp_enabled())stop 85
if(mode=='redshift')then
 call get_command_argument(3,arg);read(arg,*)nfreq
 allocate(xray(n),eray(n),previous(n))
 ray(1)%nz=n
 allocate(ray(1)%i_p(n),ray(1)%i_m(n),ray(1)%i_p_prev(n),ray(1)%i_m_prev(n), &
  ray(1)%s_p(n),ray(1)%s_m(n),ray(1)%b_p(n),ray(1)%b_m(n),ray(1)%mu_p(n))
 do i=1,n
  ray(1)%s_p(i)=real(n-i,ldp)/(n-1);ray(1)%s_m(i)=1-ray(1)%s_p(i)
 end do
 ray(1)%b_p=.25_ldp;ray(1)%b_m=.25_ldp;ray(1)%mu_p=1
 xray=0;eray=0;ray(1)%i_p_prev=exp(3._ldp);ray(1)%i_m_prev=0
 dlog=1._ldp/nfreq
 do k=1,nfreq
  freq=exp(1._ldp-real(k,ldp)/nfreq);previous=ray(1)%i_p_prev
  call solve_cmf_formal_v2(xray,eray,1,freq,1._ldp/dlog,'DIFFUSION',freq**3,0._ldp,n,1,1)
  ray(1)%i_p_prev=ray(1)%i_p
 end do
 open(newunit=lu,file='REDSHIFT_FIELD.dat',status='new')
 do i=1,n
  write(lu,'(4ES25.16E3)')ray(1)%s_p(i),ray(1)%i_p(i),previous(i),ray(1)%b_p(i)
 end do
 close(lu)
 write(6,'(A,2I8,ES25.16E3)')'NATIVE_REDSHIFT_MANUFACTURED ',n,nfreq,maxval(abs(ray(1)%i_p-1))
 stop
end if
nc=min(n,129)
call get_command_argument(3,arg)
if(len_trim(arg)>0)read(arg,*)nc
call get_command_argument(4,mapping)
if(mapping/=' '.and.mapping/='cosine')stop 85
np=nc+n
if(nc<1.or.np>500)stop 85
allocate(r(n),v(n),sigma(n),p(np),chi(n),eta(n),scat(n),j(n),jnew(n),f(n),iplus(np),vdop(n))
do i=1,n
 r(i)=2._ldp-real(i-1,ldp)/(n-1)
 if(mapping=='cosine')r(i)=1.5_ldp+.5_ldp*cos(acos(-1._ldp)*real(i-1,ldp)/(n-1))
end do
do i=1,nc
 p(i)=sqrt(1-(1-real(i-1,ldp)/nc)**2)
end do
do i=1,n
 p(nc+i)=r(n-i+1)
end do
v=0;sigma=0;vdop=3000;chi=1;eta=1;scat=0;j=0;bplanck=1;dbb=0
if(mode=='zero')then
 eta=0;bplanck=0
else if(mode=='thermal')then
 chi=100;eta=100
else if(mode=='scatter')then
 chi=10;scat=9;j=1
else if(mode=='thick')then
 chi=100;scat=99;j=1
else if(mode=='sharp')then
 eta=1+.2_ldp*tanh((r-1.5_ldp)/.03_ldp)
end if
freq=1;dlog=.01_ldp
open(newunit=lu,file='MANUFACTURED_ITERATIONS.dat',status='new')
do count=0,19
 call fp_start(freq,count,n,j,eta,scat,.true.,.true.,.true.)
 call cmf_formal_rel_v4(eta+scat*j,chi,scat,v,sigma,r,p,jnew,f,inner_h,outer_h,iplus,freq,dlog,bplanck,dbb, &
  'DIFFUSION',.false.,vdop,1._ldp,1.5_ldp,'LOGMON',count==0,count==0,nc,np,n)
 call fp_record(n,np,jnew,f,.false.)
 j=jnew
 write(lu,'(I5,5ES25.16E3)')count,fp_j_residual,fp_source_residual,fp_h_residual,fp_f_change,fp_support_min
 flush(lu)
 inaccurate=.false.;call fp_end_iteration(count+1,fp_f_change,1.e-4_ldp,inaccurate)
 if(.not.inaccurate)exit
 call fp_next_source(n,jnew,j)
end do
close(lu)
open(newunit=lu,file='MANUFACTURED_NODES.dat',status='new')
do i=1,n
 write(lu,'(I6,8ES25.16E3)')i,r(i),j(i),hnu_store(i),knu_store(i),nnu_store(i),eta(i),chi(i),scat(i)
end do
close(lu)
open(newunit=lu,file='MANUFACTURED_OUTER.dat',status='new')
do ip=1,np-1
 write(lu,'(I6,2ES25.16E3)')ip,p(ip),ray(ip)%i_p(1)
end do
close(lu)
write(6,'(A,A,I6)')'NATIVE_PRIMARY_MANUFACTURED ',trim(mode),fp_calls
end program
