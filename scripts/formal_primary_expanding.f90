! I_nu/nu^3 불변 source의 팽창 구형 제조해. 생산 whole/앵커 자료가 아니다.
program formal_primary_expanding
use lumina_formal_primary
use lumina_formal_diag
implicit none
integer n,nf,nc,np,i,k,lu
character(len=80) arg
real(ldp),allocatable::r(:),v(:),sigma(:),p(:),chi(:),eta(:),scat(:),j(:),f(:),iplus(:),vdop(:)
real(ldp) freq,dlog,inner_h,outer_h
call get_command_argument(1,arg);read(arg,*)n
call get_command_argument(2,arg);read(arg,*)nf
call get_command_argument(3,arg);read(arg,*)nc
np=n+nc
if(n<7.or.n>129.or.nf<32.or.nf>4096.or.nc<1.or.np>500.or..not.fp_enabled())stop 85
allocate(r(n),v(n),sigma(n),p(np),chi(n),eta(n),scat(n),j(n),f(n),iplus(np),vdop(n))
do i=1,n
 r(i)=1.5_ldp+.5_ldp*cos(acos(-1._ldp)*real(i-1,ldp)/(n-1))
end do
do i=1,nc
 p(i)=sqrt(1-(1-real(i-1,ldp)/nc)**2)
end do
do i=1,n
 p(nc+i)=r(n-i+1)
end do
v=.025_ldp*299792.458_ldp*r;sigma=0;chi=1;scat=0;vdop=3000;dlog=1._ldp/nf
open(newunit=lu,file='EXPANDING_TRANSFER.bin',status='new',access='stream',form='unformatted')
write(lu)n,nf+1,nc,np
do k=0,nf
 freq=exp(1._ldp-real(k,ldp)/nf);eta=freq**3
 call cmf_formal_rel_v4(eta,chi,scat,v,sigma,r,p,j,f,inner_h,outer_h,iplus,freq,dlog,freq**3,0._ldp, &
  'DIFFUSION',.false.,vdop,1._ldp,1.5_ldp,'LOGMON',k==0,.true.,nc,np,n)
 if(k==0)write(lu)r,v,sigma,gam_rel_store,p
 write(lu)freq,jnu_store,hnu_store,knu_store,nnu_store
end do
close(lu)
open(newunit=lu,file='EXPANDING_RADIAL_COORDINATE.dat',status='new')
do i=1,n
 write(lu,'(2ES25.16E3)')r(i),ray(1)%s_p(ray(1)%lnk(i))
end do
close(lu)
write(6,'(A,3I8)')'EXPANDING_MANUFACTURED_NOT_ANCHOR ',n,nf,nc
end program
