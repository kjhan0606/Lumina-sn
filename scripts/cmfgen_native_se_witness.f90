! 관찰용 private build 전용이다. 원래 native 소스/현재 실행은 변경하지 않는다.
module lumina_native_se_witness
use iso_fortran_env, only: int32
use ieee_arithmetic
use ieee_exceptions
use set_kind_module, only: ldp
use steq_data_mod, only: se,steq_ed
use mod_cmfgen, only: num_species,species_beg_id,species_end_id,pop_species
use lumina_se_collision_bound
implicit none
logical,save::enabled=.false.
integer,save::nr=0,ndepth=0
real(ldp),allocatable,save::ci(:,:),co(:,:)
logical,allocatable,save::coll_ok(:,:)
integer(int32),allocatable,save::coll_count(:,:),seen(:,:)
contains

subroutine reset(nt,nd,nion)
integer,intent(in)::nt,nd,nion
character(8)::setting
integer::status
enabled=.false.
call get_environment_variable('LUMINA_NATIVE_SE_WITNESS',setting,status=status)
if (status/=0.or.trim(setting)/='1') return
nr=nt-2;ndepth=nd
if (nr<1.or.nd<1.or.nion<1.or.storage_size(0.0_ldp)/=64) return
if (allocated(ci)) deallocate(ci,co,coll_ok,coll_count,seen)
allocate(ci(nr,nd),co(nr,nd),coll_ok(nr,nd),coll_count(nr,nd),seen(nion,nd))
ci=0;co=0;coll_ok=.true.;coll_count=0;seen=0
enabled=.true.
end subroutine

subroutine collect(id,n,nd,k,cnm,hn,hnst,ne)
integer,intent(in)::id,n,nd,k
real(ldp),intent(in)::cnm(n,n),hn(n),hnst(n),ne
real(ldp)::inc(n+1),out(n+1),value
logical::ok,flags(size(ieee_all))
integer::j,row
if (.not.enabled) return
if (nd/=ndepth.or.id<1.or.id>size(seen,1).or.k<1.or.k>ndepth) then
  coll_ok=.false.;return
end if
call ieee_get_flag(ieee_all,flags)
seen(id,k)=seen(id,k)+1
call collision_lower(n,cnm,hn,hnst,ne,inc,out,ok)
if (seen(id,k)/=1.or.n+1>=se(id)%n_se) ok=.false.
do j=1,min(n+1,se(id)%n_se-1)
  row=se(id)%eq_in_ba(j)
  if (row<1.or.row>nr) then
    coll_ok=.false.;cycle
  end if
  coll_ok(row,k)=coll_ok(row,k).and.ok
  if (.not.coll_ok(row,k)) cycle
  value=lower_add(ci(row,k),inc(j),ok);ci(row,k)=value
  value=lower_add(co(row,k),out(j),ok);co(row,k)=value
  coll_ok(row,k)=ok
  coll_count(row,k)=coll_count(row,k)+1
end do
call ieee_set_flag(ieee_all,flags)
end subroutine

subroutine write_state(pops,nt,nd,nion,counter)
integer,intent(in)::nt,nd,nion,counter
real(ldp),intent(in)::pops(nt,nd)
real(ldp)::raw(nt-2,nd),lo(nt-2,nd),hi(nt-2,nd)
real(ldp)::nc(num_species,nd),nc_lo(num_species,nd),nc_hi(num_species,nd)
integer(int32)::raw_count(nt-2),raw_valid(nt-2,nd),coll_valid(nt-2,nd)
integer(int32)::nc_valid(num_species,nd)
logical::ok,found,flags(size(ieee_all))
integer::s,id,j,k,row,lu,ios
character(64)::filename
if (.not.enabled) return
if (nr/=nt-2.or.ndepth/=nd.or.size(seen,1)/=nion) return
call ieee_get_flag(ieee_all,flags)
raw=0;lo=0;hi=0;raw_count=0;raw_valid=1
nc=0;nc_lo=0;nc_hi=0;nc_valid=0
do s=1,num_species
  found=.false.
  do id=species_beg_id(s),species_end_id(s)-1
    if (.not.se(id)%xzv_pres) cycle
    if (.not.found) then
      nc_valid(s,:)=1;found=.true.
    end if
    do j=1,se(id)%n_se-1
      row=se(id)%eq_in_ba(j)
      if (row<1.or.row>nr) then
        raw_valid=0;cycle
      end if
      raw_count(row)=raw_count(row)+1
      do k=1,nd
        ok=raw_valid(row,k)==1
        call raw_interval_add(lo(row,k),hi(row,k),se(id)%steq(j,k),ok)
        raw(row,k)=raw(row,k)+se(id)%steq(j,k)
        raw_valid(row,k)=merge(1,0,ok.and.ieee_is_finite(raw(row,k)))
        if (seen(id,k)/=1) coll_ok(row,k)=.false.
      end do
    end do
    do k=1,nd
      ok=nc_valid(s,k)==1
      call raw_interval_add(nc_lo(s,k),nc_hi(s,k),se(id)%steq(se(id)%n_se,k),ok)
      nc(s,k)=nc(s,k)+se(id)%steq(se(id)%n_se,k)
      nc_valid(s,k)=merge(1,0,ok.and.ieee_is_finite(nc(s,k)))
    end do
  end do
end do
do row=1,nr
  if (raw_count(row)==0) raw_valid(row,:)=0
end do
coll_valid=merge(1,0,coll_ok.and.coll_count>0)
write(filename,'(A,I6.6,A)')'NATIVE_SE_BOUND_',counter,'.bin'
open(newunit=lu,file=trim(filename),status='new',access='stream',form='unformatted',iostat=ios)
if (ios==0) then
  ! counter는 갱신 번호다. 아래 인구/잔차는 갱신 전 물질의 평가값이다.
  write(lu,iostat=ios)'NATIVESEBOUNDV1 ', &
    int([1,64,nt-2,nd,nt,nion,counter,counter-1,num_species],int32), &
    pops,raw,lo,hi,ci,co,raw_count,raw_valid,coll_valid,coll_count, &
    nc,nc_lo,nc_hi,transpose(pop_species),nc_valid,steq_ed
  close(lu)
end if
if (ios/=0) write(6,*)'LUMINA_SE_WITNESS output unavailable; no residual certificate'
call ieee_set_flag(ieee_all,flags)
end subroutine
end module

subroutine lumina_native_se_reset(nt,nd,nion)
use lumina_native_se_witness, only: reset
integer,intent(in)::nt,nd,nion
call reset(nt,nd,nion)
end subroutine

subroutine lumina_native_se_collect(id,n,nd,k,cnm,hn,hnst,ne)
use set_kind_module, only: ldp
use lumina_native_se_witness, only: collect
integer,intent(in)::id,n,nd,k
real(ldp),intent(in)::cnm(n,n),hn(n),hnst(n),ne
call collect(id,n,nd,k,cnm,hn,hnst,ne)
end subroutine

subroutine lumina_native_se_write(pops,nt,nd,nion,counter)
use set_kind_module, only: ldp
use lumina_native_se_witness, only: write_state
integer,intent(in)::nt,nd,nion,counter
real(ldp),intent(in)::pops(nt,nd)
call write_state(pops,nt,nd,nion,counter)
end subroutine
