! 실제 READ_BA_DATA_V3와 원 cmfgen_sub의 초기화 블록만 쓰는 소형 재현.
program native_restart_ba
  use set_kind_module
  use steq_data_mod
  implicit none
  logical :: compute_ba,compute_bardin,chk,success,newmod,lambda_iteration
  logical :: fixed_t,wrbamat,wrbamat_rdin,available
  integer :: lu_ba,nion,num_bnds,nd,nlbegin,n_its_to_fix_ba,values(4),i
  character(len=20)::arg
  do i=1,4
    call get_command_argument(i,arg);read(arg,*)values(i)
  end do
  compute_bardin=values(1)==1;available=values(2)==1
  newmod=values(3)==1;compute_ba=values(4)==1
  success=.true.;lambda_iteration=.false.;fixed_t=.true.
  wrbamat=.false.;wrbamat_rdin=.false.
  lu_ba=33;nion=1;num_bnds=1;nd=1;nlbegin=0;n_its_to_fix_ba=0
  allocate(se(1)%ba(1,1,1,1),ba_ed(1,1,1),ba_t(1,1,1),ba_t_ehb(1,1,1))
  se(1)%ba=7._LDP;ba_ed=7._LDP;ba_t=7._LDP;ba_t_ehb=7._LDP
  if(available)then
    open(unit=10,file='BAMATPNT',status='new')
    write(10,*) .true.
    write(10,*) .false.
    write(10,*) .false.
    write(10,*) 1
    write(10,*) 1
    write(10,*) 1
    close(10)
    open(unit=10,file='BAMAT',status='new',form='unformatted')
    write(10)se(1)%ba;write(10)ba_ed;write(10)ba_t;write(10)ba_t_ehb
    close(10)
  end if
  se(1)%ba=999._LDP
! RESTART_BLOCK
  write(*,'(A,5L2,F10.1)')'STATE ',compute_ba,success,chk,fixed_t,wrbamat,se(1)%ba(1,1,1,1)
end program
