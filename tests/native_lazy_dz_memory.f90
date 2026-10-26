! 큰 가상배열에서 원/수정 native helper를 별도 프로세스로 측정한다.
! 전체 poison/검사/초기화는 하지 않는다. 실제 모델 peak 예측이 아니다.
program memory_probe
  use set_kind_module
  use line_mod, only: line_storage_used,weak_line
  implicit none
  integer, parameter :: NM=16006, NX=90, M=8000
  real(kind=LDP), allocatable :: dz(:,:,:)
  integer :: l,unit,status,zero_columns
  character(len=256) :: line,mode
  call get_command_argument(1,mode)
  zero_columns=M
  if(trim(mode)=='prefix256') zero_columns=256
  if(trim(mode)/='full'.and.trim(mode)/='prefix256') error stop '메모리 시험 범위 누락'
  allocate(dz(NM,NX,M),line_storage_used(M),weak_line(M))
  print *, 'VIRTUAL_ALLOCATION_DONE',NM,NX,M
  flush(6)
  line_storage_used=.false.
  weak_line=.true.
  do l=1,4
    line_storage_used(l)=.true.
    weak_line(l)=.false.
    dz(:,:,l)=1.0_LDP
  end do
  print *, 'FOUR_STRONG_COLUMNS_READY'
  flush(6)
  call ZERO_dZ(dz,7,NM,NX,zero_columns)
  print *, 'ZEROED_COLUMNS',zero_columns
  if(any(dz(7,:,1:4)/=0.0_LDP)) error stop '사용 strong 행 초기화 실패'
  print *, 'ACTIVE_SUM',sum(dz(:,:,1:4))
  open(newunit=unit,file='/proc/self/status',status='old')
  do
    read(unit,'(a)',iostat=status) line
    if(status/=0) exit
    if(index(line,'VmHWM:')==1) print '(a)',trim(line)
  end do
  close(unit)
  open(newunit=unit,file='/proc/self/smaps_rollup',status='old',iostat=status)
  if(status==0) then
    do
      read(unit,'(a)',iostat=status) line
      if(status/=0) exit
      if(index(line,'AnonHugePages:')==1) print '(a)',trim(line)
    end do
    close(unit)
  end if
end program
