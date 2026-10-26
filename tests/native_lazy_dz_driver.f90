! 실제 native helper/행 해제/강제 회수/새 sweep reset 블록의 작은 수명주기 시험.
! 복사 수송·SE·BA 산술의 대용물이 아니다.
program lifecycle
  use set_kind_module
  use line_mod
  implicit none
  integer, parameter :: NM=30, ND=4, NUM_BNDS=2, MAX_SIM=12
  real(kind=LDP) :: dZ(NM,NUM_BNDS,ND,MAX_SIM)
  real(kind=LDP) :: TX(ND,ND,NM), TVX(ND-1,ND,NM)
  integer :: unit, phase=0, NUM_OF_WEAK_LINES
  integer :: line_loc(MAX_SIM)
  allocate(line_storage_used(MAX_SIM), weak_line(MAX_SIM))
  allocate(low_pointer(MAX_SIM), up_pointer(MAX_SIM), var_in_use_cnt(NM), var_lev_id(NM))
  allocate(imp_trans_vec(NM), sim_line_pointer(MAX_SIM))
  allocate(resonance_zone(MAX_SIM), end_res_zone(MAX_SIM))
  dZ=-12345.0_LDP
  TX=123.0_LDP
  TVX=456.0_LDP
  open(newunit=unit,file='projection.bin',access='stream',form='unformatted',status='new')
  call sweep_reset()
  call add_line(1,.false.,7)
  call add_line(2,.true.,0)
  call add_line(3,.false.,9)
  call fill_live()
  call observe()
  ! resonance 밖이라도 사용 중인 strong 컬럼은 지워야 한다.
  resonance_zone=.false.
  resonance_zone(1)=.true.
  call ZERO_dZ(dZ,7,NM,NUM_BNDS*ND,MAX_SIM)
  call observe()
  call retire_line(1)
  call observe()
  call add_line(4,.false.,11)
  call fill_live()
  call add_line(1,.true.,0)
  call observe()
  call retire_line(1)
  ! 이전 weak/미사용 컬럼의 poison을 strong으로 쓰기 전에 전체 reset.
  call add_line(1,.false.,7)
  if(any(dZ(:,:,:,1)/=0.0_LDP)) error stop '새 strong 컬럼 미초기화'
  call observe()
  call fill_live()
  call force_reclaim(3)
  call add_line(5,.false.,9)
  call observe()
  ! 슬롯 부족 회수 뒤 같은 컬럼 재진입도 확인한다.
  call add_line(3,.false.,13)
  if(any(dZ(:,:,:,3)/=0.0_LDP)) error stop '회수 컬럼 미초기화'
  call observe()
  call fill_live()
  call sweep_reset()
  call add_line(6,.false.,7)
  if(any(dZ(:,:,:,6)/=0.0_LDP)) error stop '새 sweep 컬럼 미초기화'
  call observe()
  close(unit)
  print *, 'LIFECYCLE_PASS',phase
contains
  subroutine add_line(slot,weak,row)
    integer, intent(in) :: slot,row
    logical, intent(in) :: weak
    integer :: SIM_INDX
    if(line_storage_used(slot)) error stop '사용 중인 슬롯 덮어쓰기'
    SIM_INDX=slot
    line_storage_used(slot)=.true.
    weak_line(slot)=weak
    sim_line_pointer(slot)=slot
    line_loc(slot)=slot
    if(weak) return
    low_pointer(slot)=row
    up_pointer(slot)=row+1
    var_in_use_cnt(row:row+1)=1
    var_lev_id(row:row+1)=[slot,slot+20]
    call ZERO_dZ(dZ,row,NM,NUM_BNDS*ND,MAX_SIM)
    call ZERO_dZ(dZ,row+1,NM,NUM_BNDS*ND,MAX_SIM)
! NEW_COLUMN_RESET
  end subroutine
  subroutine retire_line(slot)
    integer, intent(in) :: slot
    integer :: NL,NUP
    line_storage_used(slot)=.false.
    sim_line_pointer(slot)=0
    if(weak_line(slot)) return
    NL=low_pointer(slot)
    var_in_use_cnt(NL)=var_in_use_cnt(NL)-1
! RETIRE_LOW_BLOCK
    low_pointer(slot)=0
    NUP=up_pointer(slot)
    var_in_use_cnt(NUP)=var_in_use_cnt(NUP)-1
! RETIRE_UP_BLOCK
    up_pointer(slot)=0
  end subroutine
  subroutine force_reclaim(slot)
    integer, intent(in) :: slot
    integer :: I,NL,NUP,FIRST_LINE
    FIRST_LINE=slot
! FORCE_RECLAIM_BLOCK
  end subroutine
  subroutine sweep_reset()
! SWEEP_RESET_BLOCK
  end subroutine
  subroutine fill_live()
    integer :: i,j,k,l
    do l=1,MAX_SIM
      if(line_storage_used(l).and..not.weak_line(l)) then
        do k=1,ND
          do j=1,NUM_BNDS
            do i=1,NM
              dZ(i,j,k,l)=real(10000*l+100*k+10*j+i,LDP)
            end do
          end do
        end do
      end if
    end do
  end subroutine
  subroutine observe()
    integer :: l
    real(kind=LDP) :: accumulator
    phase=phase+1
    write(unit) phase
    accumulator=0.0_LDP
    do l=1,MAX_SIM
      if(line_storage_used(l).and..not.weak_line(l)) then
        write(unit) l,dZ(:,:,:,l)
        accumulator=accumulator+sum(dZ(:,:,:,l))
      end if
    end do
    write(unit) accumulator
  end subroutine
end program
