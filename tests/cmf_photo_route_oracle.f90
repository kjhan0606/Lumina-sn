! 실제 CMFGEN GET_NUM_PHOT_ROUTES_V1을 호출하는 최소 I/O 어댑터.
! 목적 상태/에너지/가중 계산은 복제하지 않고 원본 Fortran에 맡긴다.
module SET_KIND_MODULE
  implicit none
  integer, parameter :: LDP=kind(1.0d0)
end module
module PHOT_DATA_MOD
  use SET_KIND_MODULE
  implicit none
  type photo_data
    real(LDP), allocatable :: EXC_FREQ(:), SCALE_FAC(:), GION(:)
    logical, allocatable :: DO_PHOT(:,:)
    integer, allocatable :: PHOT_GRID(:), ION_LEV_ID(:)
    character(len=20), allocatable :: FINAL_ION_NAME(:)
  end type
  type(photo_data) :: PD(1)
end module

program route_oracle
  use PHOT_DATA_MOD
  implicit none
  integer :: n, present_flag, tables, routes, i
  integer :: terms(26), pairs(26)
  integer, allocatable :: sl(:)
  real(LDP), allocatable :: edge(:), g(:)
  character(len=64), allocatable :: names(:)
  read(*,*) n,present_flag
  allocate(sl(n),edge(n),g(n),names(n))
  do i=1,n
    read(*,*) names(i),g(i),edge(i),sl(i)
  enddo
  open(99,file='reference-reading.log',status='new')
  call GET_NUM_PHOT_ROUTES_V1(tables,routes,terms,pairs,26, &
    present_flag/=0,edge,g,sl,names,n,1,'TEST',20,99)
  close(99)
  write(*,*) 'COUNTS',tables,routes
  do i=1,routes
    write(*,'(A,3I6,3ES26.17)') 'ROUTE',PD(1)%PHOT_GRID(i),PD(1)%ION_LEV_ID(i),i, &
      PD(1)%EXC_FREQ(i),PD(1)%GION(i),PD(1)%SCALE_FAC(i)
  enddo
end program

integer function ERROR_LU()
  ERROR_LU=0
end function
integer function ICHRLEN(text)
  implicit none
  character(len=*), intent(in) :: text
  ICHRLEN=len_trim(text)
end function
real(kind(1.0d0)) function SPEED_OF_LIGHT()
  SPEED_OF_LIGHT=2.99792458d10
end function
real(kind(1.0d0)) function RD_FREE_VAL(text,first,last,next,label)
  implicit none
  character(len=*), intent(in) :: text,label
  integer, intent(in) :: first,last
  integer, intent(out) :: next
  integer :: ios
  read(text(first:last),*,iostat=ios) RD_FREE_VAL
  if(ios/=0) error stop 'reference header parsing failed'
  next=last+1
end function
subroutine GEN_ASCI_OPEN(unit,path,status,unused,action,zero,ios)
  implicit none
  integer, intent(in) :: unit,zero
  integer, intent(out) :: ios
  character(len=*), intent(in) :: path,status,unused,action
  open(unit,file=path,status=status,action=action,iostat=ios)
end subroutine
