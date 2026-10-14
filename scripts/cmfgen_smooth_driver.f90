! 실제 CMFGEN SM_PHOT_V3/CUT_POINTS_V3를 바꾸지 않고 호출하는 입출력 어댑터.
module set_kind_module
  implicit none
  integer, parameter :: ldp=selected_real_kind(15,307)
end module
program smooth_driver
  use set_kind_module
  implicit none
  integer, parameter :: capacity=200000
  real(ldp), allocatable :: nu(:),cross(:)
  real(ldp) :: width,spacing,accuracy
  integer :: n,i,above_flag
  allocate(nu(capacity),cross(capacity))
  read(*,*) n,width,spacing,accuracy,above_flag
  if(n<2.or.n>=capacity/2.or.width<=0.or.spacing<=0.or.accuracy<=0)error stop 'invalid smoothing input'
  do i=1,n
    read(*,*) nu(i),cross(i)
  end do
  call sm_phot_v3(nu,cross,n,capacity,width,spacing,accuracy,above_flag/=0)
  write(*,'(a,i0)') 'SMOOTHED ',n
  do i=1,n
    write(*,'(2(es25.17e3,1x))')nu(i),cross(i)
  end do
end program
