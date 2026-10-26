module set_kind_module
end module

program cmfgen_same_n_parity
  implicit none
  character(len=256) :: lower, upper
  character(len=520) :: line
  integer :: ios, separator
  logical, external :: same_n
  do
    read(*,'(A)',iostat=ios) line
    if (ios /= 0) exit
    separator=index(line,' ')
    if (separator <= 1) stop 2
    lower=trim(line(:separator-1))
    upper=trim(adjustl(line(separator+1:)))
    if (same_n(trim(lower),trim(upper))) then
      print '(A)', '1'
    else
      print '(A)', '0'
    endif
  enddo
end program

integer function ichrlen(value)
  implicit none
  character(len=*), intent(in) :: value
  ichrlen=len_trim(value)
end function

integer function error_lu()
  implicit none
  error_lu=6
end function
