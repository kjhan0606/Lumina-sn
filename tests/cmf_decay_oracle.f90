! 실제 CMFGEN DO_SPECIES_DECAYS_V2/DO_LEV_POP_DECAYS 호출용 합성 상태.
! 모듈은 필요한 저장소만 제공한다. 광도/벡터 출력 stub은 물리 배열을 바꾸지 않는다.
module set_kind_module
  integer,parameter::ldp=kind(1d0)
end module
module nuc_iso_mod
  use set_kind_module
  integer,parameter::num_isotopes=3,num_parents=3,num_decay_paths=2
  logical::do_rad_decays=.true.
  real(ldp)::nuc_iso_mod_delta_t,radioactive_decay_energy(2),kinetic_decay_energy(2)
  type isotope
    real(ldp)::old_pop(2),old_pop_decay(2),num_decays(2),decay_lum(2),mass
    integer::lnk_to_par
    character(10)::species
  end type
  type parent
    real(ldp)::old_pop(2),old_pop_decay(2)
    integer::ispec
    logical::decay_chain_available=.true.
  end type
  type reaction
    real(ldp)::decay_const,energy_per_decay,kinetic_per_decay
    integer::lnk_to_iso,daughter_lnk_to_iso,baryon_number=56
    character::sequence
    character(10)::species,daughter
  end type
  type(isotope)::iso(3)
  type(parent)::par(3)
  type(reaction)::nuc(2)
end module
module mod_cmfgen
  integer::species_beg_id(3)=[1,3,5],species_end_id(3)=[2,4,6]
  logical::species_pres(3)=.true.
  type ion
    integer::eqxzv,nxzv=1
  end type
  type(ion)::atm(6)
end module
integer function error_lu()
  error_lu=6
end function
subroutine get_decay_lum_v1(total,values,nd)
  use set_kind_module
  integer::nd
  real(ldp)::total,values(nd)
  total=sum(values)
end subroutine
subroutine writv_v2(values,nd,ncol,label,lu)
  use set_kind_module
  integer::nd,ncol,lu
  real(ldp)::values(nd)
  character(*)::label
end subroutine
program oracle
  use nuc_iso_mod
  use mod_cmfgen
  real(ldp)::pops(6,2),fraction(3)=[.7d0,.2d0,.4d0],dt
  integer::i,s
  iso(1)%species='NICK';iso(2)%species='COB';iso(3)%species='IRON'
  iso(1)%old_pop=[100d0,200d0];iso(2)%old_pop=[30d0,40d0];iso(3)%old_pop=[20d0,30d0]
  do i=1,3
    iso(i)%lnk_to_par=i;iso(i)%mass=56d0;par(i)%ispec=i
    par(i)%old_pop=iso(i)%old_pop
    pops(2*i-1,:)=fraction(i)*iso(i)%old_pop
    pops(2*i,:)=iso(i)%old_pop-pops(2*i-1,:)
  enddo
  do i=1,6
    atm(i)%eqxzv=i
  enddo
  nuc(1)%lnk_to_iso=1;nuc(1)%daughter_lnk_to_iso=2
  nuc(2)%lnk_to_iso=2;nuc(2)%daughter_lnk_to_iso=3
  nuc(1)%species='NICK';nuc(1)%daughter='COB';nuc(1)%sequence='F'
  nuc(2)%species='COB';nuc(2)%daughter='IRON';nuc(2)%sequence='S'
  nuc(1)%decay_const=log(2d0)/(6.0749d0*86400d0)
  nuc(2)%decay_const=log(2d0)/(77.233d0*86400d0)
  ! 합성 에너지 가중치: nuclear 입력/공개 SN의 절대 침적 모형을 검증하지 않는다.
  nuc(1)%energy_per_decay=2d0;nuc(2)%energy_per_decay=4d0
  nuc(1)%kinetic_per_decay=0d0;nuc(2)%kinetic_per_decay=.2d0
  dt=8d0*86400d0
  call do_species_decays_v2(.false.,dt,2)
  call do_lev_pop_decays(pops,2,6)
  open(21,file='cmf_decay_oracle.dat',status='new')
  do s=1,2
    write(21,'(13ES26.17)') (iso(i)%old_pop_decay(s),i=1,3), &
      iso(1)%num_decays(s),iso(2)%num_decays(s), &
      radioactive_decay_energy(s),kinetic_decay_energy(s),(pops(i,s),i=1,6)
  enddo
  close(21)
end program
