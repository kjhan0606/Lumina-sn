! 단일 실패 packet에서 실제 native source 분기만 관측한다. 수송값은 수정하지 않는다.
module lumina_formal_source_branch
use lumina_formal_primary
implicit none
integer,save::branch_lu=0
logical,save::branch_open=.false.
contains
subroutine branch_note(ip,iz,direction,t1,s0,s1,s2,d0,d1,a,b,c)
integer ip,iz,direction,link
real(ldp)t1,s0,s1,s2,d0,d1,a,b,c
logical near
if(.not.fp_in_call)return
near=.false.
if(ip<=size(jqw_p,2)+1-50)then
 link=ray(ip)%lnk(50);near=abs(iz-link)<=1
end if
if(.not.near.and.t1>=0)return
if(.not.branch_open)then
 open(newunit=branch_lu,file='ACTUAL_SOURCE_BRANCHES.bin',status='new',access='stream',form='unformatted')
 branch_open=.true.
end if
write(branch_lu)fp_count,ip,iz,direction,t1,s0,s1,s2,d0,d1,a,b,c
flush(branch_lu)
end subroutine
end module
