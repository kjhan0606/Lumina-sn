"""원식을 보존하는 1회 기울기 제안을 native 90회 미수렴 경로에만 넣는다."""
from prepare_cmfgen_mom_capture import once


def accelerate_moment(source):
    source = once(source, '\tUSE LUMINA_WHOLE_MONITOR\n',
                  '\tUSE LUMINA_WHOLE_MONITOR\n\tUSE LUMINA_MOM_ACCELERATION\n')
    source = once(source, '\tREAL(KIND=LDP) MAX_ER\n', '''\tREAL(KIND=LDP) MAX_ER
\tREAL(KIND=LDP) LUM_ACC_TRI(4,ND_SM),LUM_ACC_USED(ND_SM),LUM_ACC_ALPHA(ND_SM),LUM_ACC_SLOPE(ND_SM)
\tLOGICAL LUM_ACC_READY
''')
    source = once(source, '\t  CALL THOMAS(TA,TB,TC,XM,ND,1)', '''\t  IF(ICOUNT==90.AND.INCL_ADVEC_TERMS)THEN
\t   LUM_ACC_TRI(1,:)=TA;LUM_ACC_TRI(2,:)=TB;LUM_ACC_TRI(3,:)=TC;LUM_ACC_TRI(4,:)=XM
\t   LUM_ACC_USED=dlnGRSQJdlnR
\t  END IF
\t  CALL THOMAS(TA,TB,TC,XM,ND,1)''')
    source = once(source, '\t    DAMP_FAC=0.8_LDP\n', '''\t    DAMP_FAC=0.8_LDP
\t    LUM_ACC_READY=.FALSE.
\t    IF(ICOUNT==90.AND..NOT.ACCURATE.AND.INCL_ADVEC_TERMS)THEN
\t     LUM_ACC_ALPHA=DTAUONQ*CON_DELTA/CHI_J
\t     LUM_ACC_ALPHA(1)=0;LUM_ACC_ALPHA(ND)=0
\t     ACCEL_LAST_META=[LUMINA_OUTER,LUMINA_FREQ_INDEX,LUMINA_JIT]
\t     CALL PROPOSE_GRADIENT(ND,R,LUM_ACC_TRI,LUM_ACC_USED,LUM_ACC_ALPHA,XM,
\t1        HU,HL,HS,EPS,EPS_PREV,GAM_RSQJNU_PREV,GAM_RSQHNU_PREV,LUM_AFFINE_FLUX_SOURCE,LUM_ACC_READY,LUM_ACC_SLOPE)
\t     ICOUNT=ICOUNT+1
\t    END IF
''')
    source = once(source, "\t      CALL DERIVCHI(TB,XM,R,ND,'LINMON')\n", '''\t      IF(LUM_ACC_READY)THEN
\t       dlnGRSQJdlnR=LUM_ACC_SLOPE
\t      ELSE
\t      CALL DERIVCHI(TB,XM,R,ND,'LINMON')
''')
    source = once(source, '\t    IF(ICOUNT .EQ.  100)THEN\n',
                  '\t      END IF\n\t    IF(ICOUNT .EQ.  100)THEN\n')
    return source
