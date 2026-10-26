"""채널별 같은 물질/장 잔차를 합한다. 잔차0과 물리 동등성 판정은 구분한다."""
import numpy as np


def balance(bf,bb,ff,adiabatic,external,artificial):
    """열원은 가열+, 단열은 냉각+다. 충돌 열량을 광자 RE에 다시 더하지 않는다."""
    terms={name:np.asarray(value,dtype=np.longdouble) for name,value in
           (('bf',bf),('bb',bb),('ff',ff),('adiabatic',adiabatic),
            ('external',external),('artificial',artificial))}
    shape=terms['bf'].shape
    if (len(shape)!=2 or shape[0]!=2 or not shape[1] or
        any(v.shape!=shape for k,v in terms.items() if k in ('bf','bb','ff')) or
        any(v.shape!=(shape[1],) for k,v in terms.items() if k not in ('bf','bb','ff')) or
        not all(np.isfinite(v).all() for v in terms.values()) or
        any(np.any(terms[k]<0) for k in ('bf','bb','ff','external','artificial'))):
        raise ValueError('열장부 채널 형상/부호/비유한 오류')
    nets={k:terms[k][0]-terms[k][1] for k in ('bf','bb','ff')}
    net=sum(nets.values())-terms['adiabatic']+terms['external']+terms['artificial']
    gross=(sum(terms[k].sum(axis=0) for k in ('bf','bb','ff'))+abs(terms['adiabatic'])+
           terms['external']+terms['artificial'])
    channel_scale=(sum(abs(v) for v in nets.values())+abs(terms['adiabatic'])+
                   terms['external']+terms['artificial'])
    def scaled(denominator):
        return np.divide(abs(net),denominator,out=np.zeros_like(net),where=denominator>0)
    return dict(net=np.asarray(net,dtype='f8'),gross=np.asarray(gross,dtype='f8'),
        gross_normalized=np.asarray(scaled(gross),dtype='f8'),
        channel_scale=np.asarray(channel_scale,dtype='f8'),
        channel_normalized=np.asarray(scaled(channel_scale),dtype='f8'))
