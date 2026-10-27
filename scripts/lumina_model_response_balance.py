"""같은 동적 C 후보의 FF·깊이 결합 AD·물리/native 광자 열장부.

외부열 정책 두 가지를 분리하며 저장0을 순간 물리 열원으로 바꾸지 않는다.
원본 reference를 수정하거나 charge 수렴을 RE 성공으로 승격하지 않는다.
"""
import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from lumina_model_balance import balance
from lumina_model_energy import ModelThermalEnergy
from lumina_model_freefree import FreeFreeCoefficients
from lumina_native_heat_sources import vector,cooling_vector,artificial_heat,instantaneous_profile
from run_lumina_fixed_state_balance import adiabatic


def ion_number_flows(roster, group, index, population, fraction, rates):
    """이미 소비한 BF율로 Si/S 각 이온의 광이온화·재결합 수를 합한다."""
    targets = ((14, 2), (14, 3), (16, 2), (16, 3))
    ns = population.shape[1]
    flows = np.zeros((len(targets), 4, ns), dtype=np.longdouble)
    for k, (z, stage) in enumerate(targets):
        selected = np.flatnonzero((roster['Z'] == z) & (roster['stage'] == stage))
        for start in range(0, len(selected), 1024):
            ids = selected[start:start+1024]
            lo, up = roster['lower'][ids], roster['target'][ids]
            upper = [group[t] if t >= 0 else index[int(zz), int(q), -1]
                     for t, zz, q in zip(up, roster['Z'][ids], roster['stage'][ids])]
            lower_pop = np.asarray(population[group[lo]]*fraction[lo], dtype=np.longdouble)
            upper_pop = np.asarray(population[upper], dtype=np.longdouble)
            values = np.asarray(rates[ids], dtype=np.longdouble)
            probability = roster['probability'][ids, None]
            flows[k, 0] += np.sum(probability*lower_pop*values[:, :, 0], axis=0)
            flows[k, 1] += np.sum(probability*upper_pop*values[:, :, 1], axis=0)
            flows[k, 2] += np.sum(probability*upper_pop*values[:, :, 2], axis=0)
        flows[k, 3] = flows[k, 0]-flows[k, 1]-flows[k, 2]
    return np.asarray(targets, dtype='i4'), flows


def ion_number_densities(labels, columns, population):
    """Roman 단계가 아닌 전하 주소로 이온당 율의 분모를 명시한다.

    photo 흐름을 이 수밀도로 나눈 값이 이온당 평균 Gamma다. 수밀도0의
    나눗셈 규약이나 새 물리 gate를 만들지 않고 실제 분모만 출력한다.
    """
    charges = np.asarray(labels, dtype='i4').copy()
    charges[:, 1] -= 1
    densities = np.asarray([population[(columns[:, 0] == z) & (columns[:, 1] == charge)].sum(axis=0)
                            for z, charge in charges])
    return charges, densities


class ResponseBalance:
    def __init__(self,run,rows,models,columns,reference,radius,epoch,adiabatic_library,
                 roster,lines,jbar,ff,ff_coefficients,seed_population):
        self.owner=ModelThermalEnergy.from_model(run,rows,models,columns,reference)
        self.run=run;self.radius=radius;self.epoch=epoch;self.library=adiabatic_library
        self.roster=roster;self.lines=lines;self.jbar=jbar;self.ff=ff
        self.coefficients=FreeFreeCoefficients(ff.signature,ff_coefficients['temperature'],
            ff_coefficients['heating'],ff_coefficients['cooling'])
        if str(ff_coefficients['signature'])!=ff.signature:raise ValueError('FF 실제 장/모형 신원 불일치')
        self.columns=columns;self.index={tuple(map(int,k)):i for i,k in enumerate(columns)}
        self.model=run/'MODEL'
        self.scaling_enabled=S.control(self.model,'SCL_LN')=='T'
        self.scaling_limit=S.number(self.model,'SCL_LN_FAC')
        self.density_limit=S.number(self.model,'SCL_DEN_LIM','1e30');self.tmin=S.number(self.model,'T_MIN')*1.e4
        if S.control(self.model,'INC_AD')!='T' or S.control(self.model,'COH_ES')!='T':
            raise ValueError('미지원 에너지 과정 계약')
        self.stored=vector((run/'RVTJ').read_text(),'Heating: radioactive decay (ergs/cm^3/s)',len(radius))
        text=(run/'GENCOOL').read_text()
        self.shock=cooling_vector(text,'Shock Power Term [ergs/cm**3/sec]',len(radius),omitted_sum_zero=True)
        self.xray=cooling_vector(text,'X-ray heating term [ergs/cm**3/sec]',len(radius),omitted_sum_zero=True)
        if any(np.any(v<0) for v in (self.stored,self.shock,self.xray)) or np.any(self.xray!=0):
            raise ValueError('미지원 외부 열원')
        self.instant,self.decay_metadata=instantaneous_profile(run,columns,seed_population,radius,epoch)

    def evaluate(self,result,trial,rates):
        pop=result['population'];f=trial.fraction;ns=len(self.radius);group=self.owner.full_to_sl
        nuclei=pop.sum(axis=0);bf=np.zeros((2,ns),dtype=np.longdouble);bb=np.zeros_like(bf);native=np.zeros_like(bf)
        for first in range(0,len(rates),1024):
            last=min(len(rates),first+1024);lo=self.roster['lower'][first:last];up=self.roster['target'][first:last]
            upper=[group[target] if target>=0 else self.index[int(z),int(stage),-1]
                   for target,z,stage in zip(up,self.roster['Z'][first:last],self.roster['stage'][first:last])]
            nlo=np.asarray(pop[group[lo]]*f[lo],dtype=np.longdouble);nup=np.asarray(pop[upper],dtype=np.longdouble)
            v=np.asarray(rates[first:last],dtype=np.longdouble)
            edge=self.roster['threshold'][first:last,None]*np.longdouble('6.62607015e-27')
            probability=self.roster['probability'][first:last,None]
            bf[0]+=np.sum(probability*nlo*(v[:,:,3]+edge*v[:,:,0]),axis=0)
            bf[1]+=np.sum(probability*nup*(v[:,:,4]+v[:,:,5]+edge*(v[:,:,1]+v[:,:,2])),axis=0)
        for first in range(0,len(self.jbar),4096):
            last=min(len(self.jbar),first+4096);lo=self.lines['lower'][first:last];up=self.lines['upper'][first:last]
            nu=self.lines['nu0'][first:last];A=self.lines['A'][first:last];bul=A*B.C**2/(2*B.H*nu**3)
            lower=np.asarray(pop[group[lo]]*f[lo],dtype=np.longdouble)
            upper=np.asarray(pop[group[up]]*f[up],dtype=np.longdouble);J=self.jbar[first:last]
            forward=lower*(bul/self.lines['g_ratio'][first:last])[:,None]*J*(B.H*nu)[:,None]
            reverse=upper*(A[:,None]+bul[:,None]*J)*(B.H*nu)[:,None]
            factor=self.owner.line_scaling(lo,up,nu,nuclei,enabled=self.scaling_enabled,
                ratio_limit=self.scaling_limit,density_limit=self.density_limit)
            bb+=np.stack([forward.sum(axis=0),reverse.sum(axis=0)])
            native+=np.stack([(forward*factor).sum(axis=0),(reverse*factor).sum(axis=0)])
        checks={}
        for name,terms,slot in (('BF',bf,0),('BB',bb,1)):
            c=result['ledger'][:,:,slot].sum(axis=0,dtype=np.longdouble)
            scale=terms.sum(axis=0);error=abs(terms[0]-terms[1]-c)
            if np.any(error>1.e-10*scale):raise ValueError('동적 C와 독립 '+name+' 광자 열 불일치')
            checks[name]=float(np.max(np.divide(error,scale,out=np.zeros_like(error),where=scale>0)))
        if not np.array_equal(self.coefficients.temperature,result['temperature']):
            self.coefficients=self.ff.prepare(result['temperature'],radiation_generation=trial.radiation_generation)
        ff=self.ff.contract(self.coefficients,{**trial.state,'superlevel':pop},
            radiation_generation=trial.radiation_generation,material_generation=trial.material_generation)['terms']
        ff={k:np.stack([v['heating'],v['cooling']]) for k,v in ff.items()}
        physical=result['energy'][:,2];raw=self.owner.internal_energy(pop,f,'native_static_raw')
        ad={name:adiabatic(self.library,self.radius,self.epoch,result['temperature'],result['ne'],nuclei,u)['signed_total']
            for name,u in (('physical',physical),('native',raw))}
        art=artificial_heat(result['temperature'],result['ne'],nuclei,self.tmin)
        common=self.shock+self.xray
        ledgers={}
        for policy,external in (('stored',common+self.stored),('instantaneous_scenario',common+self.instant)):
            ledgers['physical_'+policy]=balance(bf,bb,ff['all_positive_ions'],ad['physical'],external,np.zeros(ns))
            ledgers['native_'+policy]=balance(bf,native,ff['native_upper_ions'],ad['native'],external,art)
        ion_labels, ion_flows = ion_number_flows(self.roster, group, self.index, pop, f, rates)
        ion_charges, ion_densities = ion_number_densities(ion_labels, self.columns, pop)
        arrays=dict(bf=bf,bb=bb,native_bb=native,physical_adiabatic=ad['physical'],native_adiabatic=ad['native'],
                    physical_internal_energy=physical,native_internal_energy=raw,artificial=art,
                    stored_decay=self.stored,instantaneous_scenario=self.instant,
                    bf_ion_Z_stage=ion_labels, bf_ion_number_flows=ion_flows,
                    bf_ion_Z_charge=ion_charges, bf_ion_number_density=ion_densities,
                    bf_ion_number_flow_channels=np.array(['photo', 'spontaneous', 'stimulated', 'net']))
        for k,v in ff.items():arrays['ff_'+k]=v
        summary={}
        for name,values in ledgers.items():
            for k,v in values.items():arrays[name+'_'+k]=v
            summary[name]=dict(max_abs_net=float(abs(values['net']).max()),
                max_gross_normalized=float(values['gross_normalized'].max()),
                max_channel_normalized=float(values['channel_normalized'].max()))
        return arrays,dict(summaries=summary,independent_C_photon_checks=checks)
