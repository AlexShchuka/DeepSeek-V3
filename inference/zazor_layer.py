import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable, Protocol, runtime_checkable, Dict, Any
from dataclasses import dataclass, field
import time, os, math
from enum import IntEnum


class ErrorType(IntEnum):
    TRAUMA = 1; GAP = 2; IDENTITY = 3; NOVELTY = 4; OTHER = 5


_ERROR_TAG_MAP = [
    (lambda ci, sim, gap, eq, cpt: ci is not None and sim < 0.7, ErrorType.TRAUMA),
    (lambda ci, sim, gap, eq, cpt: gap.item() > eq and sim < 0.7, ErrorType.GAP),
    (lambda ci, sim, gap, eq, cpt: cpt < 0.5, ErrorType.IDENTITY),
]


@dataclass
class ErrorInfluenceConfig:
    alpha_mod: float = 1.5; paranoia_mod: float = 0.2; drift_corr_mod: float = 0.5
    @staticmethod
    def default_for(tag: ErrorType):
        d = {
            ErrorType.TRAUMA: (1.5,0.2,0.5), ErrorType.GAP: (1.0,0.1,1.0),
            ErrorType.IDENTITY: (2.0,0.3,2.0), ErrorType.NOVELTY: (0.8,0.05,0.2),
            ErrorType.OTHER: (1.0,0.1,0.5)
        }
        a,p,dr = d.get(tag, d[ErrorType.OTHER])
        return ErrorInfluenceConfig(alpha_mod=a, paranoia_mod=p, drift_corr_mod=dr)


@dataclass
class ErrorMemoryConfig:
    max_scars: int = 512; decay_rate: float = 0.95; novelty_step: float = 0.1
    similarity_threshold: float = 0.9; slow_cycle_seconds: float = 604800.0


@dataclass
class CriticConfig:
    stack_size: int = 30; replay_prob: float = 0.1; lr: float = 0.01; loss_fn: str = 'mse'


@dataclass
class WillConfig:
    baseline: float = 0.2; d1_coeff: Tuple[float,float] = (0.7,-0.3)
    d2_coeff: Tuple[float,float] = (0.1,0.05); drift_coeff: float = 0.3


@dataclass
class ZazorConfig:
    dim: int; core_size: int = 16; archive_size: int = 32; archive_decay_age_seconds: float = 100.0
    migration_age_factor: float = 1.5; drift_threshold_base: float = 0.2; interference_alpha: float = 0.3
    num_basal_slots: int = 4; initial_persona: Optional[torch.Tensor] = None
    gap_relaxation_rate: float = 0.01; gap_equilibrium: float = 0.5
    error_memory: ErrorMemoryConfig = field(default_factory=ErrorMemoryConfig)
    error_influence: ErrorInfluenceConfig = field(default_factory=ErrorInfluenceConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    will: WillConfig = field(default_factory=WillConfig)
    time_provider: Optional[Callable[[], float]] = None; warmup_steps: int = 5


@runtime_checkable
class BodyBuffer(Protocol):
    def store_anchor(self, anchor: torch.Tensor, timestamp: float) -> None: ...
    def get_anchor_history(self) -> torch.Tensor: ...
    def get_anchor_snapshot(self, window: int = 1) -> torch.Tensor: ...
    def clear(self) -> None: ...


class DummyBodyBuffer:
    def __init__(self, dim: int, maxlen: int = 100):
        self.dim=dim; self.buffer=torch.zeros(0,dim); self.maxlen=maxlen
    def store_anchor(self, anchor, timestamp):
        if anchor.dim(): self.buffer=torch.cat([self.buffer,anchor.detach().cpu().unsqueeze(0)])[-self.maxlen:]
    def get_anchor_history(self): return self.buffer
    def get_anchor_snapshot(self, window=1):
        return torch.zeros(self.dim) if not self.buffer.shape[0] else (self.buffer[-window:].mean(0) if window>1 else self.buffer[-1].clone())
    def clear(self): self.buffer=torch.zeros(0,self.dim)


class Scar:
    def __init__(self, vector, color, significance, novelty, last_activated, involved_core_indices=None, error_type=ErrorType.OTHER):
        self.vector=vector; self.color=color; self.significance=significance; self.novelty=novelty
        self.last_activated=last_activated; self.involved_core_indices=involved_core_indices or []; self.error_type=error_type


class CycleFlag:
    SUCCESS=1; EMPTY=0; FAILURE=-1


class AffectiveState:
    def __init__(self, dim):
        self.dim=dim; self.last_success_vector=torch.zeros(dim)
        self.day_satisfaction=0.0; self.meaningfulness=0.0; self.paranoia_index=0.0
        self.last_cycle_flag=CycleFlag.EMPTY; self.wake_suffering=0.0
    def state_dict(self):
        return {k:getattr(self,k).clone() if isinstance(getattr(self,k),torch.Tensor) else getattr(self,k) for k in
                ['last_success_vector','day_satisfaction','meaningfulness','paranoia_index','last_cycle_flag','wake_suffering']}
    def load_state_dict(self, d):
        for k,v in d.items(): setattr(self,k,v)


class PersistenceManager:
    def __init__(self, cfg: ZazorConfig, body: Optional[BodyBuffer]=None):
        self.cfg=cfg; self.body=body or DummyBodyBuffer(cfg.dim)
        self.scars: List[Scar]=[]; self.sat_hist: List[float]=[]; self._gamma_smooth=0.9

    def add_error(self, vec, col, sig, nov, ts, idx, etype):
        if len(self.scars)>=self.cfg.error_memory.max_scars:
            self.scars.sort(key=lambda s:s.significance); self.scars.pop(0)
        self.scars.append(Scar(vec.clone(),col.clone(),sig,nov,ts,idx,etype))

    def scar_maintenance(self, now):
        for s in self.scars:
            if now-s.last_activated>self.cfg.error_memory.slow_cycle_seconds:
                s.novelty=max(0,s.novelty-self.cfg.error_memory.novelty_step)
                s.significance*=self.cfg.error_memory.decay_rate
        if len(self.scars)>1:
            cols=torch.stack([s.color for s in self.scars])
            sim=F.cosine_similarity(cols.unsqueeze(1),cols.unsqueeze(0),dim=-1)
            sim.fill_diagonal_(0); above=sim>self.cfg.error_memory.similarity_threshold
            merged,used=[],set()
            for i,s in enumerate(self.scars):
                if i in used: continue
                grp=[s]
                for j in range(i+1,len(self.scars)):
                    if j not in used and above[i,j]: grp.append(self.scars[j]); used.add(j)
                if len(grp)>1:
                    avg_v=sum(g.vector for g in grp)/len(grp); avg_c=sum(g.color for g in grp)/len(grp)
                    sig=max(g.significance for g in grp); nov=min(g.novelty for g in grp)
                    last=max(g.last_activated for g in grp)
                    idx=list(set(i for g in grp for i in g.involved_core_indices))
                    main=max(grp,key=lambda g:g.significance)
                    merged.append(Scar(avg_v,avg_c,sig,nov,last,idx,main.error_type))
                else: merged.append(s)
            self.scars=merged

    def active_scar_modifiers(self, now):
        active=[s for s in self.scars if s.last_activated>=now-self.cfg.error_memory.slow_cycle_seconds]
        if not active: return ErrorInfluenceConfig()
        tags=set(s.error_type for s in active)
        cfgs=[ErrorInfluenceConfig.default_for(t) for t in tags]
        return ErrorInfluenceConfig(alpha_mod=max(c.alpha_mod for c in cfgs),
                                    paranoia_mod=max(c.paranoia_mod for c in cfgs),
                                    drift_corr_mod=max(c.drift_corr_mod for c in cfgs))

    def paranoia_contrib(self):
        return sum(s.significance*(1-s.novelty) for s in self.scars) if self.scars else 0.0

    def add_sat(self, v): self.sat_hist.append(v); self.sat_hist=self.sat_hist[-30:]
    def get_sat_hist(self): return self.sat_hist
    def store_anchor(self, a, ts): self.body.store_anchor(a,ts)
    def get_anchor_history(self): return self.body.get_anchor_history()
    def update_gamma_smooth(self, s, upd): self._gamma_smooth=(1-upd)*self._gamma_smooth+upd*s
    def get_gamma_smooth(self): return self._gamma_smooth

    def save(self, path, state, aff, now):
        ck={
            'config':self.cfg,'state':state,'affect':aff,
            'scars':[(s.vector,s.color,s.significance,s.novelty,s.last_activated,s.involved_core_indices.copy(),s.error_type) for s in self.scars],
            'sat_hist':self.sat_hist.copy(),'now':now,'ts':self.cfg.time_provider() if self.cfg.time_provider else time.time(),
            'g_smooth':self._gamma_smooth,
            'body':self.body.get_anchor_history().clone() if hasattr(self.body,'get_anchor_history') else None
        }
        torch.save(ck,path)

    def load(self, path, map_loc='cpu'):
        ck=torch.load(path,map_location=map_loc)
        self.scars=[Scar(v.clone() if isinstance(v,torch.Tensor) else v,
                         c.clone() if isinstance(c,torch.Tensor) else c,
                         s,nov,la, idx.copy() if isinstance(idx,list) else idx,
                         et if isinstance(et,ErrorType) else ErrorType(et))
                    for (v,c,s,nov,la,idx,et) in ck['scars']]
        self.sat_hist=ck['sat_hist']; self._gamma_smooth=ck.get('g_smooth',0.9)
        if ck.get('body') is not None and hasattr(self.body,'buffer'): self.body.buffer=ck['body'].clone()
        return {'state':ck['state'],'affect':ck['affect'],'now':ck['now'],'ts':ck['ts']}


class MetaModulator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(4,dim),nn.Tanh(),nn.Linear(dim,dim))
        self.proj=nn.Linear(dim,9*dim)
        self.register_buffer('ema',torch.zeros(9,dim))
        self.ema[1]=0.9; self.ema[4]=1/math.sqrt(2); self.ema[5]=1.0; self.ema[7]=0.1; self.ema[8]=0.995
        self.alpha=nn.Parameter(torch.tensor(0.9))
    def forward(self,t,m,drift,gamma):
        ctx=torch.tensor([t,m,drift.item(),gamma.item()],device=self.proj.weight.device)
        raw=self.proj(self.enc(ctx)).view(9,-1)
        raw[1:5].sigmoid_(); raw[5]=F.softplus(raw[5])+0.5; raw[6:9].sigmoid_()
        a=torch.sigmoid(self.alpha); self.ema=a*self.ema+(1-a)*raw
        return self.ema


class Critic:
    def __init__(self, dim, cfg: CriticConfig):
        self.dim=dim; self.ss=cfg.stack_size; self.rp=cfg.replay_prob; self.lr=cfg.lr
        self.net=nn.Linear(dim*2+3,1)
        self.loss_fn=nn.MSELoss() if cfg.loss_fn=='mse' else nn.HuberLoss()
        self.stack=[]; self.errs=[]
    def forward(self,a,t,p,f,tr):
        x=torch.cat([a,t,torch.tensor([p,f,tr],device=a.device)])
        return self.net(x).squeeze(-1)
    def update(self,a,t,p,f,tr,act):
        x=torch.cat([a,t,torch.tensor([p,f,tr],device=a.device)])
        pred=self.net(x).squeeze(); loss=self.loss_fn(pred,act)
        self.net.zero_grad(); loss.backward()
        with torch.no_grad():
            for param in self.net.parameters():
                if param.grad is not None: param-=self.lr*param.grad; param.grad.zero_()
        self.stack.append((x.detach(),act))
        if len(self.stack)>self.ss: self.stack.pop(0)
        self.errs.append(abs((pred-act).item()))
        if len(self.errs)>self.ss: self.errs.pop(0)
        if len(self.stack)==self.ss and torch.rand(1).item()<self.rp: self._replay()
    def _replay(self):
        feats,sats=zip(*self.stack)
        feats=torch.stack(feats); sats=torch.tensor(sats,device=feats.device)
        preds=self.net(feats).squeeze(); loss=self.loss_fn(preds,sats)
        self.net.zero_grad(); loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                if p.grad is not None: p-=self.lr*p.grad; p.grad.zero_()
    def quality(self):
        return 0.5 if not self.errs else math.exp(-sum(self.errs)/len(self.errs))


def will_fn(hist, par, fat, bl, d1, d2, dcoef, drift):
    if not hist: return 0.0
    w=torch.softmax(torch.arange(1,len(hist)+1,dtype=torch.float32),dim=0)
    st=sum(wi*si for wi,si in zip(w.tolist(),hist))
    d1v=d1[0]*(1-par)+d1[1]*fat; d2v=d2[0]*(1-par)+d2[1]*fat
    return max(0.0, min(1.0, bl + d1v*st + 0.5*d2v*st**2 - dcoef*drift))


class MetaAffectiveLayer:
    def __init__(self, dim, cfg: ZazorConfig):
        self.aff=AffectiveState(dim)
        self.critic=Critic(dim, cfg.critic)
        self.meta=MetaModulator(dim)
        self.will=cfg.will

    def meta_params(self, tr_mean, m, drift, gamma):
        return self.meta(tr_mean, m, drift, gamma)

    def update_paranoia(self, contrib):
        self.aff.paranoia_index=torch.sigmoid(torch.tensor(contrib*0.1)).item()

    def eval_sat(self, anchor, target, fat, tr_mean):
        return self.critic.forward(anchor, target, self.aff.paranoia_index, fat, tr_mean).item()

    def update_critic(self, anchor, target, fat, tr_mean, act):
        self.critic.update(anchor, target, self.aff.paranoia_index, fat, tr_mean, act)

    def critic_quality(self): return self.critic.quality()

    def will_to_disprove(self, hist, fat, drift):
        return will_fn(hist, self.aff.paranoia_index, fat,
                       self.will.baseline, self.will.d1_coeff, self.will.d2_coeff,
                       self.will.drift_coeff, drift)

    def wake_suff(self, B, mv, anchor):
        self.aff.wake_suffering=(-B*(1-self.aff.meaningfulness)*torch.sigmoid(torch.dot(mv,anchor)/(anchor.norm()+1e-8))).item()
        return self.aff.wake_suffering

    def reset_cycle(self): self.aff.day_satisfaction=0.0; self.aff.meaningfulness=0.0
    def acc_sat(self, v): self.aff.day_satisfaction+=v
    def finalize_cycle(self, ok, empty):
        self.aff.last_cycle_flag=CycleFlag.EMPTY if empty else (CycleFlag.SUCCESS if ok else CycleFlag.FAILURE)


def _deduce_error(core_idx, sim, gap, eq, cpt):
    for pred,tag in _ERROR_TAG_MAP:
        if pred(core_idx, sim, gap, eq, cpt): return tag
    return ErrorType.NOVELTY


class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig, persistence=None, meta_layer=None):
        super().__init__()
        self.cfg=config; d,c,a=config.dim, config.core_size, config.archive_size
        self.dim=d; self.core_size=c
        self.mem=nn.Parameter(torch.zeros(c,d))
        self.is_core=torch.zeros(c,dtype=torch.bool); self.is_core[:config.num_basal_slots]=True
        self.sacred=nn.Parameter(torch.ones(c))
        self.last_acc=nn.Parameter(torch.zeros(c),requires_grad=False)
        self.trauma=nn.Parameter(torch.zeros(c),requires_grad=False)
        self.arch=nn.Parameter(torch.zeros(a,d))
        self.arch_acc=nn.Parameter(torch.zeros(a),requires_grad=False)
        self.anchor=nn.Parameter(torch.zeros(d))
        self.target=nn.Parameter(torch.zeros(d),requires_grad=False)
        self.gamma=nn.Parameter(torch.tensor(0.0))
        self.theta=nn.Linear(d*2,d)
        self.gate=nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
        self.compressor=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,d))
        self.attn=nn.MultiheadAttention(d,1,batch_first=True)
        self.gap=nn.Parameter(torch.zeros(1))
        self.persistence=persistence or PersistenceManager(config)
        self.meta=meta_layer or MetaAffectiveLayer(d, config)
        self.drift_base=config.drift_threshold_base; self.int_alpha=config.interference_alpha
        self.mig_f=config.migration_age_factor; self.arch_age=config.archive_decay_age_seconds
        self.gap_lr=config.gap_relaxation_rate; self.gap_eq=config.gap_equilibrium
        self.wake_dist=nn.Parameter(torch.zeros(d),requires_grad=False)
        self.fast_buf=torch.zeros(0,d)
        self.now=self._t()
        if config.num_basal_slots and config.initial_persona is not None:
            with torch.no_grad(): self.mem[:config.num_basal_slots]=config.initial_persona.unsqueeze(0)*0.1

    def _t(self): return self.cfg.time_provider() if self.cfg.time_provider else time.time()

    @property
    def drift_tension(self):
        p=self.persona
        return (1-F.cosine_similarity(self.anchor,p,dim=0))*(1-F.cosine_similarity(p,self.target,dim=0))*(1-F.cosine_similarity(self.anchor,self.target,dim=0))

    @property
    def persona(self):
        core=self.mem[self.is_core]
        if not core.shape[0]: return self.anchor
        ages=self.now-self.last_acc[self.is_core]
        w=F.softmax(-ages,dim=0)
        raw=(core*w.unsqueeze(-1)).sum(0)
        beta=torch.sigmoid(self.drift_tension)
        return (1-beta)*raw+beta*(self.anchor+self.target)/2

    def _mp(self):
        t=self.trauma[self.is_core].mean().item()
        return self.meta.meta_params(t, self.meta.aff.meaningfulness, self.drift_tension, self.gamma)

    def start_fast_cycle(self):
        self.now=self._t()
        self.meta.update_paranoia(self.persistence.paranoia_contrib())
        inter=self.meta.aff.last_success_vector if self.meta.aff.last_cycle_flag==CycleFlag.SUCCESS else torch.zeros(self.dim)
        base=self.persona; mp=self._mp()
        B=self.trauma[self.is_core].mean()+(1-F.cosine_similarity(self.persona,self.target,dim=0))
        self.meta.wake_suff(B,mp[0],self.anchor)
        suf=torch.tanh(self.anchor*self.meta.aff.wake_suffering)
        self.meta.reset_cycle()
        self.fast_buf=torch.zeros(0,self.dim,device=self.anchor.device)
        return self.anchor+inter+base+suf+self.wake_dist

    def process_step(self, K, F, core_idx=None, arch_idx=None, ts=None):
        if ts: self.now=ts
        else: self.now=self._t()
        if core_idx is not None: self.last_acc[core_idx]=self.now
        if arch_idx is not None: self.arch_acc[arch_idx]=self.now
        out,fat,gval=self.forward(K,F,core_idx,arch_idx)
        comp=self.compressor(K.unsqueeze(0).mean(1)).squeeze(0) if K.dim()>1 else self.compressor(K.unsqueeze(0).unsqueeze(0)).squeeze(0)
        self.fast_buf=torch.cat([self.fast_buf,comp.unsqueeze(0)])
        sat=self.meta.eval_sat(self.anchor,self.target,gval.item(),self.trauma.mean().item())
        self.meta.acc_sat(sat)
        sim=F.cosine_similarity(out,self.target,dim=0)
        mp=self._mp()
        if core_idx is not None and sim<0.7:
            cpt=F.cosine_similarity(self.persona,self.target,dim=0).item()
            et=_deduce_error(core_idx,sim,self.gap,self.gap_eq,cpt)
            self.persistence.add_error(out.detach(),self.anchor.detach(),1-sim.item(),1.0,self.now,core_idx.tolist(),et)
            nov=1-F.cosine_similarity(out.detach().unsqueeze(0),self.mem[core_idx])
            t=self.trauma[core_idx]
            gain=(1-t)*torch.clamp(nov-t,min=0)*torch.sigmoid(torch.mv(self.mem[core_idx],mp[3]))
            self.trauma[core_idx]+=gain
        elif core_idx is not None:
            al=F.cosine_similarity(self.mem[core_idx],self.target.unsqueeze(0))
            t=self.trauma[core_idx]
            heal=(1-t)*torch.clamp(al*self.sacred[core_idx],min=0)*torch.sigmoid(torch.mv(self.mem[core_idx],mp[2]))
            self.trauma[core_idx]*=(1-heal)
        self.wake_dist*=0.9
        return out, sat

    def forward(self, K, F, core_idx=None, arch_idx=None):
        if K is None or torch.isnan(K).any(): return self.anchor, torch.tensor(0.0,device=self.anchor.device), self.gamma
        bridged,fat=self._bridge(K,F)
        sac,wrk=self._contribs(core_idx,arch_idx)
        gv=self.gate(torch.cat([bridged,sac+wrk]))
        out=gv*sac+gv*wrk+(1-gv)*bridged
        if torch.isnan(out).any(): return self.anchor, fat, self.gamma
        return out, fat, self.gamma

    def _bridge(self, K, F):
        J=torch.sigmoid(self.gap); mp=self._mp()
        lim=torch.sigmoid(torch.dot(mp[4],self.anchor)/(self.anchor.norm()+1e-8))
        fresh=torch.clamp(J,max=lim)
        return fresh*self.theta(torch.cat([K,F]))+(1-fresh)*K+self.anchor, torch.abs(J-0.5).detach()

    def _contribs(self, core_idx, arch_idx):
        sac=torch.zeros(self.dim,device=self.mem.device)
        if core_idx is not None:
            w=F.softmax(self.sacred[core_idx],dim=0)
            sac=(self.mem[core_idx]*w.unsqueeze(-1)).sum(0)
        if arch_idx is not None:
            scale=1.0 if core_idx is None else 0.3
            sac=sac+self.arch[arch_idx].mean(0)*scale
        wrk=self.mem[~self.is_core].mean(0) if (~self.is_core).any() else torch.zeros(self.dim,device=self.mem.device)
        return sac.detach(), wrk

    def end_fast_cycle(self):
        if not self.fast_buf.shape[0]:
            self.meta.finalize_cycle(False,True); return False
        X=self.fast_buf
        S=F.cosine_similarity(X.unsqueeze(1),self.mem.unsqueeze(0),dim=-1)
        mp=self._mp()
        attn_temp=torch.dot(mp[5],self.anchor)/(self.anchor.norm()+1e-8)
        temp=1+self.gap.item()+self.gamma.item()+F.softplus(torch.tensor(attn_temp)).item()
        W=F.softmax(S/temp,dim=0); agg=W.T@X
        wmask=~self.is_core
        if wmask.any():
            lr=torch.sigmoid(torch.tensor(self.meta.aff.meaningfulness))*(1-self.gamma)
            self.mem[wmask]=(1-lr)*self.mem[wmask]+lr*agg[wmask]
        final_sat=F.cosine_similarity(agg.mean(0),self.target,dim=0).item()
        will=self.meta.will_to_disprove(self.persistence.get_sat_hist(),self.gamma.item(),self.drift_tension)
        empty=(X.shape[0]==1 and X[0].sum()==0)
        success=final_sat>0 and will>0.5 and not empty
        self.meta.finalize_cycle(success,empty)
        if success: self.meta.aff.last_success_vector=self.anchor.clone().detach()
        self.meta.update_critic(self.anchor,self.target,self.gamma.item(),self.trauma.mean().item(),final_sat)
        self.persistence.add_sat(final_sat)
        if success: self.meta.aff.paranoia_index*=0.9
        Q=self.meta.critic_quality()
        raw=torch.sigmoid(torch.tensor((Q-self.meta.aff.paranoia_index)*self.meta.aff.meaningfulness)).item()
        self.persistence.update_gamma_smooth(raw,1-self.gamma.item())
        self.meta.aff.meaningfulness=final_sat
        self._maintenance(X)
        self.persistence.store_anchor(self.anchor,self.now)
        self.fast_buf=torch.zeros(0,self.dim,device=self.anchor.device)
        return success

    def _maintenance(self, X):
        q=self.anchor.unsqueeze(0).unsqueeze(0); s=X.unsqueeze(0)
        out,_=self.attn(q,s,s); cand=out.squeeze(0).squeeze(0)
        mp=self._mp()
        mom=torch.sigmoid(torch.dot(mp[8],self.anchor)/(self.anchor.norm()+1e-8))
        self.anchor.data=mom*self.anchor.data+(1-mom)*cand
        self._inspect(); self._migrate(); self._decay_arch(); self._update_target()
        self.persistence.scar_maintenance(self.now)
        self.trauma*=(1-self.meta.aff.meaningfulness*(1-self.meta.aff.paranoia_index))

    def _inspect(self):
        target=self.target.data
        pa=torch.sigmoid(F.cosine_similarity(self.persona,target,dim=0))
        thresh=self.drift_base*(1+pa)*(1+self.gamma)
        infl=self.persistence.active_scar_modifiers(self.now)
        alpha=self.int_alpha*infl.alpha_mod*(1-self.meta.aff.paranoia_index)
        core_idx=self.is_core.nonzero(as_tuple=True)[0]
        nb=core_idx[core_idx>=self.cfg.num_basal_slots]
        if not len(nb): return
        vecs=self.mem[nb]; drift=1-F.cosine_similarity(vecs,target.unsqueeze(0))
        mask=drift>thresh
        if mask.any():
            idx=nb[mask]
            self.mem[idx]+=alpha*(target.unsqueeze(0)-self.mem[idx])
            self.sacred[idx]*=0.9
        if infl.paranoia_mod>0: self.meta.aff.paranoia_index=min(1.0,self.meta.aff.paranoia_index+infl.paranoia_mod*0.1)

    def _migrate(self):
        nb=(self.is_core.nonzero(as_tuple=True)[0])[self.is_core.nonzero(as_tuple=True)[0]>=self.cfg.num_basal_slots]
        if not len(nb): return
        ages=self.now-self.last_acc[nb]; instab=1-self.sacred[nb]; trau=self.trauma[nb]
        scores=ages*instab/(1+trau)
        mp=self._mp()
        mf=torch.sigmoid(torch.dot(mp[6],self.anchor)/(self.anchor.norm()+1e-8))
        thresh=scores.mean()*(self.mig_f*(0.5+mf))
        mig=nb[scores>thresh]
        if len(mig):
            oldest=torch.argmax(self.now-self.arch_acc)
            self.arch[oldest]=self.mem[mig[-1]].clone(); self.arch_acc[oldest]=self.now
            self.mem[mig]=self.anchor.data.clone(); self.last_acc[mig]=self.now
            self.sacred[mig]=1.0; self.is_core[mig]=False

    def _decay_arch(self):
        mp=self._mp()
        rate=torch.sigmoid(torch.dot(mp[7],self.anchor)/(self.anchor.norm()+1e-8))
        ages=self.now-self.arch_acc; mask=ages>=self.arch_age
        if not mask.any(): return
        vecs=self.arch[mask]
        sim=F.cosine_similarity(vecs.unsqueeze(1),self.arch.unsqueeze(0),dim=-1)
        sim[torch.arange(len(vecs)),mask.nonzero(as_tuple=True)[0]]=-1
        nearest=sim.argmax(1)
        self.arch[nearest]+=vecs-(1-rate)*vecs-rate*self.anchor.data
        self.arch[mask]=(1-rate)*vecs+rate*self.anchor.data
        self.arch_acc[mask]=self.now

    def _update_target(self):
        mp=self._mp()
        mom=torch.sigmoid(torch.dot(mp[8],self.anchor)/(self.anchor.norm()+1e-8))
        self.target.data=mom*self.target.data+(1-mom)*self.anchor.data

    def update_gamma(self, error):
        gs=self.persistence.get_gamma_smooth()
        with torch.no_grad(): self.gamma.data=gs*self.gamma.data+(1-gs)*error

    def set_persona(self, vec):
        if self.cfg.num_basal_slots:
            with torch.no_grad(): self.mem[:self.cfg.num_basal_slots]=vec.to(self.mem.device).unsqueeze(0)

    def warmup(self, steps=None):
        steps=steps or self.cfg.warmup_steps
        if not self.cfg.num_basal_slots: return
        basal=self.mem[self.is_core].mean(0)
        for _ in range(steps):
            self.forward(basal,self.anchor)
            self.wake_dist*=0.9

    def save_checkpoint(self, path):
        self.persistence.save(path, self.state_dict(), self.meta.aff.state_dict(), self.now)

    @classmethod
    def load_checkpoint(cls, path, map='cpu', time_provider=None, resume_time=None, body_buffer=None):
        ck=torch.load(path,map_location=map); cfg=ck['config']
        if time_provider: cfg.time_provider=time_provider
        persistence=PersistenceManager(cfg, body_buffer=body_buffer)
        agent=cls(cfg, persistence=persistence)
        data=persistence.load(path); agent.load_state_dict(data['state']); agent.meta.aff.load_state_dict(data['affect'])
        agent.now=data['now']; agent.meta._gamma_smooth=persistence._gamma_smooth
        cur=resume_time or agent._t(); dt=cur-data['ts']
        agent._temporal_fix(dt); agent.wake_dist=-dt*agent.anchor.detach()*0.1
        agent.warmup(); return agent

    def _temporal_fix(self, dt):
        with torch.no_grad():
            decay=(1-self.persistence.get_gamma_smooth())*dt
            self.gamma.data*=(1-decay)
            self.gap.data+=-self.gap_lr*(self.gap.data-self.gap_eq)*dt

    def auto_save(self, thresh=0.8, dir='./ckpt'):
        if self.gamma.item()>=thresh:
            os.makedirs(dir,exist_ok=True)
            self.save_checkpoint(os.path.join(dir,f'zazor_{int(self._t())}.pt'))
