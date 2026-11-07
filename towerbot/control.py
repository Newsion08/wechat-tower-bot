"""Timing and state logic; independent of Windows and usable in replay/tests."""
from collections import deque
from dataclasses import dataclass, field, asdict
import math
import numpy as np


class Track:
    def __init__(self, seconds=.24):
        self.points=deque()
        self.seconds=seconds

    def add(self,t,x,y):
        if self.points and t<=self.points[-1][0]: return
        self.points.append((t,x,y))
        while len(self.points)>3 and t-self.points[0][0]>self.seconds:
            self.points.popleft()

    def clear(self): self.points.clear()

    def fit(self,axis=1,degree=2):
        if len(self.points)<4: return None
        data=np.array(self.points)
        t=data[:,0]-data[-1,0]
        if np.ptp(t)<.045: return None
        degree=min(degree,len(t)-2)
        weights=np.exp(t/.18)
        A=np.vander(t,degree+1,increasing=True)
        coef=np.linalg.lstsq(A*weights[:,None],data[:,axis]*weights,rcond=None)[0]
        residual=data[:,axis]-A@coef
        noise=max(.35,float(np.sqrt(np.mean(residual**2))))
        covariance=np.linalg.pinv((A*weights[:,None]).T@(A*weights[:,None]))*noise**2
        return Fit(float(data[-1,0]),coef,covariance,noise)


@dataclass
class Fit:
    t: float
    coef: np.ndarray
    covariance: np.ndarray
    noise: float

    def predict(self,dt):
        a=np.power(dt,np.arange(len(self.coef)))
        return float(a@self.coef),float(np.sqrt(max(0,a@self.covariance@a)))

    @property
    def velocity(self): return float(self.coef[1])


@dataclass
class PeriodicFit:
    t: float
    omega: float
    coef: np.ndarray
    covariance: np.ndarray
    noise: float

    def predict(self,dt):
        a=np.array([1.,math.sin(self.omega*dt),math.cos(self.omega*dt)])
        return float(a@self.coef),float(math.sqrt(self.noise**2+max(0,a@self.covariance@a)))

    @property
    def velocity(self):return float(self.omega*self.coef[1])


class PeriodicTrack(Track):
    """Identify the current slab's oscillation; never assume last floor's speed."""
    def __init__(self):
        super().__init__(1.2)
        self.omega=None;self.frequency_time=-float('inf')

    def clear(self):
        super().clear();self.omega=None;self.frequency_time=-float('inf')

    def fit(self,axis=1,degree=2):
        if len(self.points)<20:return None
        a=np.array(self.points);ts=a[:,0]-a[-1,0];ys=a[:,1]
        if np.ptp(ts)<.8:return None
        if self.omega is None or a[-1,0]-self.frequency_time>.12:
            best=(float('inf'),None)
            for omega in np.linspace(2.,14.,121):
                design=np.column_stack([np.ones_like(ts),np.sin(omega*ts),np.cos(omega*ts)])
                c=np.linalg.lstsq(design,ys,rcond=None)[0]
                error=float(np.mean((design@c-ys)**2))
                if error<best[0]:best=(error,omega)
            self.omega=best[1];self.frequency_time=a[-1,0]
        design=np.column_stack([np.ones_like(ts),np.sin(self.omega*ts),np.cos(self.omega*ts)])
        coef=np.linalg.lstsq(design,ys,rcond=None)[0]
        noise=max(.5,float(np.sqrt(np.mean((design@coef-ys)**2))))
        if noise>5:return None
        cov=np.linalg.pinv(design.T@design)*noise**2
        return PeriodicFit(float(a[-1,0]),self.omega,coef,cov,noise)


@dataclass
class Calibration:
    latency: float=.022
    gravity: float=4400.
    drift_ratio: float=.02
    bias_right: float=0.
    bias_left: float=0.
    samples: int=0
    latencies: list=field(default_factory=list)
    balance_response: float=.20
    balance_samples: int=0

    @classmethod
    def from_dict(cls,d):
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})

    def dict(self): return asdict(self)


class Controller:
    def __init__(self,calibration=None,require_periodic=True):
        self.cal=calibration or Calibration()
        self.block_track=Track(.24)
        self.periodic_track=PeriodicTrack()
        self.require_periodic=require_periodic
        self.target_track=Track(.28)
        self.state='PAUSED'
        self.armed=False
        self.last_t=-float('inf')
        self.obs=None
        self.clicked_at=None
        self.drop=[]
        self.drop_origin=None
        self.drop_target=None
        self.release_velocity=0.
        self.landings=0
        self.events=[]
        self.last_seen=-float('inf')
        self.last_click=-float('inf')
        self.drop_perfect=False
        self.lean_track=Track(.7)
        self.drop_balance=None
        self.last_candidate=None
        self.drop_width=0.
        self.drop_target_width=0.
        self.wait_reason='paused'

    def arm(self):
        self.armed=True
        self.state='WAIT'
        self.wait_reason='collecting_motion'
        self.block_track.clear()
        self.periodic_track.clear()
        self.target_track.clear()
        self.lean_track.clear()
        self.clicked_at=None
        self.drop=[]

    def pause(self,reason='manual'):
        self.armed=False
        self.state='PAUSED'
        self.wait_reason=reason
        self.block_track.clear()
        self.periodic_track.clear()
        self.target_track.clear()
        self.events.append({'type':'pause','reason':reason})

    def observe(self,o):
        if o.t<=self.last_t: return
        self.last_t=o.t
        previous=self.obs
        self.obs=o
        if o.modal:
            if self.armed: self.pause('dialog')
            return
        if not self.armed: return
        if o.target:self.lean_track.add(o.t,o.lean,0)
        if self.state=='FALLING' and o.perfect:self.drop_perfect=True
        if o.target:
            if previous and previous.target and abs(o.target.y-previous.target.y)>15:
                self.target_track.clear()
            self.target_track.add(o.t,*o.target.top)
        if o.block:
            self.last_seen=o.t
            if self.state=='FALLING':
                # A new suspended slab appears far above the previous falling
                # slab, after either landing or a miss. Never retry a click.
                new_spawn=(o.t-self.last_click>.35 and self.drop and
                           o.block.y<self.drop[-1][2]-90)
                if new_spawn:
                    self.finish_drop(o)
                    self.state='WAIT'
                    self.block_track.clear()
                    self.periodic_track.clear()
                else:
                    self.drop.append((o.t,o.block.x,o.block.y,o.target.x if o.target else float('nan')))
                    return
            self.block_track.add(o.t,*o.block.bottom)
            self.periodic_track.add(o.t,*o.block.bottom)
            fy=self.block_track.fit(axis=2,degree=1)
            fx=self.block_track.fit(axis=1,degree=1)
            # A hanging slab follows the tilted crane rail. Remove that
            # vertical component before declaring an actual release.
            descent=fy.velocity-math.tan(math.radians(o.block.angle))*fx.velocity if fy and fx else 0.
            if fy and descent>75:
                # Manual releases during replay/arming cannot trigger again.
                self.state='FALLING'
                self.last_click=o.t
                self.clicked_at=None
                self.drop=[(o.t,o.block.x,o.block.y,o.target.x if o.target else float('nan'))]
                return
            if len(self.block_track.points)>=5 and fy and abs(descent)<50:
                self.state='TRACK'
        elif self.state=='FALLING':
            # Keep waiting for the next slab: landing animations can take .5s.
            if o.t-self.last_click>4: self.pause('landing_timeout')
        elif o.t-self.last_seen>.12:
            self.block_track.clear()
            # Keep the same slab's periodic history through an off-screen arc.
            # Fit is never used to click while the slab itself is invisible.
            if o.t-self.last_seen>2.:self.periodic_track.clear()
            self.state='WAIT'

    def propose(self,now):
        o=self.obs
        self.last_candidate=None
        if self.armed:self.wait_reason=self.state.lower()
        if self.armed and o and o.support_clipped:
            self.wait_reason='support_clipped';return None
        if not self.armed or self.state!='TRACK' or not o or not o.block or not o.target:
            if self.armed and o and self.state!='FALLING':
                self.wait_reason='block_not_visible' if not o.block else 'support_not_visible' if not o.target else 'collecting_motion'
            return None
        if now-o.t>.10 or now<o.t-.002:
            self.wait_reason='stale_frame';return None
        b=self.periodic_track.fit() if self.require_periodic else self.block_track.fit()
        target=self.target_track.fit(degree=1)
        if not b or not target:
            self.wait_reason='motion_fit_unavailable' if not b else 'support_fit_unavailable';return None
        gap=o.target.top[1]-o.block.bottom[1]
        if gap<65:
            self.wait_reason='block_near_support';return None
        T=math.sqrt(2*gap/self.cal.gravity)
        direction=1 if b.velocity>=0 else -1
        bias=self.cal.bias_right if direction>0 else self.cal.bias_left
        # Aiming only at the previous floor preserves accumulated offset. The
        # crane shaft supplies the unshifted construction axis; lean feedback
        # requests a restoring offset while keeping ample support overlap.
        leanfit=self.lean_track.fit(degree=1)
        rate=leanfit.velocity if leanfit else 0.
        axis_error=o.target.top[0]-o.axis_x if o.axis_x is not None else 0.
        width=min(o.block.w,o.target.w)
        # Large post-impact rotation moves the high tower top by many pixels.
        # A 0.35 s linear extrapolation during that transient can overshoot by
        # a substantial fraction of a narrow slab. Wait for it to settle.
        if self.require_periodic and abs(target.velocity)>max(12.,.2*width):
            self.wait_reason='tower_still_moving';return None
        angle=o.lean+.15*rate
        correction=0.
        if abs(o.lean)>.4 or abs(axis_error)>4:
            raw=-.6*axis_error-.35*math.radians(angle)/self.cal.balance_response*width
            correction=0. if abs(raw)<.04*width else float(np.clip(raw,-.18*width,.18*width))
            # Small offsets get snapped back to the existing (misplaced) top.
            if abs(axis_error)>8 and 0<abs(correction)<.095*width:
                correction=math.copysign(.095*width,correction)
        # Wait for an upcoming crossing, never fire late to catch a past one.
        start=max(0.,now-o.t)+self.cal.latency+.002
        stop=start+.12
        grid=np.linspace(start,stop,61)
        candidates=[]
        desired_correction=correction
        support_limit=(o.block.w+o.target.w)/2-.60*width
        for s in grid:
            xb,sb=b.predict(o.t+s-b.t)
            xt,st=target.predict(o.t+s+T-target.t)
            drift=self.cal.drift_ratio*b.velocity*T
            latency_noise=(np.std(self.cal.latencies) if len(self.cal.latencies)>2 else .006)
            sigma=math.sqrt(sb*sb+st*st+(b.velocity*latency_noise)**2)
            allowance=support_limit-2*sigma-1.
            if allowance<=0:continue
            actual_correction=float(np.clip(desired_correction,-allowance,allowance))
            error=xb+drift+bias-(xt+actual_correction)
            candidates.append((error,sigma,float(s),xb+drift,xt,actual_correction))
        # Prefer the next zero crossing; a closest frame is insufficient at
        # high speed. An edge minimum requires an actually small error.
        if not candidates:
            self.wait_reason='uncertainty_exceeds_support';return None
        error,sigma,s,landing,xt,correction=min(candidates,key=lambda z:abs(z[0]))
        width=min(o.block.w,o.target.w)
        self.last_candidate={'error':error,'sigma':sigma,'correction':correction,'width':width}
        if abs(error)>max(.8,.012*width):
            self.wait_reason='waiting_for_crossing';return None
        if abs(correction)+abs(error)+2*sigma>support_limit:
            self.wait_reason='insufficient_overlap';return None
        self.wait_reason='scheduled'
        return {'deadline':o.t+s-self.cal.latency,'release':o.t+s,
                'error':error,'sigma':sigma,'landing_x':landing,'target_x':xt,
                'flight':T,'velocity':b.velocity,'observed':o.t,
                'correction':correction,'requested_correction':desired_correction,
                'lean':o.lean,'axis_error':axis_error,'width':width}

    def clicked(self,t,proposal):
        if self.state!='TRACK' or not self.armed:
            raise RuntimeError('Click attempted outside armed TRACK state')
        self.state='FALLING'
        self.last_click=self.clicked_at=t
        o=self.obs
        self.drop_origin=(o.block.x,o.block.y)
        self.drop_target=(o.target.x,o.target.y)
        self.drop_width=o.block.w;self.drop_target_width=o.target.w
        self.release_velocity=proposal['velocity']
        self.drop=[]
        self.drop_perfect=False
        self.drop_balance={k:proposal.get(k,0) for k in ('correction','lean','axis_error','width')}
        self.events.append({'type':'click','t':t,**proposal})

    def finish_drop(self,o):
        if self.clicked_at is None or len(self.drop)<5 or not o.target: return
        data=np.array(self.drop)
        data=data[np.isfinite(data).all(axis=1)]
        # Discard images acquired before input and the still-suspended phase;
        # otherwise pre-release travel is incorrectly learned as falling drift.
        data=data[(data[:,0]>self.clicked_at)&(data[:,2]>self.drop_origin[1]+2)]
        if len(data)<5:return
        observed_error=float(data[-1,1]-data[-1,3])
        if self.drop_width and self.drop_target_width:
            overlap_limit=(self.drop_width+self.drop_target_width)/2-.5*min(self.drop_width,self.drop_target_width)
            if abs(observed_error)>overlap_limit:
                self.events.append({'type':'missed_drop','t':o.t,'error':observed_error})
                self.clicked_at=None
                return
        data=data[np.r_[True,np.abs(np.diff(data[:,2]))>.1]]
        if len(data)<5:return
        # Reject a missed slab; the support has not gained a floor. The camera
        # may start scrolling at high floors, so compare vertical separation
        # before collision as well as the support's visible displacement.
        near=data[-1,2]+15>=self.drop_target[1]-40
        if not near:
            self.events.append({'type':'miss_or_unobserved','t':o.t})
            return
        t=data[:,0]-self.clicked_at
        p=np.polyfit(t,data[:,2],2)
        delay=-p[1]/(2*p[0]) if p[0]>0 else -1
        fit_error=float(np.std(np.polyval(p,t)-data[:,2]))
        if 1000<2*p[0]<12000 and -.005<delay<.15 and fit_error<8:
            self.cal.gravity=.75*self.cal.gravity+.25*float(2*p[0])
            self.cal.latencies.append(float(max(0,delay)))
            self.cal.latencies=self.cal.latencies[-15:]
            self.cal.latency=float(np.median(self.cal.latencies))
        dx=float(data[-1,1]-data[0,1])
        duration=data[-1,0]-data[0,0]
        if abs(self.release_velocity)>30 and duration>.15:
            ratio=dx/(self.release_velocity*duration)
            if -.10<ratio<.20:
                self.cal.drift_ratio=.8*self.cal.drift_ratio+.2*ratio
        # Use the last pre-impact support (not the animated post-impact tower)
        # to measure alignment. Bias is updated conservatively in block widths.
        err=float(data[-1,1]-data[-1,3])
        commanded=self.drop_balance['correction'] if self.drop_balance else 0.
        residual=err-commanded
        if abs(residual)<.15*o.target.w:
            key='bias_right' if self.release_velocity>=0 else 'bias_left'
            setattr(self.cal,key,float(np.clip(getattr(self.cal,key)+.12*residual,-10,10)))
        if self.drop_balance and abs(commanded)>.06*self.drop_balance['width']:
            delta=math.radians(o.lean-self.drop_balance['lean'])
            normalized=err/self.drop_balance['width']
            response=delta/normalized if abs(normalized)>.03 else 0
            if .01<response<1.5:
                self.cal.balance_response=.85*self.cal.balance_response+.15*response
                self.cal.balance_samples+=1
            self.events.append({'type':'balance_response','t':o.t,
                                'before_deg':self.drop_balance['lean'],'after_deg':o.lean,
                                'commanded_offset':commanded,'observed_offset':err,
                                'response':response,'model_response':self.cal.balance_response,
                                'axis_error_before':self.drop_balance['axis_error'],
                                'axis_error_after':o.target.top[0]-o.axis_x if o.axis_x is not None else None})
        self.cal.samples+=1
        self.landings+=1
        self.events.append({'type':'landing','t':o.t,'preimpact_error':err,
                            'perfect_visible':self.drop_perfect,'calibration':self.cal.dict()})
        self.clicked_at=None
