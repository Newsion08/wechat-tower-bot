from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    angle: float
    confidence: float
    kind: str

    @property
    def top(self):
        a = np.deg2rad(self.angle)
        return self.x + np.sin(a)*self.h/2, self.y-np.cos(a)*self.h/2

    @property
    def bottom(self):
        a = np.deg2rad(self.angle)
        return self.x - np.sin(a)*self.h/2, self.y+np.cos(a)*self.h/2

    def points(self):
        return cv2.boxPoints(((self.x, self.y), (self.w, self.h), self.angle))


@dataclass
class Observation:
    t: float
    block: Box | None
    target: Box | None
    modal: bool
    perfect: bool = False
    floors: int = 0
    axis_x: float | None = None
    lean: float = 0.
    support_clipped: bool = False

    def dict(self):
        return asdict(self)


def rectangle(contour, kind, confidence=1.):
    (x, y), (w, h), a = cv2.minAreaRect(contour)
    if w < h:
        w, h = h, w
        a += 90
    a = (a + 90) % 180 - 90
    return Box(float(x), float(y), float(w), float(h), float(a), confidence, kind)


class Detector:
    """All geometry uses a 720-pixel-wide analysis image, independent of DPI."""
    def __init__(self):
        self.previous = None
        self.previous_target = None
        self.feedback_template = None

    def detect(self, frame, timestamp):
        h0, w0 = frame.shape[:2]
        f = cv2.resize(frame, (720, round(h0*720/w0))) if w0 != 720 else frame
        h, w = f.shape[:2]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ymin = int(h*.18)
        # Both the dark revival backdrop and pale final-score backdrop have a
        # large uniform tinted upper play field; normal gameplay is white.
        background = float(np.median(g[int(h*.23):int(h*.38), 45:170]))
        modal = background < 244
        if modal:
            self.previous = None
            self.previous_target = None
            return Observation(timestamp, None, None, True), f
        # Hollow slabs: their dark outline encloses a nearly white interior.
        mask = (g < 180).astype(np.uint8)*255
        mask[:ymin] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []
        for c in contours:
            b = rectangle(c, 'block')
            # At high floors the suspension wires also form a closed white
            # rectangle (about 22 px high); slabs remain about 27-29 px high.
            if not (35 < b.w < 235 and 24 < b.h < 43 and b.w/b.h>1.2 and abs(b.angle) < 30):
                continue
            xs=b.points()[:,0]
            if xs.min()<12 or xs.max()>w-12:
                # A cut-off slab changes its apparent width and center. Wait
                # for the complete slab to re-enter, keeping periodic history.
                continue
            if cv2.contourArea(c) < .77*b.w*b.h:
                continue
            # Inspect an oriented inset; axis-aligned crops mix the dark border
            # with background when the tower rotates.
            inset = Box(b.x,b.y,b.w-9,b.h-9,b.angle,1,'')
            local = np.zeros(g.shape,np.uint8)
            cv2.fillConvexPoly(local,np.int32(inset.points()),255)
            mean = cv2.mean(f,local)[:3]
            if min(mean) > 240 and max(mean)-min(mean) < 10:
                blocks.append(b)
        # Filled floors use a small repeating palette. Discover its dominant
        # colors per frame, avoiding any learned model or fixed screen scale.
        ystart,yend=ymin,h
        if self.previous_target and timestamp-self.previous_target[0]<.15:
            old_target=self.previous_target[1]
            ystart=max(ymin,int(old_target.y-115))
            yend=min(h,int(old_target.y+65))
        # A small angle at a great height can put the whole top near an edge.
        # Do not assume the crane or the support stays in the central band.
        xstart,xend=12,w-12
        crop = f[ystart:yend, xstart:xend]
        sample = crop[::3, ::3].reshape(-1,3).astype(np.int16)
        valid = (sample.max(1)<238)&(sample.min(1)>35)&(np.ptp(sample,axis=1)>7)
        colors = (sample[valid]//12)*12+6
        floors = []
        clipped_floors=[]
        if len(colors):
            unique, counts = np.unique(colors,axis=0,return_counts=True)
            chosen=[]
            for k in np.argsort(counts)[::-1]:
                color=unique[k]
                if counts[k]<6 or len(chosen)>=14: break
                if any(np.max(np.abs(color-q))<24 for q in chosen): continue
                chosen.append(color)
                cm=cv2.inRange(crop,np.maximum(0,color-13).astype(np.uint8),
                               np.minimum(255,color+13).astype(np.uint8))
                cs,_=cv2.findContours(cm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in cs:
                    cx,cy,cw,ch=cv2.boundingRect(c)
                    clipped=cx<=1 or cx+cw>=crop.shape[1]-1
                    if clipped and 5<cw<235 and 15<ch<55:
                        clipped_floors.append(cy+ystart+ch/2)
                        continue
                    b=rectangle(c,'floor')
                    b.x+=xstart;b.y+=ystart
                    if (35<b.w<235 and 15<b.h<43 and b.w/b.h>1.2 and abs(b.angle)<35
                            and cv2.contourArea(c)>.65*b.w*b.h and xstart<b.x<xend):
                        if not any(abs(b.x-q.x)<4 and abs(b.y-q.y)<5 for q in floors):
                            floors.append(b)
        target=min(floors,key=lambda b:b.y,default=None)
        support_clipped=bool(clipped_floors and (target is None or min(clipped_floors)<target.y+12))
        if support_clipped:target=None
        if target:self.previous_target=(timestamp,target)
        if target is None and (not support_clipped or not floors):
            # First floor: a dark platform, before the colored floors exist.
            bm=((g<95)&(np.indices(g.shape)[0]>int(h*.55))).astype(np.uint8)*255
            bm=cv2.morphologyEx(bm,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            cs,_=cv2.findContours(bm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            bases=[rectangle(c,'base') for c in cs]
            target=next((b for b in bases if 140<b.w<235 and 8<b.h<30
                         and abs(b.x-w/2)<.06*w and b.y>.82*h and abs(b.angle)<2),None)
            if target:support_clipped=False
        # Exclude the smaller rectangle formed by the two suspension wires.
        if target:
            blocks=[b for b in blocks if b.y<target.y-12]
        block=None
        if blocks:
            if self.previous and timestamp-self.previous[0]<.15:
                old=self.previous[1]
                block=min(blocks,key=lambda b:abs(b.x-old.x)+.3*abs(b.y-old.y)+3*abs(b.w-old.w))
            else:
                block=max(blocks,key=lambda b:b.w*b.h)
            self.previous=(timestamp,block)
        elif self.previous and timestamp-self.previous[0]>.15:
            self.previous=None
        # Locate a slightly larger dark rectangle surrounding the top floor.
        # Its center is the game's explicit aiming cue when present.
        if target:
            for c in contours:
                q=rectangle(c,'guide')
                if (abs(q.x-target.x)<8 and abs(q.y-target.y)<8
                    and target.w+2<q.w<target.w+18 and target.h+2<q.h<target.h+18):
                    target=Box(q.x,q.y,target.w,target.h,q.angle,.95,'guide')
                    break
        perfect=False
        if self.feedback_template is not None and block is None and target:
            tpl=self.feedback_template
            feedback=g[max(ymin,int(target.y-300)):max(ymin,int(target.y-45)),120:600]
            if feedback.shape[0]>tpl.shape[0] and feedback.shape[1]>tpl.shape[1]:
                perfect=float(cv2.minMaxLoc(cv2.matchTemplate(feedback,tpl,cv2.TM_CCOEFF_NORMED))[1])>.82
        axis_x=None
        lean=target.angle if target else 0.
        if block and target:
            start=max(ymin,int(block.bottom[1]+6))
            end=min(h,int(target.top[1]-5))
            if end-start>95:
                # Window borders can be long vertical lines too. Exclude the
                # non-game chrome and require agreement with the floor angle.
                shaft_left,shaft_right=14,w-14
                shaft=mask[start:end,shaft_left:shaft_right]
                lines=cv2.HoughLinesP(shaft,1,np.pi/720,45,
                                     minLineLength=max(70,int((end-start)*.55)),maxLineGap=10)
                candidates=[]
                if lines is not None:
                    for x1,y1,x2,y2 in lines.reshape(-1,4):
                        dy=float(y2-y1)
                        if abs(dy)>70 and abs(x2-x1)<.65*abs(dy):
                            a=float(x2-x1)/dy
                            b=float(x1+shaft_left)-a*float(y1+start)
                            projected=a*target.top[1]+b
                            line_angle=float(np.rad2deg(np.arctan(-a)))
                            if (abs(projected-target.top[0])<max(65,target.w)
                                    and abs(line_angle-target.angle)<2.5):
                                candidates.append((abs(dy),a,b))
                if candidates:
                    _,a,b=max(candidates)
                    axis_x=a*target.top[1]+b
                    lean=float(np.rad2deg(np.arctan(-a)))
        return Observation(timestamp,block,target,False,perfect,len(floors),axis_x,lean,support_clipped),f


def overlay(frame, obs, state='', prediction=None):
    out=frame.copy()
    for b,color in [(obs.block,(30,160,250)),(obs.target,(80,190,40))]:
        if b:
            cv2.polylines(out,[np.int32(b.points())],True,color,2)
            cv2.circle(out,(round(b.x),round(b.y)),4,color,-1)
    cv2.putText(out,state,(14,55),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,190),2)
    if prediction:
        x=int(prediction.get('landing_x',0))
        cv2.line(out,(x,70),(x,out.shape[0]-20),(200,40,200),1)
        cv2.putText(out,f"error={prediction['error']:.1f}px  sigma={prediction['sigma']:.1f}",
                    (14,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,190),1)
        cv2.putText(out,f"lean={obs.lean:+.2f} deg  correction={prediction.get('correction',0):+.1f}px",
                    (14,105),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,190),1)
    if obs.axis_x is not None and obs.target:
        cv2.circle(out,(round(obs.axis_x),round(obs.target.top[1])),6,(255,60,50),2)
    return out
