import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .vision import Detector,overlay
from .control import Calibration,Controller

ROOT=Path(__file__).resolve().parent.parent


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def detector():
    d=Detector()
    template=ROOT/'assets'/'perfect.png'
    if template.exists(): d.feedback_template=cv2.imread(str(template),cv2.IMREAD_GRAYSCALE)
    return d


def session_dir(base):
    path=Path(base)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path.mkdir(parents=True,exist_ok=False)
    return path


def replay(args):
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {args.video}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    out=session_dir(args.logdir)
    d=detector();ctrl=Controller();ctrl.arm()
    index=0;detected=0;valid=0;modal=0;times=[];writer=None
    with (out/'frames.jsonl').open('w',encoding='utf-8') as log:
        try:
            while True:
                ok,f=cap.read()
                if not ok: break
                t=index/fps;start=time.perf_counter();obs,img=d.detect(f,t)
                if not ctrl.armed and not obs.modal:ctrl.arm()
                ctrl.observe(obs);p=ctrl.propose(t)
                elapsed=time.perf_counter()-start;times.append(elapsed)
                if obs.block:detected+=1
                if not obs.modal:valid+=1
                else:modal+=1
                log.write(json.dumps({'frame':index,**obs.dict(),'proposal':p,'state':ctrl.state})+'\n')
                if args.annotated:
                    if writer is None:
                        writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(img.shape[1],img.shape[0]))
                        if not writer.isOpened():raise RuntimeError('Could not create annotated video')
                    writer.write(overlay(img,obs,ctrl.state,p))
                index+=1
        finally:
            cap.release()
            if writer:writer.release()
    summary={'source':str(Path(args.video).resolve()),'frames':index,'fps':fps,
             'frames_with_block':detected,'non_modal_frames':valid,'modal_frames':modal,
             'processing_ms_median':float(np.median(times)*1000),'processing_ms_p95':float(np.percentile(times,95)*1000),
             'note':'Detection counts are not recall. Replay does not test counterfactual heights or live input latency.'}
    write_json(out/'summary.json',summary)
    print(json.dumps({'output':str(out),**summary},ensure_ascii=False,indent=2))


def configure(args,window):
    cfg_path=Path(args.config)
    config=json.loads(cfg_path.read_text('utf-8')) if cfg_path.exists() else {}
    rect=window.rect();left,top,right,bottom=rect
    if config.get('window_rect')!=list(rect) or args.setup:
        config.update(window_rect=list(rect),region=list(rect),
                      click=[round(left+.14*(right-left)),round(top+.30*(bottom-top))],window_title=window.title)
        if args.setup:
            import dxcam
            cam=dxcam.create(output_idx=args.output,output_color='BGR')
            frame=cam.grab(region=rect,new_frame_only=False);cam.release()
            if frame is None:raise RuntimeError('No setup frame')
            cv2.namedWindow('Select game area - ENTER to accept',cv2.WINDOW_NORMAL)
            x,y,w,h=cv2.selectROI('Select game area - ENTER to accept',frame,False)
            cv2.destroyAllWindows()
            if w<200 or h<300: raise RuntimeError('Selection cancelled or too small')
            config['region']=[left+x,top+y,left+x+w,top+y+h]
            point=[]
            def choose(event,x1,y1,flags,param):
                if event==cv2.EVENT_LBUTTONDOWN:point.append((left+x+x1,top+y+y1))
            cv2.namedWindow('Click a safe gameplay point',cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('Click a safe gameplay point',choose)
            cv2.imshow('Click a safe gameplay point',frame[y:y+h,x:x+w])
            while not point:
                if cv2.waitKey(20)==27:raise RuntimeError('Setup cancelled')
            cv2.destroyAllWindows();config['click']=list(point[0])
    config['output']=args.output
    write_json(cfg_path,config)
    return config


def live(args):
    from .windows import find_window,ExclusiveWindow
    window=find_window(args.title)
    with ExclusiveWindow(window.hwnd):
        live_session(args,window)


def live_session(args,window):
    from .windows import Capture,Scheduler,Hotkeys
    config=configure(args,window)
    region=config['region'];point=tuple(config['click'])
    if not (region[0]<=point[0]<region[2] and region[1]<=point[1]<region[3]):
        raise ValueError('Click point must be inside selected game area')
    cal=Calibration.from_dict(config.get('calibration',{}))
    ctrl=Controller(cal);d=detector()
    out=session_dir(args.logdir)
    capture=Capture(region,args.output,args.fps)
    scheduler=Scheduler(lambda:window.click(point),lambda:window.check(point))
    try:hotkeys=Hotkeys()
    except Exception:
        scheduler.close();capture.close();raise
    latest_t=-1.;start=time.perf_counter();last_preview=0.;last_jpeg=0.;last_status=0.
    count=0;times=[];ring=deque(maxlen=24);saved_failure=False;pred=None;signature=None
    print(f'Logs: {out}\nF8: start/pause | Esc: stop. Focus the game before F8.',flush=True)
    print('Calibration uses the first 3 observed landings.' if args.mode=='calibrate' else 'Initially paused.',flush=True)
    log=(out/'frames.jsonl').open('w',encoding='utf-8')
    events=(out/'events.jsonl').open('w',encoding='utf-8')
    error=None
    try:
        while True:
            now=time.perf_counter()
            if args.seconds and now-start>=args.seconds:break
            if capture.error:raise RuntimeError(capture.error)
            keys=hotkeys.poll()
            if 'ESC' in keys:scheduler.cancel();break
            if 'F8' in keys:
                if ctrl.armed:scheduler.cancel();ctrl.pause()
                elif not window.check(point):window.park(point);ctrl.arm();print('ARMED',flush=True)
                else:print('Focus the unchanged game window, then press F8.',flush=True)
            if args.auto_arm and not ctrl.armed and now-start<3 and not window.check(point):
                # Explicit CLI flag used for controlled local acceptance runs.
                window.park(point);ctrl.arm();args.auto_arm=False;print('ARMED (automatic test)',flush=True)
            reason=window.check(point)
            if reason:
                scheduler.cancel()
                if ctrl.armed:ctrl.pause(reason)
                # DXcam sees the desktop. Pixels of an overlapping application
                # must never become a game dialog or alter the vision history.
                for event in ctrl.events:
                    events.write(json.dumps(event)+'\n');events.flush();print(event,flush=True)
                ctrl.events.clear()
                if now-last_status>1:
                    write_json(out/'status.json',{'state':ctrl.state,'reason':reason,'t':now})
                    last_status=now
                time.sleep(.02);continue
            result=scheduler.take()
            if result:
                if 'fired' in result:
                    if ctrl.armed and ctrl.state=='TRACK':ctrl.clicked(result['fired'],result['proposal'])
                    else:events.write(json.dumps({'type':'late_result',**result})+'\n')
                elif 'error' in result:ctrl.pause(result['error'])
                else:events.write(json.dumps({'type':'skipped',**result})+'\n')
            value=capture.latest
            if value is None or value[1]<=latest_t:
                if ctrl.armed and now-latest_t>.12:
                    scheduler.cancel()
                    if latest_t>0:ctrl.pause('capture_stalled')
                time.sleep(.002);continue
            frame,latest_t=value
            new_signature=frame[::4,::4]
            if signature is not None and np.array_equal(new_signature,signature):
                if ctrl.armed and ctrl.obs and now-ctrl.obs.t>.15:
                    scheduler.cancel();ctrl.pause('game_frame_stalled')
                time.sleep(.001);continue
            signature=new_signature.copy()
            began=time.perf_counter();obs,img=d.detect(frame,latest_t)
            # A scheduler may have fired while vision was processing this frame.
            result=scheduler.take()
            if result:
                if 'fired' in result and ctrl.armed and ctrl.state=='TRACK':ctrl.clicked(result['fired'],result['proposal'])
                elif 'error' in result:ctrl.pause(result['error'])
                else:events.write(json.dumps({'type':'scheduler_result',**result})+'\n')
            ctrl.observe(obs)
            pred=ctrl.propose(time.perf_counter())
            if pred:scheduler.replace(pred)
            else:scheduler.cancel()
            elapsed=time.perf_counter()-began;times.append(elapsed);count+=1
            log.write(json.dumps({**obs.dict(),'state':ctrl.state,'reason':ctrl.wait_reason,'proposal':pred,'processing_ms':elapsed*1000})+'\n')
            if now-last_status>1:
                write_json(out/'status.json',{'state':ctrl.state,'reason':ctrl.wait_reason,'t':now,
                                            'lean':obs.lean,'observed_landings':ctrl.landings})
                last_status=now
            for event in ctrl.events:
                events.write(json.dumps(event)+'\n');events.flush()
                if event['type'] in ('click','landing','pause'):print(event,flush=True)
            ctrl.events.clear()
            if now-last_jpeg>.12:
                ok,jpg=cv2.imencode('.jpg',overlay(img,obs,f'{ctrl.state}: {ctrl.wait_reason}',pred),[cv2.IMWRITE_JPEG_QUALITY,80])
                if ok:ring.append((count,jpg))
                last_jpeg=now
            if obs.modal and not saved_failure:
                for n,jpg in ring:(out/f'context_{n:06d}.jpg').write_bytes(jpg.tobytes())
                saved_failure=True
            if not obs.modal:saved_failure=False
            if args.preview and now-last_preview>.08:
                cv2.imshow('Tower Bot - preview (keep game focused)',cv2.resize(overlay(img,obs,f'{ctrl.state}: {ctrl.wait_reason}',pred),(360,round(img.shape[0]/2))))
                cv2.waitKey(1);last_preview=now
            if args.mode=='calibrate' and ctrl.landings>=3:
                scheduler.cancel();ctrl.pause('calibration_complete');break
            if args.stop_on_dialog and obs.modal and now-start>2:break
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        scheduler.close();capture.close();hotkeys.close();cv2.destroyAllWindows();log.close();events.close()
        config['calibration']=ctrl.cal.dict();write_json(args.config,config)
        elapsed=time.perf_counter()-start
        summary={'frames':count,'elapsed_s':elapsed,'processed_fps':count/max(elapsed,.001),
                 'processing_ms_median':float(np.median(times)*1000) if times else None,
                 'processing_ms_p95':float(np.percentile(times,95)*1000) if times else None,
                 'observed_landings':ctrl.landings,'calibration':ctrl.cal.dict(),'error':error,
                 'score_note':'Observed landings are not a verified HUD score; inspect final screenshot.'}
        write_json(out/'summary.json',summary)
        for n,jpg in list(ring)[-8:]:(out/f'final_{n:06d}.jpg').write_bytes(jpg.tobytes())
        print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description='Vision-only WeChat tower player')
    sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('replay');r.add_argument('video');r.add_argument('--annotated',action='store_true')
    r.add_argument('--logdir',default=str(ROOT/'runs'))
    for name in ('play','calibrate'):
        q=sub.add_parser(name)
        q.add_argument('--title',default='今天盖楼了吗')
        q.add_argument('--config',default=str(ROOT/'config.json'))
        q.add_argument('--logdir',default=str(ROOT/'runs'))
        q.add_argument('--fps',type=int,default=120)
        q.add_argument('--output',type=int,default=0)
        q.add_argument('--setup',action='store_true')
        q.add_argument('--preview',action='store_true')
        q.add_argument('--seconds',type=float,default=0)
        q.add_argument('--auto-arm',action='store_true')
        q.add_argument('--stop-on-dialog',action='store_true')
    args=p.parse_args()
    if args.mode=='replay':replay(args)
    else:live(args)
