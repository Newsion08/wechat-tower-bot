import threading
import time
import unittest
from towerbot.control import Controller,Track,Calibration,PeriodicTrack
from towerbot.vision import Box,Observation
from towerbot.windows import Scheduler


def obs(t,x,target=360,y=300):
    return Observation(t,Box(x,y,144,28,0,1,'block'),Box(target,610,144,28,0,1,'floor'),False)


class MotionTests(unittest.TestCase):
    def test_accelerating_prediction_and_duplicate_frames(self):
        track=Track()
        for i in range(15):
            t=i/120;track.add(t,100+300*t+100*t*t,200)
            track.add(t,-999,-999)
        fit=track.fit()
        predicted,_=fit.predict(.06)
        t=14/120+.06
        self.assertAlmostEqual(predicted,100+300*t+100*t*t,places=5)
        self.assertEqual(len(track.points),15)

    def test_future_click_compensates_latency(self):
        ctrl=Controller(Calibration(latency=.03,drift_ratio=0),require_periodic=False)
        ctrl.arm()
        for i in range(16):
            t=10+i/120;ctrl.observe(obs(t,310+240*(t-10)))
        proposal=ctrl.propose(t+.005)
        self.assertIsNotNone(proposal)
        self.assertAlmostEqual(proposal['release']-proposal['deadline'],.03)
        self.assertLess(abs(proposal['error']),1)
        ctrl.clicked(proposal['deadline'],proposal)
        self.assertIsNone(ctrl.propose(proposal['deadline']))
        with self.assertRaises(RuntimeError):ctrl.clicked(proposal['deadline'],proposal)

    def test_past_crossing_is_not_clicked(self):
        ctrl=Controller(Calibration(drift_ratio=0),require_periodic=False);ctrl.arm()
        for i in range(16):
            t=i/120;ctrl.observe(obs(t,370+300*t))
        self.assertIsNone(ctrl.propose(t))

    def test_reverse_motion_and_moving_target(self):
        ctrl=Controller(Calibration(drift_ratio=0),require_periodic=False);ctrl.arm()
        for i in range(20):
            t=i/120;ctrl.observe(obs(t,435-300*t,360+10*t))
        proposal=ctrl.propose(t)
        self.assertIsNotNone(proposal)
        self.assertLess(proposal['velocity'],0)
        self.assertGreater(proposal['target_x'],360+10*t)

    def test_modal_and_stale_frames_prevent_input(self):
        ctrl=Controller();ctrl.arm()
        for i in range(20):ctrl.observe(obs(i/120,320+160*i/120))
        self.assertIsNone(ctrl.propose(2))
        ctrl.observe(Observation(3,None,None,True))
        self.assertFalse(ctrl.armed)
        self.assertIsNone(ctrl.propose(3))

    def test_manual_falling_block_not_released_twice(self):
        ctrl=Controller();ctrl.arm()
        for i in range(12):ctrl.observe(obs(i/60,360,y=300+250*i/60))
        self.assertEqual(ctrl.state,'FALLING')
        self.assertIsNone(ctrl.propose(12/60))

    def test_periodic_prediction_uses_observed_speed(self):
        import math
        p=PeriodicTrack()
        for i in range(145):
            t=i/120;p.add(t,360+125*math.sin(5.2*t),300)
        f=p.fit()
        self.assertAlmostEqual(f.predict(.035)[0],360+125*math.sin(5.2*(1.2+.035)),places=5)

    def test_offscreen_arc_does_not_discard_the_same_slab(self):
        import math
        ctrl=Controller(Calibration(drift_ratio=0));ctrl.arm();proposals=[]
        for i in range(480):
            t=i/120;x=50+140*math.sin(6*t)
            o=obs(t,x,target=50)
            if x<73:o.block=None  # Entire visible segment is shorter than 0.8 s.
            ctrl.observe(o);p=ctrl.propose(t+.005)
            if o.block is None:self.assertIsNone(p)
            if p:proposals.append(p)
        self.assertGreater(len(proposals),0)
        self.assertGreater(ctrl.periodic_track.points[-1][0]-ctrl.periodic_track.points[0][0],.8)

    def test_tilted_rail_motion_is_not_a_release(self):
        import math
        ctrl=Controller(require_periodic=False);ctrl.arm()
        for i in range(35):
            t=i/120;x=450-800*t
            o=obs(t,x,y=300+math.tan(math.radians(-9))*(x-450))
            o.block.angle=-9;ctrl.observe(o)
        self.assertEqual(ctrl.state,'TRACK')
        self.assertIsNone(ctrl.clicked_at)

    def test_waits_for_post_impact_tower_motion_to_settle(self):
        import math
        ctrl=Controller(Calibration(drift_ratio=0));ctrl.arm()
        for i in range(150):
            t=i/120;ctrl.observe(obs(t,360+170*math.sin(6*t),360+100*t))
        self.assertIsNone(ctrl.propose(t+.005))
        self.assertEqual(ctrl.wait_reason,'tower_still_moving')
        proposals=[]
        for i in range(150,480):
            t=i/120;ctrl.observe(obs(t,360+170*math.sin(6*t),480))
            p=ctrl.propose(t+.005)
            if p:proposals.append(p)
        self.assertTrue(proposals)

    def test_recorded_edge_stall_has_a_visible_release_opportunity(self):
        import json
        from pathlib import Path
        data=json.loads((Path(__file__).parent/'fixtures'/'edge_stall.json').read_text())
        ctrl=Controller(Calibration.from_dict(data['calibration']));ctrl.arm();proposals=[]
        for record in data['observations']:
            for key in ('block','target'):
                if record[key]:record[key]=Box(**record[key])
            o=Observation(**record);ctrl.observe(o)
            p=ctrl.propose(o.t+.010)
            if p:proposals.append(p)
        self.assertGreater(len(proposals),0)
        self.assertTrue(all(abs(p['correction'])+abs(p['error'])+2*p['sigma']<.5*p['width'] for p in proposals))

    def test_restoring_offsets_have_correct_sign_and_overlap(self):
        for sign in (-1,1):
            ctrl=Controller(Calibration(drift_ratio=0),require_periodic=False);ctrl.arm()
            for i in range(20):
                t=i/120;o=obs(t,360+sign*(75-300*t),360+sign*40)
                o.axis_x=360;o.lean=sign*4;ctrl.observe(o)
            p=ctrl.propose(t)
            self.assertIsNotNone(p)
            self.assertLess(sign*p['correction'],0)
            self.assertLessEqual(abs(p['correction']),.18*144)

    def test_intentional_correction_is_not_learned_as_input_bias(self):
        import numpy as np
        c=Controller();c.clicked_at=1.;c.drop_origin=(360,300);c.drop_target=(360,610)
        c.drop_balance={'correction':-20.,'lean':5.,'axis_error':40.,'width':144.}
        c.release_velocity=300
        c.drop=[(1+t,340.,300+2000*t*t,360.) for t in np.linspace(.10,.38,20)]
        o=obs(2.,360,target=340);o.lean=3.;o.axis_x=320
        c.finish_drop(o)
        self.assertAlmostEqual(c.cal.bias_right,0)
        self.assertEqual(c.cal.balance_samples,1)
        self.assertEqual(c.landings,1)
        self.assertTrue(any(e['type']=='balance_response' for e in c.events))


class SchedulerTests(unittest.TestCase):
    def test_cancel_and_single_fire(self):
        calls=[]
        s=Scheduler(lambda:calls.append(time.perf_counter()),lambda:None)
        try:
            now=time.perf_counter()
            s.replace({'deadline':now+.04,'observed':now});s.cancel()
            time.sleep(.06);self.assertEqual(calls,[])
            now=time.perf_counter()
            s.replace({'deadline':now+.02,'observed':now})
            time.sleep(.04)
            s.replace({'deadline':time.perf_counter()+.01,'observed':time.perf_counter()})
            time.sleep(.03)
            self.assertEqual(len(calls),1)
            self.assertIn('fired',s.take())
        finally:s.close()

    def test_guard_and_capture_stall(self):
        calls=[]
        s=Scheduler(lambda:calls.append(1),lambda:'window_not_focused')
        try:
            now=time.perf_counter();s.replace({'deadline':now+.01,'observed':now})
            time.sleep(.03);self.assertEqual(calls,[])
            self.assertEqual(s.take()['skipped'],'window_not_focused')
            s.guard=lambda:None
            now=time.perf_counter();s.replace({'deadline':now+.01,'observed':now-1})
            time.sleep(.03);self.assertEqual(s.take()['skipped'],'stale_frame')
        finally:s.close()


if __name__=='__main__':unittest.main()
