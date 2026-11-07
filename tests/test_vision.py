from pathlib import Path
import unittest
import cv2
from towerbot.vision import Detector


class VisionTests(unittest.TestCase):
    def frame(self,name):
        image=cv2.imread(str(Path(__file__).parent/'fixtures'/f'{name}.png'))
        self.assertIsNotNone(image)
        return image

    def test_top_floor_not_lower_floor(self):
        o,_=Detector().detect(self.frame('early'),0)
        self.assertFalse(o.modal)
        self.assertIsNotNone(o.block);self.assertIsNotNone(o.target)
        self.assertAlmostEqual(o.block.x,347.5,delta=2)
        self.assertAlmostEqual(o.target.x,359,delta=2)
        self.assertAlmostEqual(o.target.y,1131,delta=3)

    def test_coordinates_are_scale_independent(self):
        image=self.frame('tilted_video')
        a,_=Detector().detect(image,0)
        b,_=Detector().detect(cv2.resize(image,None,fx=1.5,fy=1.5),0)
        self.assertIsNotNone(a.block);self.assertIsNotNone(b.block)
        self.assertLess(abs(a.block.x-b.block.x),2)
        self.assertLess(abs(a.target.x-b.target.x),3)

    def test_dialog_is_not_a_moving_block(self):
        o,_=Detector().detect(self.frame('modal'),0)
        self.assertTrue(o.modal);self.assertIsNone(o.block);self.assertIsNone(o.target)

    def test_crane_axis_and_lean_direction(self):
        image=self.frame('leaning_live')
        a,_=Detector().detect(image,0)
        b,_=Detector().detect(cv2.flip(image,1),0)
        self.assertGreater(a.lean,4);self.assertLess(a.lean,6)
        self.assertLess(b.lean,-4);self.assertGreater(b.lean,-6)
        self.assertGreater(a.target.x-a.axis_x,30)
        self.assertLess(b.target.x-b.axis_x,-30)
        self.assertAlmostEqual(a.target.y,594,delta=3)

    def test_support_and_crane_are_detected_near_both_edges(self):
        import numpy as np
        a,image=Detector().detect(self.frame('leaning_live'),0)
        shift=round(645-a.target.x)
        translated=cv2.warpAffine(image,np.float32([[1,0,shift],[0,1,0]]),
                                  (image.shape[1],image.shape[0]),borderValue=(255,255,255))
        for frame in (translated,cv2.flip(translated,1)):
            o,_=Detector().detect(frame,0)
            self.assertIsNotNone(o.block);self.assertIsNotNone(o.target)
            self.assertIsNotNone(o.axis_x)
            self.assertAlmostEqual(abs(o.lean),a.lean,delta=.6)
            self.assertAlmostEqual(o.target.y,a.target.y,delta=2)
            self.assertLess(min(abs(o.target.x-645),abs(o.target.x-74)),2)

    def test_clipped_top_does_not_select_a_lower_floor_or_window_border(self):
        o,_=Detector().detect(self.frame('clipped_live'),0)
        self.assertTrue(o.support_clipped)
        self.assertIsNone(o.target)
        self.assertIsNone(o.axis_x)

    def test_new_round_base_and_ground_line_are_not_clipped_floors(self):
        image=self.frame('base_live')
        o,_=Detector().detect(image,0)
        self.assertFalse(o.support_clipped)
        self.assertEqual(o.target.kind,'base')
        self.assertAlmostEqual(o.target.x,360,delta=2)
        cv2.rectangle(image,(281,1100),(439,1125),(130,110,90),-1)
        o,_=Detector().detect(image,0)
        self.assertFalse(o.support_clipped)
        self.assertAlmostEqual(o.target.y,1112.5,delta=2)

    def test_suspension_window_and_cutoff_slab_do_not_become_the_block(self):
        d=Detector()
        for i,name in enumerate(('valid_slab_high','edge_slab','wire_false_positive','valid_slab_high')):
            o,_=d.detect(self.frame(name),i*.04)
            if name=='valid_slab_high':
                self.assertIsNotNone(o.block)
                self.assertAlmostEqual(o.block.y,308.28,delta=1)
                self.assertGreater(o.block.w,57)
            elif name=='edge_slab':
                self.assertIsNone(o.block)
            elif o.block:
                # A real slab may already be fully visible below the wires;
                # the 22 px suspension window must never be returned.
                self.assertGreater(o.block.y,300)
                self.assertGreater(o.block.h,24)


if __name__=='__main__':unittest.main()
