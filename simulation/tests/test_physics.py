"""Integration checks for CAD-derived physics and the intended sensor limitations."""
import sys,math,unittest,json
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ros2_ws/src/vbmd_description/scripts'))
from sim_core import RobotSim,CAM_WIDTH,CAM_HEIGHT
class PhysicsTests(unittest.TestCase):
 def setUp(self):self.sim=RobotSim(noise=False)
 def tearDown(self):self.sim.close()
 def settle(self):
  for _ in range(100):self.sim.advance(.01)
 def test_urdf_is_importable_and_retains_visuals(self):
  m=mujoco.MjModel.from_xml_path(str(ROOT/'ros2_ws/src/vbmd_description/urdf/vbmd.urdf'))
  self.assertGreater(m.nmesh,50);self.assertEqual(m.njnt,4)
 def test_stable_ground_contact_and_sonar_directions(self):
  self.settle();s=self.sim
  self.assertTrue(np.isfinite(s.data.qpos).all());self.assertLess(abs(s.data.qpos[2]),.001);self.assertGreaterEqual(s.data.ncon,4)
  self.assertAlmostEqual(s.sample_sonar('front'),.6454,places=3)
  self.assertAlmostEqual(s.sample_sonar('rear'),1.8854,places=3)
  for side in ['left','right']:self.assertAlmostEqual(s.sample_sonar(side),1.398,delta=.01)
  times=sorted(t for t in s.range_times.values() if t is not None)
  self.assertEqual(len(times),4)
  self.assertTrue(all(b-a>=.059 for a,b in zip(times,times[1:])))
 def test_forward_motion_and_command_watchdog(self):
  self.settle();s=self.sim;x=s.data.qpos[0]
  for _ in range(100):s.set_twist(.10,0);s.advance(.01)
  self.assertGreater(s.data.qpos[0]-x,.07);self.assertLess(abs(s.data.qpos[1]),.01)
  for _ in range(100):s.advance(.01)
  self.assertTrue(np.allclose(s.data.ctrl,0));self.assertLess(np.linalg.norm(s.data.qvel[:3]),.01)
 def test_skid_steer_turns_without_tipping(self):
  self.settle();s=self.sim
  for _ in range(200):s.set_twist(0,1);s.advance(.01)
  q=s.data.qpos[3:7];yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
  self.assertGreater(yaw,.2);self.assertLess(abs(q[1]),.05);self.assertLess(abs(q[2]),.05)
 def test_only_rgb_camera_and_four_sonar_observations(self):
  self.settle();s=self.sim;image=s.camera()
  self.assertEqual(image.shape,(CAM_HEIGHT,CAM_WIDTH,3));self.assertGreater(float(image.std()),15)
  self.assertEqual(set(s.ranges),{'front','rear','left','right'})
  self.assertEqual(s.model.ncam,1)
  self.assertEqual(s.model.nsensor,0) # Cone ultrasound is computed in Python; no hidden IMU/depth/lidar.
if __name__=='__main__':unittest.main(verbosity=2)
