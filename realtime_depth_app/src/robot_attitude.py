"""Gyro integration, accelerometer tilt and tilt-compensated magnetic yaw.
Does not consume simulator orientation quaternions or scene poses.
Robot axes are +X forward, +Y left, +Z up. World +Y is magnetic north in the test world.
"""
import math
import numpy as np

def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi
class AttitudeFilter:
    def __init__(self):
        self.roll=self.pitch=self.yaw=0.;self.last_imu=None;self.last_mag=None
        self.accel=np.zeros(3);self.gyro=np.zeros(3);self.field=np.zeros(3)
        self.tilt_ready=False;self.heading_ready=False
    def update_imu(self,t,gyro,accel):
        gyro=np.asarray(gyro,float);accel=np.asarray(accel,float)
        if not np.isfinite(gyro).all() or not np.isfinite(accel).all():return
        dt=t-self.last_imu if self.last_imu is not None else 0.
        self.last_imu=t;self.gyro=gyro;self.accel=accel
        if 0<dt<.25:
            sr,cr=math.sin(self.roll),math.cos(self.roll);cp=math.cos(self.pitch)
            cp=math.copysign(max(.15,abs(cp)),cp)
            gx,gy,gz=gyro
            self.roll=wrap(self.roll+dt*(gx+math.tan(self.pitch)*(sr*gy+cr*gz)))
            self.pitch=wrap(self.pitch+dt*(cr*gy-sr*gz))
            self.yaw=wrap(self.yaw+dt*(sr*gy+cr*gz)/cp)
        norm=np.linalg.norm(accel)
        if 7.<norm<12.5:
            ax,ay,az=accel;roll=math.atan2(ay,az);pitch=math.atan2(-ax,math.hypot(ay,az))
            alpha=math.exp(-max(dt,0.)/.5) if self.tilt_ready else 0.
            self.roll=wrap(roll+alpha*wrap(self.roll-roll));self.pitch=wrap(pitch+alpha*wrap(self.pitch-pitch));self.tilt_ready=True
    def update_mag(self,t,field):
        field=np.asarray(field,float)
        if not np.isfinite(field).all() or not 1e-6<np.linalg.norm(field)<1e-3:return
        if not self.tilt_ready:return
        dt=t-self.last_mag if self.last_mag is not None else 0.;self.last_mag=t;self.field=field
        mx,my,mz=field;sr,cr=math.sin(self.roll),math.cos(self.roll);sp,cp=math.sin(self.pitch),math.cos(self.pitch)
        hx=mx*cp+my*sr*sp+mz*cr*sp;hy=my*cr-mz*sr
        if math.hypot(hx,hy)<1e-7:return
        measured=wrap(math.pi/2-math.atan2(hy,hx))
        gain=1-math.exp(-max(0.,dt)/.8) if self.heading_ready else 1.
        self.yaw=wrap(self.yaw+gain*wrap(measured-self.yaw));self.heading_ready=True
    def snapshot(self):
        return {'roll_deg':math.degrees(self.roll),'pitch_deg':math.degrees(self.pitch),'yaw_deg':math.degrees(self.yaw),'heading_deg':(90-math.degrees(self.yaw))%360,'acceleration_m_s2':self.accel.tolist(),'angular_velocity_deg_s':np.degrees(self.gyro).tolist(),'magnetic_field_uT':(self.field*1e6).tolist(),'ready':self.tilt_ready and self.heading_ready}
