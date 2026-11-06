"""Experimental flat-floor visual odometry and ultrasonic occupancy/frontier mapping.
Metric scale comes from configured camera height and intrinsics, not simulator poses.
No loop closure: this is local mapping with drift, not a complete SLAM implementation.
"""
import math
from collections import deque
import cv2
import numpy as np

SENSOR_POSES={'front':(.0846,0,0),'left':(0,.0676,math.pi/2),'rear':(-.0846,0,math.pi),'right':(0,-.0676,-math.pi/2)}
def rotation(r,p,y):
 cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
 return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])

class GroundVisualOdometry:
 def __init__(self):
  self.k=None;self.xy=np.zeros(2);self.previous=None;self.last_good=None;self.inliers=0;self.sigma=.01;self.samples=0
 def ground_vectors(self,points,rpy):
  fx,fy,cx,cy=self.k
  u,v=points[:,0],points[:,1]
  body=np.column_stack([np.ones_like(u),-(u-cx)/fx,-(v-cy)/fy])
  R=rotation(*rpy);ray=body@R.T;origin=R@np.array([.0757,0,.072])
  t=-origin[2]/np.minimum(ray[:,2],-1e-7)
  result=origin+ray*t[:,None]
  valid=(ray[:,2]<-.04)&(t>.03)&(t<1.5)
  return result[:,:2],valid
 def translation(self,p0,p1,rpy0,rpy1,dt):
  a,va=self.ground_vectors(p0,rpy0);b,vb=self.ground_vectors(p1,rpy1)
  delta=(a-b)[va&vb]
  if len(delta)<6:return None,0
  median=np.median(delta,axis=0);err=np.linalg.norm(delta-median,axis=1);good=delta[err<.015]
  if len(good)<6 or len(good)<len(delta)*.45:return None,len(good)
  shift=np.median(good,axis=0)
  if np.linalg.norm(shift)>.24*dt+.008:return None,len(good)
  return shift,len(good)
 def update(self,bgr,t,rpy):
  gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);current=(gray,t,tuple(rpy));old=self.previous;self.previous=current
  if self.k is None or old is None:return False
  previous,pt,prpy=old;dt=t-pt
  if not 0<dt<.5:return False
  mask=np.zeros_like(gray);h,w=gray.shape;mask[int(.60*h):int(.96*h),int(.04*w):int(.96*w)]=255
  points=cv2.goodFeaturesToTrack(previous,250,.01,6,mask=mask)
  if points is None or len(points)<6:return False
  nxt,status,error=cv2.calcOpticalFlowPyrLK(previous,gray,points,None,winSize=(21,21),maxLevel=3)
  if nxt is None:return False
  back,ok,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None,winSize=(21,21),maxLevel=3)
  if back is None:return False
  good=status.ravel().astype(bool)&ok.ravel().astype(bool)&(np.linalg.norm(back-points,axis=2).ravel()<1.)
  a=points.reshape(-1,2)[good];b=nxt.reshape(-1,2)[good]
  if len(a)<6:return False
  shift,count=self.translation(a,b,prpy,rpy,dt);self.inliers=count
  if shift is None:return False
  self.xy+=shift;self.last_good=t;self.samples+=1;self.sigma=math.sqrt(self.sigma**2+.0005**2+(.03*np.linalg.norm(shift))**2)
  return True

class ExplorationMap:
 def __init__(self,resolution=.05,size=120):
  self.res=resolution;self.size=size;self.origin=-size*resolution/2;self.logodds=np.zeros((size,size),np.float32);self.path=[];self.updates=0
 def cell(self,x,y):return (int((y-self.origin)/self.res),int((x-self.origin)/self.res))
 def inside(self,i,j):return 0<=i<self.size and 0<=j<self.size
 def observe(self,pose,side,distance,maximum=4.,fov=math.pi/12):
  x,y,yaw=pose;sx,sy,heading=SENSOR_POSES[side];c,s=math.cos(yaw),math.sin(yaw)
  ox=x+c*sx-s*sy;oy=y+s*sx+c*sy;hit=math.isfinite(distance) and .02<=distance<maximum-.02
  if math.isnan(distance) or distance<.02:return
  distance=min(distance,maximum)
  for da in np.linspace(-fov/2,fov/2,9):
   a=yaw+heading+da
   for length in np.arange(.02,max(.03,distance-.045),self.res*.65):
    i,j=self.cell(ox+length*math.cos(a),oy+length*math.sin(a))
    if self.inside(i,j):self.logodds[i,j]=max(-3.,self.logodds[i,j]-.12)
   if hit:
    i,j=self.cell(ox+distance*math.cos(a),oy+distance*math.sin(a))
    if self.inside(i,j):self.logodds[i,j]=min(3.,self.logodds[i,j]+.65)
  i,j=self.cell(x,y)
  if self.inside(i,j):
   cv2.circle(self.logodds,(j,i),2,-2.,-1)
  if not self.path or math.dist(self.path[-1],[x,y])>.02:self.path.append([x,y])
  self.updates+=1
 def frontiers(self,pose):
  free=self.logodds<-.3;unknown=np.abs(self.logodds)<.08;occupied=self.logodds>.45
  blocked=cv2.dilate(occupied.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
  safe=free&~blocked;near_unknown=cv2.dilate(unknown.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);frontier=safe&near_unknown
  i,j=self.cell(*pose[:2])
  if not self.inside(i,j):return []
  # The robot's current cell is free even when conservative inflation touches its boundary.
  queue=deque([(i,j)]);seen={(i,j)};parents={};targets=[]
  while queue:
   a,b=queue.popleft()
   if frontier[a,b] and math.hypot(a-i,b-j)>4:targets.append((a,b))
   for u,v in [(a+1,b),(a-1,b),(a,b+1),(a,b-1)]:
    if self.inside(u,v) and safe[u,v] and (u,v) not in seen:seen.add((u,v));parents[(u,v)]=(a,b);queue.append((u,v))
  candidates=[]
  for goal in sorted(targets,key=lambda q:-math.hypot(q[0]-i,q[1]-j))[:40]:
   route=[goal]
   while route[-1] in parents and route[-1]!=(i,j):route.append(parents[route[-1]])
   if len(route)<3:continue
   step=route[max(0,len(route)-5)];gx=self.origin+(step[1]+.5)*self.res;gy=self.origin+(step[0]+.5)*self.res
   angle=math.atan2(gy-pose[1],gx-pose[0])
   if all(abs((angle-a+math.pi)%(2*math.pi)-math.pi)>.35 for a in candidates):candidates.append(angle)
   if len(candidates)>=5:break
  return candidates
 def image(self,pose,frontiers=None):
  img=np.full((self.size,self.size,3),(69,57,45),np.uint8)
  img[self.logodds<-.3]=(222,225,221);img[self.logodds>.45]=(55,70,220)
  if len(self.path)>1:
   pts=np.array([[self.cell(x,y)[1],self.cell(x,y)[0]] for x,y in self.path],np.int32);cv2.polylines(img,[pts],False,(130,150,30),1)
  i,j=self.cell(*pose[:2]);yaw=pose[2]
  if self.inside(i,j):
   cv2.circle(img,(j,i),2,(80,220,70),-1);cv2.line(img,(j,i),(round(j+6*math.cos(yaw)),round(i+6*math.sin(yaw))),(80,220,70),1)
  return cv2.flip(cv2.resize(img,(360,360),interpolation=cv2.INTER_NEAREST),0)
