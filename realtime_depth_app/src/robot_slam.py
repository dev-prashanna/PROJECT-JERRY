"""Experimental flat-floor visual/compass SLAM and conservative frontier planning.

Metric scale comes from camera calibration and measured mounting height, never
from relative depth, commanded motion, wheel joints, or simulator ground truth.
All arrays use world +X/+Y; image rendering flips Y only at the display boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import heapq
import json
import math
import threading
import time
import multiprocessing as mp
import queue

import cv2
import numpy as np
from scipy import ndimage
from .robot_avoidance import Decision
from .camera import LatestSlot


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def localization_health(snapshot,simulation_time,heading_ready,camera_age,clock_age):
    trusted=snapshot.get('last_good_stamp')
    age=simulation_time-trusted if simulation_time is not None and trusted is not None else math.inf
    fresh=heading_ready and snapshot['age_s']<.30 and camera_age<.65 and clock_age<1.
    return {'trusted_pose_age_s':age,
            'can_bridge_pose':bool(fresh and -.1<=age<=.20 and snapshot.get('plan_ready',False)),
            'can_relocalize':bool(fresh and age>=.80)}


def segment_clear(start,end,safe,origin,resolution):
    count=max(2,int(math.dist(start,end)/resolution*3)+1)
    cells=np.floor((np.linspace(start,end,count)-origin)/resolution).astype(int)
    if not ((cells>=0).all() and (cells[:,0]<safe.shape[1]).all() and (cells[:,1]<safe.shape[0]).all()):return False
    if not safe[cells[:,1],cells[:,0]].all():return False
    for (x,y),(nx,ny) in zip(cells[:-1],cells[1:]):
        if x!=nx and y!=ny and not (safe[y,nx] and safe[ny,x]):return False
    return True


def simplify_route(points,safe,origin,resolution):
    """Remove grid staircases using only collision-checked line-of-sight links."""
    if len(points)<3:return points
    route=[points[0]];index=0
    while index<len(points)-1:
        selected=index+1
        for candidate in range(len(points)-1,index,-1):
            if math.dist(points[index],points[candidate])>.8:continue
            if segment_clear(points[index],points[candidate],safe,origin,resolution):
                selected=candidate;break
        route.append(points[selected]);index=selected
    return route


def route_target(pose,path,grid,origin,resolution):
    """Select a short lookahead only when its entire swept route is known free."""
    free=grid==0
    radius=int(math.ceil(.13/resolution))
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    blocked=ndimage.binary_dilation(grid==100,structure=xx*xx+yy*yy<=radius*radius)
    safe=free&~blocked&(ndimage.distance_transform_edt(free)*resolution>=.11)
    nearest=int(np.argmin([math.dist(pose[:2],p) for p in path]))
    for index in range(min(nearest+4,len(path)-1),nearest,-1):
        target=path[index]
        count=max(2,int(math.dist(pose[:2],target)/resolution*3))
        cells=np.floor((np.linspace(pose[:2],target,count)-origin)/resolution).astype(int)
        if (cells>=0).all() and (cells[:,0]<grid.shape[1]).all() and (cells[:,1]<grid.shape[0]).all() and safe[cells[:,1],cells[:,0]].all():
            return target
    return None


class PathFollower:
    """Continuous arc-length lookahead and turn hysteresis for a fixed goal."""
    def __init__(self):
        self.goal=None
        self.aligning=False

    def command(self,pose,path,goal,safety,navigation=None):
        if tuple(goal)!=self.goal:
            self.goal=tuple(goal);self.aligning=False
        points=np.asarray(path,float)
        if len(points)<2:points=np.array([pose[:2],goal],float)
        vectors=np.diff(points,axis=0)
        lengths=np.linalg.norm(vectors,axis=1)
        fractions=np.clip(np.sum((np.asarray(pose[:2])-points[:-1])*vectors,axis=1)/np.maximum(lengths**2,1e-10),0.,1.)
        projected=points[:-1]+fractions[:,None]*vectors
        nearest=int(np.argmin(np.linalg.norm(projected-pose[:2],axis=1)))
        cumulative=np.r_[0.,np.cumsum(lengths)]
        progress=cumulative[nearest]+fractions[nearest]*lengths[nearest]
        safe=None
        if navigation and 'grid' in navigation:
            grid=navigation['grid'];resolution=navigation['resolution']
            radius=int(math.ceil(.13/resolution));yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
            safe=(grid==0)&~ndimage.binary_dilation(grid==100,structure=xx*xx+yy*yy<=radius*radius)
            safe &= ndimage.distance_transform_edt(grid==0)*resolution>=.11
        target=None
        for lookahead in [.40,.32,.25,.20,.15,.10,.07]:
            along=min(cumulative[-1],progress+lookahead)
            index=min(len(lengths)-1,max(0,int(np.searchsorted(cumulative,along,side='right')-1)))
            point=points[index]+vectors[index]*np.clip((along-cumulative[index])/max(lengths[index],1e-10),0.,1.)
            if safe is not None:
                if not segment_clear(pose[:2],point,safe,np.asarray(navigation['origin']),resolution):continue
            target=point;break
        if target is None:return Decision(0.,0.,'PLANNING','Destination retained; waiting for safe route clearance')
        delta=target-pose[:2]
        error=angle(math.atan2(delta[1],delta[0])-pose[2])
        tangent=vectors[nearest]
        route_error=angle(math.atan2(tangent[1],tangent[0])-pose[2])
        # A wide engage threshold and narrower release threshold prevent
        # alternating rotate/drive commands around one heading threshold.
        if abs(error)>1.05 and abs(route_error)>.70:self.aligning=True
        elif abs(error)<.40 or abs(route_error)<.45:self.aligning=False
        if self.aligning:
            return Decision(0.,float(np.clip(error*1.2,-.40,.40)),'ALIGN','Aligning with the route to the committed destination')
        remaining=math.dist(pose[:2],goal)
        approach=.065
        if cumulative[-1]-progress<.5:
            approach=max(.012,.24*max(0.,remaining-.10))
        linear=min(.065,safety.linear,approach)*max(.20,math.cos(error))
        angular=2.*linear*math.sin(error)/max(.10,float(np.linalg.norm(delta)))
        if abs(angular)>.35:
            linear*=.35/abs(angular);angular=math.copysign(.35,angular)
        return Decision(float(linear),float(np.clip(angular,-.35,.35)),
                        'EXPLORE','Following the fixed destination with continuous steering')


@dataclass
class Calibration:
    k: np.ndarray
    # Optical center in the robot base frame, metres. Supplied Gazebo CAD mount.
    camera_xyz: tuple = (.0757, 0., .072)

    def ground(self, pixels, roll=0., pitch=0.):
        """Intersect optical rays with a level floor; reject sky and distant rays."""
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(self.k).T
        # Optical right/down/forward -> base forward/left/up.
        rays = rays[:, [2, 0, 1]] * [1., -1., -1.]
        cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        tilt = np.array([[cp, sp*sr, sp*cr], [0, cr, -sr], [-sp, cp*sr, cp*cr]])
        rays = rays @ tilt.T
        origin = tilt @ np.asarray(self.camera_xyz)
        scale = -origin[2] / np.where(rays[:, 2] < -1e-5, rays[:, 2], -1e-5)
        points = origin + rays * scale[:, None]
        valid = (rays[:, 2] < -.04) & (scale > 0) & (points[:, 0] > .10) & (points[:, 0] < 1.4)
        return points[:, :2], valid


def consensus_translation(old, new, minimum=8, threshold=.012):
    """Robust metric translation with compass rotation already removed."""
    delta = np.asarray(old) - np.asarray(new)
    if len(delta) < minimum:
        return None, 0
    # Mixed floor/foreground tracks can put the coordinate-wise median between
    # two motions, rejecting dozens of consistent floor tracks. Find the largest
    # translation consensus instead; reject genuinely ambiguous competing modes.
    proposals=delta[np.linspace(0,len(delta)-1,min(64,len(delta)),dtype=int)]
    support=np.linalg.norm(delta[None,:,:]-proposals[:,None,:],axis=2)<threshold
    counts=support.sum(axis=1)
    best=int(np.argmax(counts));center=np.median(delta[support[best]],axis=0)
    keep = np.linalg.norm(delta-center, axis=1) < threshold
    count = int(keep.sum())
    if count < minimum or count < len(delta)*.25:
        return None, count
    competitors=np.linalg.norm(proposals-center,axis=1)>2.5*threshold
    if np.any(competitors) and int(counts[competitors].max())>=count*.85:
        return None,count
    # Reject point clusters that cannot constrain a stable floor transform.
    if np.linalg.norm(np.ptp(np.asarray(new)[keep], axis=0)) < .06:
        return None, count
    return np.median(delta[keep], axis=0), count


def floor_translation(old,new,threshold=.012):
    """Prefer near-floor geometry when distant wall features dominate the ROI."""
    old=np.asarray(old);new=np.asarray(new)
    nearby=(np.linalg.norm(old,axis=1)<.70)&(np.linalg.norm(new,axis=1)<.70)
    if nearby.sum()>=4:
        a,b=old[nearby],new[nearby]
        delta,count=consensus_translation(a,b,minimum=4,threshold=min(threshold,.008))
        if delta is not None and count>=len(a)*.60:
            supporting=b[np.linalg.norm(a-b-delta,axis=1)<min(threshold,.008)]
            # Four points suffice for compass-constrained translation only when
            # they span a plane, rather than one edge or a tiny image patch.
            covariance=np.cov(supporting.T)
            if np.linalg.eigvalsh(covariance).min()>.00015:return delta,count
    return consensus_translation(old,new,minimum=8,threshold=threshold)


class OccupancyMap:
    def __init__(self, size=240, resolution=.05):
        self.size = size
        self.resolution = resolution
        self.origin = np.array([-size*resolution/2]*2)
        self.logodds = np.zeros((size, size), np.float32)
        self.observed = np.zeros((size, size), bool)

    def cell(self, xy):
        return np.floor((np.asarray(xy)-self.origin)/self.resolution).astype(int)

    def world(self, cell):
        return self.origin+(np.asarray(cell)+.5)*self.resolution

    def inside(self, cell):
        return bool(np.all(np.asarray(cell) >= 0) and np.all(np.asarray(cell) < self.size))

    def update(self, pose, readings):
        x, y, yaw = pose
        r = rotation(yaw)
        # Mark only the footprint actually occupied by the robot as free.
        c = self.cell([x, y])
        if self.inside(c):
            mask = np.zeros_like(self.logodds, np.uint8)
            cv2.circle(mask, tuple(c), 2, 1, -1)
            self.logodds[mask > 0] = -3
            self.observed[mask > 0] = True
        mounts = {'front': (.0846, 0., 0.), 'left': (0., .0676, math.pi/2),
                  'rear': (-.0846, 0., math.pi), 'right': (0., -.0676, -math.pi/2)}
        for side, (value, maximum) in readings.items():
            if side not in mounts or math.isnan(value) or value == -math.inf:
                continue
            mx, my, bearing = mounts[side]
            start = np.array([x, y])+r @ [mx, my]
            distance = min(value, maximum, 2.5)
            hit = math.isfinite(value) and value < min(maximum, 2.5)-.03
            # Gazebo's published minimum is a lower bound on all nine beam rays.
            # Integrate the 15-degree cone up to that bound, leaving a 6 cm hit
            # margin. One observation updates each cell once regardless of rays.
            free_distance=max(0.,distance-(.06 if hit else .02))
            samples = max(2, int(free_distance/self.resolution*2))
            bearings=yaw+bearing+np.linspace(-math.pi/24,math.pi/24,9)
            directions=np.column_stack((np.cos(bearings),np.sin(bearings)))
            points=start+directions[:,None,:]*np.linspace(0.,free_distance,samples)[None,:,None]
            cells=self.cell(points.reshape(-1,2))
            cells = cells[(cells >= 0).all(axis=1) & (cells < self.size).all(axis=1)]
            if len(cells):
                indices=np.unique(cells[:,1]*self.size+cells[:,0])
                self.logodds.ravel()[indices] -= .55
                self.observed.ravel()[indices] = True
            if hit:
                bearings=yaw+bearing+np.linspace(-math.pi/24,math.pi/24,7)
                cells=self.cell(start+distance*np.column_stack((np.cos(bearings),np.sin(bearings))))
                cells=cells[(cells>=0).all(axis=1)&(cells<self.size).all(axis=1)]
                # Repeated endpoints carry the same beam evidence as before.
                np.add.at(self.logodds,(cells[:,1],cells[:,0]),1.1)
                self.observed[cells[:,1],cells[:,0]]=True
        np.clip(self.logodds, -4., 4., out=self.logodds)

    def grid(self):
        g = np.full(self.logodds.shape, -1, np.int8)
        g[self.observed & (self.logodds < -.25)] = 0
        g[self.observed & (self.logodds > .65)] = 100
        return g

    def save(self, directory):
        import yaml
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        g = self.grid()
        image = np.full(g.shape, 205, np.uint8)
        image[g == 0] = 254
        image[g == 100] = 0
        cv2.imwrite(str(directory/'map.pgm'), np.flipud(image))
        (directory/'map.yaml').write_text(yaml.safe_dump({
            'image': 'map.pgm', 'resolution': self.resolution,
            'origin': [*map(float, self.origin), 0.], 'negate': 0,
            'occupied_thresh': .65, 'free_thresh': .196, 'mode': 'trinary'}))


class FrontierPlanner:
    """Dijkstra on observed free space, with robot inflation and no corner cutting."""
    def __init__(self):
        self.path = []
        self.frontiers = []
        self.goal = None
        self.state = 'WAITING'
        self.last_plan = -math.inf
        self.visits = {}
        self.interest = 'Looking for a new place'
        self.selected_at = None
        self.last_visit=None
        self.last_visit_time=-math.inf
        self.goal_id=0
        self.completed_goals=0
        self.last_completed_goal=None
        self.arrived_since=None
        self.arrived_at=None
        self.goal_distance=None
        self.route_revision=0
        self.navigator=PathFollower()

    def plan(self, occupancy, pose, now=0.):
        """Keep one destination for the entire mission, including blocked routes."""
        grid = occupancy.grid()
        free = grid == 0
        obstacle_clearance=(ndimage.distance_transform_edt(grid!=100)*occupancy.resolution
                            if np.any(grid==100) else np.full(grid.shape,np.inf))
        radius = int(math.ceil(.13 / occupancy.resolution))
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        inflated = ndimage.binary_dilation(grid == 100, structure=xx*xx+yy*yy <= radius*radius)
        traversable = free & ~inflated & (ndimage.distance_transform_edt(free)*occupancy.resolution >= .11)
        start = occupancy.cell(pose[:2])
        self.goal_distance = math.dist(pose[:2], self.goal) if self.goal is not None else None
        visit_cell = tuple(np.floor(np.asarray(pose[:2])/.25).astype(int))
        if visit_cell != self.last_visit or now-self.last_visit_time > 3.:
            self.visits[visit_cell] = self.visits.get(visit_cell, 0)+1
            self.last_visit = visit_cell
            self.last_visit_time = now

        # Completion is explicit and debounced. No deadline can replace a goal.
        if self.arrived_at is not None:
            self.path = []
            self.state = 'ARRIVED'
            self.interest = f'Destination #{self.goal_id} reached; observing before the next destination'
            if now-self.arrived_at < 1.5:
                return
            self.goal = None
            self.arrived_at = self.arrived_since = None
        if self.goal is not None and math.dist(pose[:2], self.goal) <= .12:
            if self.arrived_since is None:
                self.arrived_since = now
            self.path = []
            self.state = 'ARRIVING'
            self.interest = f'Confirming arrival at destination #{self.goal_id}'
            if now-self.arrived_since >= .5:
                self.completed_goals += 1
                self.last_completed_goal = list(self.goal)
                self.arrived_at = now
                self.state = 'ARRIVED'
                self.interest = f'Destination #{self.goal_id} reached'
            return
        self.arrived_since = None

        if not occupancy.inside(start) or not traversable[start[1], start[0]]:
            self.path = []
            self.state = 'GOAL_HELD' if self.goal is not None else 'NEED_SCAN'
            self.interest = (f'Destination #{self.goal_id} retained; waiting for local map clearance'
                             if self.goal is not None else 'Observing the surroundings before choosing a destination')
            return

        # Retain the route too, while the untraveled section remains safe.
        # This prevents equivalent Dijkstra paths from making the robot zigzag.
        if self.goal is not None and self.path:
            points=np.asarray(self.path)
            if len(points)>1:
                vectors=np.diff(points,axis=0)
                fractions=np.clip(np.sum((np.asarray(pose[:2])-points[:-1])*vectors,axis=1)/np.maximum(np.sum(vectors**2,axis=1),1e-10),0.,1.)
                projections=points[:-1]+fractions[:,None]*vectors
                nearest=int(np.argmin(np.linalg.norm(projections-pose[:2],axis=1)))
                remaining=[projections[nearest].tolist()]+self.path[nearest+1:]
            else:remaining=self.path
            safe=all(segment_clear(a,b,traversable,occupancy.origin,occupancy.resolution)
                     for a,b in zip(remaining[:-1],remaining[1:]))
            if safe:
                self.state = 'EXPLORING'
                self.interest = f'Travelling to destination #{self.goal_id}; {self.goal_distance:.2f} m remaining'
                return

        boundary = free & ndimage.binary_dilation(grid == -1)
        candidates = traversable & ndimage.binary_dilation(boundary, iterations=3)
        distances = np.full(grid.shape, np.inf)
        sx, sy = map(int, start)
        distances[sy, sx] = 0.
        parent = {}
        queue = [(0., sx, sy)]
        options = []
        self.frontiers = []
        unknown_density = ndimage.uniform_filter((grid == -1).astype(float), size=21)
        novelty_cache = {}
        while queue:
            d, x, y = heapq.heappop(queue)
            if d > distances[y, x]:
                continue
            if candidates[y, x] and math.dist(occupancy.world([x,y]),pose[:2])>=.25 and obstacle_clearance[y,x]>=.35:
                world = occupancy.world([x, y])
                self.frontiers.append(world.tolist())
                key = tuple(np.floor(world/.25).astype(int))
                if key not in novelty_cache:
                    nearby = sum(self.visits.get((key[0]+dx, key[1]+dy), 0)*math.exp(-(dx*dx+dy*dy)/2.)
                                 for dx in range(-3, 4) for dy in range(-3, 4))
                    novelty_cache[key] = 1./(1.+nearby)
                turn = abs(angle(math.atan2(world[1]-pose[1], world[0]-pose[0])-pose[2]))
                score = 2.5*unknown_density[y, x]+novelty_cache[key]-.18*d*occupancy.resolution-.12*turn
                if self.last_completed_goal is None or math.dist(world, self.last_completed_goal) > .3:
                    options.append((score, (x, y)))
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < occupancy.size and 0 <= ny < occupancy.size and traversable[ny, nx]):
                    continue
                if dx and dy and not (traversable[y, nx] and traversable[ny, x]):
                    continue
                # Prefer room to steer; a shortest path hugging inflated walls
                # repeatedly trips the independent forward-sonar caution band.
                nd = d+math.hypot(dx, dy)*(1.+.04/max(.05,obstacle_clearance[ny,nx])**2)
                if nd < distances[ny, nx]:
                    distances[ny, nx] = nd
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(queue, (nd, nx, ny))

        if self.goal is not None:
            gx, gy = occupancy.cell(self.goal)
            if not occupancy.inside([gx, gy]) or not np.isfinite(distances[gy, gx]):
                # Route to a safe arrival position within the same destination's
                # tolerance. The destination coordinates and ID never move.
                possible=[]
                for cy,cx in np.argwhere(np.isfinite(distances)):
                    if math.dist(occupancy.world([cx,cy]),self.goal)<=.115:
                        possible.append((distances[cy,cx],(int(cx),int(cy))))
                if not possible:
                    self.path = []
                    self.state = 'GOAL_BLOCKED'
                    self.interest = f'Destination #{self.goal_id} retained; route blocked, observing for a safe route'
                    return
                best=min(possible,key=lambda item:item[0])[1]
            else:best = (int(gx), int(gy))
        elif options:
            best = max(options, key=lambda item: item[0])[1]
            self.goal = occupancy.world(best).tolist()
            self.goal_id += 1
            self.selected_at = now
            self.goal_distance = math.dist(pose[:2], self.goal)
        else:
            self.path = []
            self.state = 'NO_REACHABLE_FRONTIER'
            self.interest = 'No reachable unfamiliar destination; observing from here'
            return

        cells = [best]
        while cells[-1] != (sx, sy):
            cells.append(parent[cells[-1]])
        self.path = [occupancy.world(cell).tolist() for cell in reversed(cells)]
        self.path = simplify_route(self.path,traversable,occupancy.origin,occupancy.resolution)
        self.route_revision += 1
        self.state = 'EXPLORING'
        self.interest = f'Travelling to destination #{self.goal_id}; {self.goal_distance:.2f} m remaining'

    def command(self, pose, readings, safety, ready):
        if safety.state == 'STOP':
            return safety
        if not ready:
            return Decision(0., 0., 'RECOVERING', 'Holding position while visual tracking recovers')
        if safety.state == 'TURN':
            return safety
        if self.state in ['ARRIVING','ARRIVED']:
            return Decision(0.,0.,self.state,'Destination reached; observing before choosing the next destination')
        if self.state in ['GOAL_HELD','GOAL_BLOCKED']:
            return Decision(0.,0.,'WAITING','Destination retained; waiting for a safe route to the same destination')
        if self.state in ['NEED_SCAN','NO_REACHABLE_FRONTIER']:
            if min(readings['left'].clearance(),readings['right'].clearance())<.17 or readings['rear'].clearance()<.14 or readings['front'].clearance()<.18:
                return Decision(0., 0., 'STOP', 'Insufficient clearance for mapping scan')
            return Decision(0., .25, 'LOOK_AROUND', 'Curious scan: looking for new reachable space')
        if not self.path:
            return Decision(0., 0., 'WAITING', 'No reachable frontier; scan or move manually to reveal space')
        if math.dist(pose[:2], self.goal) <= .12:
            return Decision(0., 0., 'ARRIVING', 'Confirming arrival at the committed destination')
        decision=self.navigator.command(pose,self.path,self.goal,safety,getattr(self,'navigation',None))
        if decision.state=='ALIGN':
            side='left' if decision.angular>0 else 'right'
            if min(readings[side].clearance(),readings['rear'].clearance())<.18:
                return Decision(0.,0.,'STOP','Destination retained; insufficient clearance to align')
        return decision


class CuriosityPolicy:
    """Complete each destination; interrupt travel only for safety/relocalization."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.recovery_origin=None
        self.recovery_started=None
        self.recovery_wait=-math.inf
        self.turn_started=None
        self.relocalize_target=None
        self.relocalize_started=-math.inf
        self.relocalize_attempted=set()
        self.navigator=PathFollower()
        self.route_scan_started=None

    def decide(self,snapshot,readings,safety,ready,now):
        planner=FrontierPlanner()
        planner.navigator=self.navigator
        if snapshot:
            planner.path=snapshot['path'];planner.goal=snapshot['goal'];planner.state=snapshot['planner_state']
            planner.navigation=snapshot
        if (not ready and snapshot and snapshot.get('can_bridge_pose') and safety.state=='ADVANCE' and planner.path
                and readings['front'].clearance()>.45
                and min(readings['left'].clearance(),readings['right'].clearance())>.20):
            command=planner.command(snapshot['pose'],readings,safety,True)
            if command.state=='EXPLORE' and abs(command.angular)<.20:
                return Decision(min(.020,command.linear),float(np.clip(command.angular,-.10,.10)),
                                'TRACKING_BRIDGE','Brief visual dropout; moving slowly within the fresh safe corridor')
        if ready and planner.state in ['GOAL_HELD','GOAL_BLOCKED'] and safety.state not in ['STOP','TURN']:
            if min(readings[s].clearance() for s in ['front','left','rear','right'])>.20:
                if self.route_scan_started is None:self.route_scan_started=now
                bearing=math.atan2(planner.goal[1]-snapshot['pose'][1],planner.goal[0]-snapshot['pose'][0])
                target=bearing+.40*math.sin((now-self.route_scan_started)*.65)
                yaw=angle(target-snapshot['pose'][2])
                return Decision(0.,float(np.clip(yaw,-.20,.20)),'RESCAN','Destination retained; observing its corridor to rebuild a safe route')
        else:self.route_scan_started=None
        if not ready and snapshot and snapshot.get('can_relocalize') and safety.state!='STOP':
            known=[p for p in snapshot.get('trajectory',[]) if math.dist(p[:2],snapshot['pose'][:2])<.35]
            if known and min(readings[s].clearance() for s in ['front','left','rear','right'])>.18:
                yaw=snapshot['pose'][2]
                if self.relocalize_target is None or now-self.relocalize_started>3.:
                    candidates=[p for p in known if round(p[2]/.2) not in self.relocalize_attempted]
                    if not candidates:
                        self.relocalize_attempted.clear();candidates=known
                    self.relocalize_target=min(candidates,key=lambda p:abs(angle(p[2]-yaw)))[2]
                    self.relocalize_attempted.add(round(self.relocalize_target/.2));self.relocalize_started=now
                error=angle(self.relocalize_target-yaw)
                if abs(error)>.06:
                    return Decision(0.,float(np.clip(error,-.2,.2)),'RELOCALIZE','Returning to a remembered view to recover visual tracking')
                return Decision(0.,0.,'RELOCALIZE','Checking the remembered view; translation held until tracking recovers')
        if ready:self.relocalize_target=None;self.relocalize_attempted.clear()
        front_stop=safety.state=='STOP' and safety.reason.startswith('FRONT emergency')
        if safety.state=='TURN':
            if self.turn_started is None:self.turn_started=now
        else:self.turn_started=None
        turning_too_long=self.turn_started is not None and now-self.turn_started>6.
        recovery_needed=front_stop or turning_too_long
        if ready and recovery_needed and now>=self.recovery_wait:
            # All upstream freshness checks precede this specific clearance stop.
            # Reverse distance is measured by vision, not integrated from commands.
            if all(s in readings and readings[s].usable() for s in ['front','left','rear','right']) and readings['rear'].clearance()>.35 and min(readings['left'].clearance(),readings['right'].clearance())>.12:
                if self.recovery_origin is None:
                    self.recovery_origin=np.array(snapshot['pose'][:2]);self.recovery_started=now
                elapsed=now-self.recovery_started
                distance=np.linalg.norm(np.asarray(snapshot['pose'][:2])-self.recovery_origin)
                if elapsed<.4:
                    return Decision(0.,0.,'STARTLED','Pausing before backing away from a close obstacle')
                if elapsed<4. and distance<.10:
                    return Decision(-.035,0.,'BACK_AWAY','Rear and sides clear; backing away to regain turning space')
                self.recovery_origin=None;self.recovery_wait=now+3.
            else:
                self.recovery_origin=None
        elif not recovery_needed:
            self.recovery_origin=None
        if not ready or safety.state in ['STOP','TURN']:
            return planner.command(snapshot['pose'] if snapshot else [0.,0.,0.],readings,safety,ready)
        # Travel is never interrupted for a curiosity timer. Observation occurs
        # only after the planner has confirmed arrival at this destination.
        return planner.command(snapshot['pose'],readings,safety,ready)


def solve_map_correction(poses,edges,scans,size,resolution):
    count=len(poses)
    matrix=np.zeros((len(edges)+1,count));rhs=np.zeros((len(edges)+1,2))
    matrix[0,0]=100.;rhs[0]=poses[0]*100.
    for row,(i,j,delta,weight) in enumerate(edges,1):
        matrix[row,i]=-weight;matrix[row,j]=weight;rhs[row]=delta*weight
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):solved=np.linalg.lstsq(matrix,rhs,rcond=None)[0]
    corrections=solved-poses
    occupancy=OccupancyMap(size,resolution)
    for record in scans:
        pose=record['pose'].copy();pose[:2]+=corrections[record['keyframe']]
        occupancy.update(pose,record['ranges'])
    return corrections,occupancy


def map_correction_process(inputs,outputs):
    outputs.cancel_join_thread()
    while True:
        job=inputs.get()
        if job is None:return
        try:outputs.put(('ok',solve_map_correction(*job)))
        except Exception:
            import traceback
            outputs.put(('error',traceback.format_exc()))


class AsyncMapOptimizer:
    """Replay loop-corrected scans without blocking live image tracking."""
    def __init__(self):
        context=mp.get_context('spawn')
        self.inputs=context.Queue(1);self.outputs=context.Queue(1)
        self.process=context.Process(target=map_correction_process,args=(self.inputs,self.outputs),daemon=True)
        self.pending=False;self.scan_count=0;self.error=None

    def start(self):self.process.start()

    def submit(self,engine):
        if self.pending:return
        scans=[{'pose':r['pose'].copy(),'ranges':dict(r['ranges']),'keyframe':r['keyframe']} for r in engine.scans]
        self.inputs.put_nowait((np.array([k['pose'][:2].copy() for k in engine.keyframes]),
                               list(engine.edges),scans,engine.map.size,engine.map.resolution))
        self.scan_count=len(scans);self.pending=True

    def collect(self,engine):
        if not self.pending:return
        try:kind,result=self.outputs.get_nowait()
        except queue.Empty:return
        self.pending=False
        if kind=='error':self.error=result;return
        corrections,occupancy=result
        engine.apply_correction(corrections,occupancy,self.scan_count)

    def stop(self):
        if self.process.is_alive():self.process.terminate()
        if self.process.pid is not None:self.process.join(timeout=1.)
        for channel in [self.inputs,self.outputs]:channel.cancel_join_thread();channel.close()


class FloorSlam:
    def __init__(self):
        self.map = OccupancyMap()
        self.planner = FrontierPlanner()
        self.pose = np.zeros(3)
        self.previous = None
        self.tracking = False
        self.status = 'Waiting for camera calibration and IMU/compass'
        self.inliers = 0
        self.keyframes = []
        self.flow_keyframes=[]
        self.edges = []
        self.scans = []
        self.loop_closures = 0
        self.last_stamp = None
        self.last_plan = -math.inf
        self.last_scan_stamp = {}
        self.last_good_stamp=None
        self.good_streak=0
        self.last_place_check=-math.inf
        self.orb = cv2.ORB_create(nfeatures=1600, edgeThreshold=10, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.optimizer=None

    def features(self, gray, calibration, roll, pitch):
        mask = np.zeros_like(gray)
        mask[int(gray.shape[0]*.52):, :] = 255
        points = cv2.goodFeaturesToTrack(gray, 600, .008, 7, mask=mask)
        if points is None:
            return np.empty((0, 2), np.float32)
        points = points.reshape(-1, 2)
        _, valid = calibration.ground(points, roll, pitch)
        return points[valid]

    def place(self, gray, calibration, roll, pitch, yaw):
        mask = np.zeros_like(gray); mask[int(gray.shape[0]*.52):, :] = 255
        keys, descriptors = self.orb.detectAndCompute(gray, mask)
        if descriptors is None:
            return None
        pixels = np.array([k.pt for k in keys])
        ground, valid = calibration.ground(pixels, roll, pitch)
        if valid.sum() < 20:
            return None
        pixels=self.features(gray,calibration,roll,pitch)
        vectors,_=calibration.ground(pixels,roll,pitch)
        return {'points': ground[valid] @ rotation(yaw).T, 'descriptors': descriptors[valid],
                'gray':gray.copy(),'pixels':pixels,'ground':vectors@rotation(yaw).T,
                'pose': self.pose.copy(), 'scan': len(self.scans)}

    def match_trusted_view(self,gray,calibration,roll,pitch,yaw,reference):
        if 'gray' not in reference or len(reference['pixels'])<8:return None
        pixels=reference['pixels'].reshape(-1,1,2)
        new,valid,_=cv2.calcOpticalFlowPyrLK(reference['gray'],gray,pixels,None,winSize=(21,21),maxLevel=3)
        back,back_valid,_=cv2.calcOpticalFlowPyrLK(gray,reference['gray'],new,None)
        vectors,floor=calibration.ground(new.reshape(-1,2),roll,pitch)
        keep=valid.ravel().astype(bool)&back_valid.ravel().astype(bool)&floor
        keep &= np.linalg.norm(back.reshape(-1,2)-pixels.reshape(-1,2),axis=1)<.6
        delta,count=floor_translation(reference['ground'][keep],vectors[keep]@rotation(yaw).T,threshold=.008)
        if delta is None:return None
        measured=reference['pose'][:2]+delta
        return (delta,count) if math.dist(measured,self.pose[:2])<.12 else None

    def match_place(self, current, reference):
        matches = self.matcher.knnMatch(current['descriptors'], reference['descriptors'], k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .65*m[1].distance and m[0].distance < 55]
        if len(good) < 20 or len({m.trainIdx for m in good}) < 20:
            return None
        old = np.array([reference['points'][m.trainIdx] for m in good])
        new = np.array([current['points'][m.queryIdx] for m in good])
        delta, count = consensus_translation(old, new, minimum=20, threshold=.015)
        if delta is None:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(new, old, method=cv2.RANSAC, ransacReprojThreshold=.015)
        if transform is None or int(inliers.sum()) < 20:
            return None
        scale = np.linalg.norm(transform[:, 0])
        if abs(scale-1.) > .04 or abs(math.atan2(transform[1, 0], transform[0, 0])) > math.radians(3):
            return None
        return delta, count

    def optimize(self):
        # Translation pose graph; compass fixes each keyframe's rotation.
        count = len(self.keyframes)
        if count < 2:
            return
        if self.optimizer is not None:
            self.optimizer.submit(self);return
        corrections,occupancy=solve_map_correction(np.array([k['pose'][:2].copy() for k in self.keyframes]),
                                                  self.edges,self.scans,self.map.size,self.map.resolution)
        self.apply_correction(corrections,occupancy,len(self.scans))

    def apply_correction(self,corrections,occupancy,scan_count):
        self.pose[:2] += corrections[-1]
        for i, k in enumerate(self.keyframes):
            k['pose'][:2] += corrections[min(i,len(corrections)-1)]
        for ref in self.flow_keyframes:
            ref['pose'][:2]+=corrections[min(ref.get('keyframe',len(corrections)-1),len(corrections)-1)]
        self.map=occupancy
        for i,record in enumerate(self.scans):
            record['pose'][:2] += corrections[min(record['keyframe'],len(corrections)-1)]
            if i>=scan_count:self.map.update(record['pose'],record['ranges'])
        self.last_plan = -math.inf

    def process(self, bgr, stamp, calibration, attitude, ranges, plan=True):
        if self.last_stamp is not None and stamp < self.last_stamp:
            optimizer=self.optimizer
            # Discard an old-epoch correction before resetting its coordinates.
            if optimizer is not None:
                optimizer.stop();optimizer=AsyncMapOptimizer();optimizer.start()
            self.__init__();self.optimizer=optimizer
        if self.optimizer is not None:self.optimizer.collect(self)
        roll, pitch, yaw, heading_ready = attitude
        if not heading_ready:
            self.tracking = False;self.good_streak=0; self.status = 'IMU/compass stale; SLAM stopped'
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.pose[2] = yaw
        good = False
        self.inliers = 0
        if self.previous is not None:
            prev_gray, prev_pixels, prev_ground, prev_stamp, prev_yaw = self.previous
            dt = stamp-prev_stamp
            # A stopped robot may regain the same view after a long dropout.
            # Keep checking its newest trusted image; an elapsed-time cutoff
            # used to discard that best recovery anchor after two seconds.
            compatible=dt<2. or abs(angle(yaw-prev_yaw))<.45
            if len(prev_pixels) >= 4 and dt>0 and compatible:
                new, valid, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pixels.reshape(-1, 1, 2), None,
                                                       winSize=(21, 21), maxLevel=3)
                back, back_valid, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, new, None)
                pixels = new.reshape(-1, 2)
                ground, floor = calibration.ground(pixels, roll, pitch)
                keep = valid.ravel().astype(bool) & back_valid.ravel().astype(bool) & floor
                keep &= np.linalg.norm(back.reshape(-1, 2)-prev_pixels, axis=1) < .8
                delta, self.inliers = floor_translation(prev_ground[keep] @ rotation(prev_yaw).T,
                                                        ground[keep] @ rotation(yaw).T)
                if delta is not None and np.linalg.norm(delta) < .25*min(dt,.5)+.006 and abs(angle(yaw-prev_yaw)) < 1.5*dt+.02:
                    self.pose[:2] += delta
                    good = True
        pixels = self.features(gray, calibration, roll, pitch)
        ground, _ = calibration.ground(pixels, roll, pitch)
        if not good:
            candidates=sorted((ref for ref in self.flow_keyframes
                               if abs(angle(yaw-ref['pose'][2]))<=.3 and math.dist(self.pose[:2],ref['pose'][:2])<=.25),
                              key=lambda ref:abs(angle(yaw-ref['pose'][2]))+math.dist(self.pose[:2],ref['pose'][:2]))[:3]
            for ref in candidates:
                if abs(angle(yaw-ref['pose'][2]))>.3 or math.dist(self.pose[:2],ref['pose'][:2])>.25:continue
                match=self.match_trusted_view(gray,calibration,roll,pitch,yaw,ref)
                if match is not None:
                    delta,self.inliers=match;self.pose[:2]=ref['pose'][:2]+delta;good=True;break
        current = None
        if not self.keyframes and len(pixels) >= 20:
            self.tracking = True; good = True
            current = self.place(gray, calibration, roll, pitch, yaw)
            if current is not None:
                self.keyframes.append(current)
        # Loop/relocalization candidates must be spatially near and view-compatible.
        interval=.4 if not good else 2.
        due=stamp-self.last_place_check>=interval
        if due:
            self.last_place_check=stamp
            current = current or self.place(gray, calibration, roll, pitch, yaw)
            if current is not None:
                candidates=[i for i,ref in enumerate(self.keyframes)
                            if abs(angle(yaw-ref['pose'][2]))<=.45 and math.dist(self.pose[:2],ref['pose'][:2])<=.5
                            and (not good or len(self.keyframes)-1-i>=8)]
                candidates=sorted(candidates,key=lambda i:abs(angle(yaw-self.keyframes[i]['pose'][2]))+
                                  math.dist(self.pose[:2],self.keyframes[i]['pose'][:2]))[:4]
                for i in candidates:
                    ref = self.keyframes[i]
                    if abs(angle(yaw-ref['pose'][2])) > .45 or math.dist(self.pose[:2], ref['pose'][:2]) > .5:
                        continue
                    match = self.match_place(current, ref)
                    if match is None and not good and abs(angle(yaw-ref['pose'][2]))<.25:
                        match=self.match_trusted_view(gray,calibration,roll,pitch,yaw,ref)
                    if match is None:
                        continue
                    delta, self.inliers = match
                    predicted=self.pose[:2].copy()
                    measured = ref['pose'][:2]+delta
                    correction = measured-self.pose[:2]
                    if np.linalg.norm(correction) > .25:
                        continue
                    self.pose[:2] = measured; good = True; self.tracking = True
                    if len(self.keyframes)-1-i >= 8 and np.linalg.norm(correction) > .015:
                        # Relative observation from an older node to the latest node.
                        latest = self.keyframes[-1]
                        edge = predicted-latest['pose'][:2]
                        self.edges.append((i, len(self.keyframes)-1, delta-edge, 3.))
                        self.loop_closures += 1
                        self.pose[:2]=predicted
                        self.optimize()
                    break
        self.tracking = good
        self.good_streak=self.good_streak+1 if good else 0
        if good:self.last_good_stamp=stamp
        self.last_stamp = stamp
        # Keep the last geometrically accepted anchor through a dropped/bad frame.
        # Replacing it on failure makes the very next good frame unrecoverable.
        if good:
            self.previous = (gray, pixels, ground, stamp, yaw)
            if len(pixels)>=8 and (not self.flow_keyframes or math.dist(self.pose[:2],self.flow_keyframes[-1]['pose'][:2])>.06 or abs(angle(yaw-self.flow_keyframes[-1]['pose'][2]))>.1):
                self.flow_keyframes.append({'gray':gray.copy(),'pixels':pixels,'ground':ground@rotation(yaw).T,'pose':self.pose.copy(),'keyframe':len(self.keyframes)-1})
                self.flow_keyframes=self.flow_keyframes[-32:]
        self.status = 'TRACKING · flat-floor visual/compass SLAM' if good else 'TRACKING LOST · waiting for a recognized keyframe'
        if not good or not self.keyframes:
            return
        last = self.keyframes[-1]
        if len(self.keyframes) < 300 and (math.dist(self.pose[:2], last['pose'][:2]) > .12 or abs(angle(yaw-last['pose'][2])) > .12):
            current = current or self.place(gray, calibration, roll, pitch, yaw)
            if current is not None:
                current['pose'] = self.pose.copy(); current['stamp'] = stamp
                self.edges.append((len(self.keyframes)-1, len(self.keyframes), self.pose[:2]-last['pose'][:2], 1.))
                self.keyframes.append(current)
                # Keep only recent optical-flow images; compact ORB landmarks remain.
                if len(self.keyframes)>32:
                    for key in ['gray','pixels','ground']:self.keyframes[-33].pop(key,None)
        self.keyframes[-1].setdefault('stamp', stamp)
        fresh = {}
        for side, (value, maximum, measurement_stamp) in ranges.items():
            if -.05 <= stamp-measurement_stamp <= .25 and measurement_stamp > self.last_scan_stamp.get(side, -math.inf):
                fresh[side] = (value, maximum); self.last_scan_stamp[side] = measurement_stamp
        if fresh:
            self.map.update(self.pose, fresh)
            self.scans.append({'pose': self.pose.copy(), 'ranges': fresh, 'keyframe': len(self.keyframes)-1})
        if plan and stamp-self.last_plan > 1.:
            self.planner.plan(self.map, self.pose,stamp); self.last_plan = stamp

    def snapshot(self):
        grid = self.map.grid()
        return {'pose': self.pose.tolist(), 'tracking': self.tracking, 'status': self.status,
                'last_good_stamp':self.last_good_stamp,'good_streak':self.good_streak,
                'inliers': self.inliers, 'keyframes': len(self.keyframes), 'loop_closures': self.loop_closures,
                'observed_m2': float((grid >= 0).sum()*self.map.resolution**2),
                'free_m2': float((grid == 0).sum()*self.map.resolution**2),
                'frontiers': self.planner.frontiers, 'path': self.planner.path,
                'goal': self.planner.goal, 'planner_state': self.planner.state,
                'goal_id':self.planner.goal_id,'completed_goals':self.planner.completed_goals,
                'goal_distance_m':self.planner.goal_distance,'route_revision':self.planner.route_revision,
                'interest':self.planner.interest,'visited_places':len(self.planner.visits),
                'grid': grid, 'origin': self.map.origin.tolist(), 'resolution': self.map.resolution,
                'trajectory': [k['pose'].tolist() for k in self.keyframes], 'stamp': self.last_stamp}


class PlanningWorker(threading.Thread):
    """Keep graph search and visit memory off the live tracking/GUI path."""
    def __init__(self):
        super().__init__(daemon=True,name='robot-frontiers')
        self.frames=LatestSlot();self.stop_event=threading.Event();self.lock=threading.Lock()
        self.planner=FrontierPlanner();self.epoch=None;self.latest={};self.error=None

    def run(self):
        seq=0
        while not self.stop_event.is_set():
            seq,item=self.frames.wait_new(seq,.2)
            if item is None:continue
            occupancy,pose,stamp,epoch=item
            if epoch!=self.epoch:self.planner=FrontierPlanner();self.epoch=epoch
            try:
                self.planner.plan(occupancy,pose,stamp)
                result={'path':self.planner.path,'frontiers':self.planner.frontiers,'goal':self.planner.goal,
                        'planner_state':self.planner.state,'interest':self.planner.interest,
                        'visited_places':len(self.planner.visits),'visits':dict(self.planner.visits),
                        'goal_id':self.planner.goal_id,'completed_goals':self.planner.completed_goals,
                        'goal_distance_m':self.planner.goal_distance,'route_revision':self.planner.route_revision,
                        'plan_stamp':stamp,'plan_finished':time.monotonic(),'epoch':epoch}
                with self.lock:self.latest=result;self.error=None
            except Exception:
                import traceback
                self.error=traceback.format_exc()
                with self.lock:
                    self.latest={**self.latest,'path':[],'planner_state':'GOAL_HELD',
                                 'interest':'Destination retained; planner error, movement held'}

    def snapshot(self):
        with self.lock:return dict(self.latest)

    def stop(self):
        self.stop_event.set();self.frames.close()


class SlamWorker(threading.Thread):
    def __init__(self, frames):
        super().__init__(daemon=True, name='robot-slam')
        self.frames = frames
        self.engine = FloorSlam()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.error = None
        self.latest = self.engine.snapshot()
        self.finished = -math.inf
        self.planning=PlanningWorker();self.epoch=0;self.processed=0;self.last_plan_request=-math.inf

    def run(self):
        seq = 0
        self.planning.start()
        self.engine.optimizer=AsyncMapOptimizer();self.engine.optimizer.start()
        try:
            while not self.stop_event.is_set():
                seq, item = self.frames.wait_new(seq, .2)
                if item is None:
                    continue
                if self.engine.last_stamp is not None and item[1]<self.engine.last_stamp:
                    self.epoch+=1;self.last_plan_request=-math.inf
                started=time.monotonic()
                try:self.engine.process(*item,plan=False)
                except (cv2.error,ValueError):
                    self.engine.tracking=False;self.engine.good_streak=0;self.engine.status='Bad camera frame; waiting for the next frame'
                self.processed+=1
                with self.lock:
                    self.latest = self.engine.snapshot();self.latest['epoch']=self.epoch
                    self.latest['processing_ms']=(time.monotonic()-started)*1000
                    self.latest['revision']=self.processed;self.finished = time.monotonic()
                if self.engine.tracking and item[1]-self.last_plan_request>.5:
                    self.last_plan_request=item[1]
                    occupancy=OccupancyMap(self.engine.map.size,self.engine.map.resolution)
                    occupancy.logodds=self.engine.map.logodds.copy();occupancy.observed=self.engine.map.observed.copy()
                    self.planning.frames.put((occupancy,self.engine.pose.copy(),item[1],self.epoch))
        except Exception:
            import traceback
            self.error = traceback.format_exc()
        finally:
            self.engine.optimizer.stop()
            self.planning.stop();self.planning.join(timeout=2.)

    def snapshot(self):
        with self.lock:
            result = dict(self.latest)
            result['age_s'] = time.monotonic()-self.finished
            trusted=result.get('last_good_stamp')
            result['trusted_pose_age_s']=(result['stamp']-trusted+result['age_s']) if trusted is not None and result['stamp'] is not None else math.inf
            result['ready'] = result['tracking'] and result.get('good_streak',0)>=2 and result['age_s'] < .45 and self.error is None
            if self.error:
                result['status'] = 'SLAM worker failed; see terminal/report'
        plan=self.planning.snapshot()
        if plan and plan['epoch']==result.get('epoch',0):
            result.update(plan)
            result['plan_age_s']=time.monotonic()-plan['plan_finished']
            result['plan_ready']=result['plan_age_s']<2. and self.planning.error is None
        else:
            result['plan_age_s']=math.inf;result['plan_ready']=False
        if self.planning.error:result['interest']='Planner error; movement stopped'
        return result

    def save(self, directory):
        # Export an immutable snapshot so the running mapping thread can continue.
        import yaml
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        s = self.snapshot(); grid = s.pop('grid')
        image = np.full(grid.shape, 205, np.uint8); image[grid == 0] = 254; image[grid == 100] = 0
        cv2.imwrite(str(directory/'map.pgm'), np.flipud(image))
        (directory/'map.yaml').write_text(yaml.safe_dump({'image': 'map.pgm', 'resolution': s['resolution'],
            'origin': [*s['origin'], 0.], 'negate': 0, 'occupied_thresh': .65, 'free_thresh': .196, 'mode': 'trinary'}))
        visits=s.pop('visits',{})
        s['visits']=[{'cell':[int(x) for x in k],'count':int(v)} for k,v in visits.items()]
        (directory/'slam_report.json').write_text(json.dumps({**s, 'error': self.error,'planner_error':self.planning.error}, indent=2))

    def stop(self):
        self.stop_event.set(); self.frames.close()


def render_map(snapshot):
    grid = snapshot['grid']
    image = np.empty((*grid.shape, 3), np.uint8); image[:] = (54, 46, 38)
    image[grid == 0] = (210, 220, 225); image[grid == 100] = (30, 30, 30)
    origin = np.asarray(snapshot['origin']); resolution = snapshot['resolution']
    def pixel(p):
        return tuple(np.floor((np.asarray(p[:2])-origin)/resolution).astype(int))
    for p in snapshot['frontiers']:
        cv2.circle(image, pixel(p), 1, (220, 180, 20), -1)
    for points, color in [(snapshot['trajectory'], (160, 100, 40)), (snapshot['path'], (80, 190, 20))]:
        if len(points) > 1:
            cv2.polylines(image, [np.array([pixel(p) for p in points], np.int32)], False, color, 1)
    pose = snapshot['pose']; center = pixel(pose)
    cv2.circle(image, center, 2, (50, 70, 250), -1)
    tip = np.array(pose[:2])+.2*np.array([math.cos(pose[2]), math.sin(pose[2])])
    cv2.arrowedLine(image, center, pixel(tip), (50, 70, 250), 1, tipLength=.4)
    # Fit the observed area, keeping a margin for nearby unknown space.
    ys,xs=np.where(grid>=0)
    if len(xs):
        lo=np.minimum([xs.min(),ys.min()],center)-8
        hi=np.maximum([xs.max(),ys.max()],center)+9
    else:
        lo=np.asarray(center)-30;hi=np.asarray(center)+31
    midpoint=(lo+hi)//2;half=max(30,int(np.max(hi-lo))//2+1)
    lo=np.maximum(midpoint-half,0);hi=np.minimum(midpoint+half,grid.shape[0])
    image=image[lo[1]:hi[1],lo[0]:hi[0]]
    return cv2.resize(np.flipud(image), (720,720), interpolation=cv2.INTER_NEAREST)
