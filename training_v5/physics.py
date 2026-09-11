"""Original physical hide-and-seek environment; MuJoCo 3.13 native/browser pair.
Coordinates are metres, Z up. No path planner or prescribed agent subgoals.
"""
from __future__ import annotations
import math, json
import numpy as np
import mujoco

DT=.08
SUBSTEPS=16
PREP=96
PLAY=144
VISION_RANGE=float("inf")
FOV=3*math.pi/4
OBJECT_SLOTS=10
LIDAR=30
OBS_DIM=10+8+OBJECT_SLOTS*16+LIDAR

class Random:
    def __init__(self,seed):self.state=int(seed)&0xffffffff or 1
    def next(self):
        x=self.state;x^=(x<<13)&0xffffffff;x^=x>>17;x^=(x<<5)&0xffffffff
        self.state=x&0xffffffff;return self.state/4294967296
    def uniform(self,a,b):return a+(b-a)*self.next()

def generate_arena(seed=1,scenario='shelter',size=8,n_boxes=3,n_ramps=1):
    if scenario not in ('shelter','rooms','open','connected-rooms','corridors','multi-exit'):raise ValueError('Unknown arena scenario')
    if not 6<=size<=12 or not 0<=n_boxes<=8 or not 0<=n_ramps<=2:raise ValueError('Arena size 6–12m, boxes 0–8, ramps 0–2')
    r=Random(seed);s=float(size);walls=[];objects=[]
    def wall(x,y,sx,sy):walls.append(dict(position=[x,y,1.1],size=[sx,sy,2.2],yaw=0))
    # The solid outside boundary is part of this original small-arena task.
    for x,y,sx,sy in [(s/2,-.1,s+.4,.2),(s/2,s+.1,s+.4,.2),(-.1,s/2,.2,s),(s+.1,s/2,.2,s)]:wall(x,y,sx,sy)
    if scenario=='shelter':
        # One room, a traversable doorway, and movable material nearby.
        left=.12*s;right=.50*s;bottom=.22*s;top=.78*s;gap=.52
        wall(left,(bottom+top)/2,.18,top-bottom)
        wall((left+right)/2,bottom,right-left,.18);wall((left+right)/2,top,right-left,.18)
        wall(right,(bottom+s/2-gap)/2,.18,s/2-gap-bottom)
        wall(right,(s/2+gap+top)/2,.18,top-s/2-gap)
        agents=[dict(position=[s*.36,s*.50,.25],yaw=0),dict(position=[s*.78,s*.50,.25],yaw=math.pi)]
    elif scenario=='rooms':
        split=r.uniform(.40,.60)*s;gap=r.uniform(.35,.65)*s
        wall(split,gap/2-.35,.18,gap-.7);wall(split,(gap+.7+s)/2,.18,s-gap-.7)
        if r.next()<.75:wall(s*.23,s*.55,s*.32,.18)
        agents=[dict(position=[s*.22,s*.22,.25],yaw=r.uniform(-math.pi,math.pi)),dict(position=[s*.78,s*.78,.25],yaw=r.uniform(-math.pi,math.pi))]
    elif scenario in ('connected-rooms','corridors','multi-exit'):
        # Each partition has two independently jittered clear openings. No
        # closed cells: the wall-only floorplan remains connected for a .5m agent.
        def partition(axis,fixed,lo,hi):
            span=hi-lo;centers=[lo+span*r.uniform(.23,.31),lo+span*r.uniform(.69,.77)]
            width=r.uniform(1.10,1.35);start=lo
            for center in centers:
                end=center-width/2
                if end>start:
                    if axis==0:wall(fixed,(start+end)/2,.18,end-start)
                    else:wall((start+end)/2,fixed,end-start,.18)
                start=center+width/2
            if start<hi:
                if axis==0:wall(fixed,(start+hi)/2,.18,hi-start)
                else:wall((start+hi)/2,fixed,hi-start,.18)
        if scenario=='connected-rooms':
            split=r.uniform(.43,.57)*s
            partition(0,split,0,s);partition(1,r.uniform(.43,.57)*s,0,s)
        elif scenario=='corridors':
            middle=r.uniform(.45,.55)*s;half=r.uniform(.75,1.0)
            partition(1,middle-half,0,s);partition(1,middle+half,0,s)
        else:
            left=.29*s;right=.71*s;bottom=.25*s;top=.75*s
            # Central shelter has an exit on each side, with room to circle it.
            for x in [left,right]:
                gap=(bottom+top)/2+r.uniform(-.2,.2);width=1.2
                wall(x,(bottom+gap-width/2)/2,.18,gap-width/2-bottom)
                wall(x,(gap+width/2+top)/2,.18,top-gap-width/2)
            for y in [bottom,top]:
                gap=(left+right)/2+r.uniform(-.2,.2);width=1.2
                wall((left+gap-width/2)/2,y,gap-width/2-left,.18)
                wall((gap+width/2+right)/2,y,right-gap-width/2,.18)
        agents=[dict(position=[s*.15,s*.15,.25],yaw=r.uniform(-math.pi,math.pi)),dict(position=[s*.85,s*.85,.25],yaw=r.uniform(-math.pi,math.pi))]
    else:agents=[dict(position=[s*.25,s*.35,.25],yaw=0),dict(position=[s*.75,s*.65,.25],yaw=math.pi)]
    def clear(x,y,sx,sy):
        for o in walls+objects:
            px,py,_=o['position'];ox,oy,_=o['size'];a=o.get('yaw',0);ex=(abs(math.cos(a))*ox+abs(math.sin(a))*oy)/2;ey=(abs(math.sin(a))*ox+abs(math.cos(a))*oy)/2
            if abs(x-px)<sx/2+ex+.12 and abs(y-py)<sy/2+ey+.12:return False
        return all(math.hypot(x-a['position'][0],y-a['position'][1])>.45+max(sx,sy)/2 for a in agents)
    for i in range(n_boxes+n_ramps):
        kind='ramp' if i>=n_boxes else 'plank' if i%2==0 else 'box'
        dims=[.9,.8,.7] if kind=='ramp' else [1.7,.25,.7] if kind=='plank' else [.7,.7,.7]
        placed=False
        for attempt in range(500):
            if scenario=='shelter' and i==0:
                x=r.uniform(.22,.40)*s;y=r.uniform(.29,.37)*s
            else:x=r.uniform(.9,s-.9);y=r.uniform(.9,s-.9)
            yaw=0 if kind=='plank' else r.uniform(-math.pi,math.pi)
            sx=abs(math.cos(yaw))*dims[0]+abs(math.sin(yaw))*dims[1];sy=abs(math.sin(yaw))*dims[0]+abs(math.cos(yaw))*dims[1]
            if clear(x,y,sx,sy):placed=True;break
        if placed:objects.append(dict(id=f'prop-{i}',kind=kind,position=[x,y,dims[2]/2+.003],size=dims,yaw=yaw,mass=1.8 if kind=='plank' else 1.2))
    # Randomize starts independently, then rotate/reflect the whole physical task.
    # This changes observations and approach directions without an agent-side script.
    for agent in agents:
        original=agent['position'].copy()
        for attempt in range(40):
            x=original[0]+r.uniform(-.45,.45);y=original[1]+r.uniform(-.45,.45)
            if not .35<x<s-.35 or not .35<y<s-.35:continue
            valid=True
            for o in walls+objects:
                xy=local_xy([x-o['position'][0],y-o['position'][1]],o.get('yaw',0))
                if math.hypot(max(0,abs(xy[0])-o['size'][0]/2),max(0,abs(xy[1])-o['size'][1]/2))<.3:valid=False;break
            if valid:agent['position']=[x,y,.25];break
        agent['yaw']+=r.uniform(-.3,.3)
    rotation=int(r.next()*4)*math.pi/2;mirror=r.next()<.5
    for item in walls+objects+agents:
        x,y,z=item['position'];angle=item.get('yaw',0)
        if mirror:x=s-x;angle=math.pi-angle
        dx=x-s/2;dy=y-s/2
        item['position']=[s/2+math.cos(rotation)*dx-math.sin(rotation)*dy,s/2+math.sin(rotation)*dx+math.cos(rotation)*dy,z]
        item['yaw']=angle+rotation
    return dict(format='hide-seek-physical-arena-v1',seed=int(seed),scenario=scenario,width=s,height=s,agents=agents,objects=objects,walls=walls)

def arena_xml(arena):
    def nums(v):return ' '.join(f'{float(x):.12g}' for x in v)
    pieces=['<mujoco model="Original hide and seek"><compiler angle="radian"/><option timestep=".005" gravity="0 0 -15" integrator="implicitfast" iterations="30"/><size njmax="2000" nconmax="300"/><default><geom friction=".5 .01 .001" condim="3" solref=".015 1"/><joint limited="false"/></default><asset><mesh name="ramp" vertex="-.5 -.5 -.5 .5 -.5 -.5 .5 -.5 .5 -.5 .5 -.5 .5 .5 -.5 .5 .5 .5" face="0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 0 3 5 0 5 2"/></asset><worldbody><geom name="floor" type="plane" size="20 20 .1" group="0"/>']
    for i,w in enumerate(arena['walls']):pieces.append(f'<geom name="wall-{i}" type="box" pos="{nums(w["position"])}" size="{nums(np.array(w["size"])/2)}" euler="0 0 {w.get("yaw",0)}" group="1"/>')
    for i,a in enumerate(arena['agents']):
        pieces.append(f'<body name="agent-{i}"><joint name="a{i}x" axis="1 0 0" type="slide" damping="12"/><joint name="a{i}y" axis="0 1 0" type="slide" damping="12"/><joint name="a{i}z" axis="0 0 1" type="slide" damping=".2"/><joint name="a{i}yaw" axis="0 0 1" type="hinge" damping=".8"/><geom name="agent-geom-{i}" type="sphere" size=".25" mass="1" friction=".12 .001 .001" group="2"/></body>')
    for i,o in enumerate(arena['objects']):
        dims=o['size'];g=f'type="box" size="{nums(np.array(dims)/2)}"'
        if o['kind']=='ramp':
            # Inline a per-ramp mesh because MJCF mesh scaling is defined in assets.
            vertices=[[-.5,-.5,-.5],[.5,-.5,-.5],[.5,-.5,.5],[-.5,.5,-.5],[.5,.5,-.5],[.5,.5,.5]]
            verts=nums(np.asarray(vertices).reshape(-1)*np.tile(dims,6))
            pieces.append(f'</worldbody><asset><mesh name="ramp-{i}" vertex="{verts}"/></asset><worldbody>')
            g=f'type="mesh" mesh="ramp-{i}"'
        pieces.append(f'<body name="object-{i}"><joint name="o{i}x" axis="1 0 0" type="slide" damping=".1"/><joint name="o{i}y" axis="0 1 0" type="slide" damping=".1"/><joint name="o{i}z" axis="0 0 1" type="slide" damping=".1"/><joint name="o{i}yaw" axis="0 0 1" type="hinge" damping=".1"/><geom name="object-geom-{i}" {g} mass="{o["mass"]}" group="3"/></body>')
    pieces.append('</worldbody><actuator>')
    for i in range(2):
        for axis,gear in [('x',24),('y',24),('yaw',2.5)]:pieces.append(f'<motor joint="a{i}{axis}" gear="{gear}"/>')
    pieces.append('</actuator><equality>')
    # Preallocated physical welds; activation preserves the current relative pose.
    for a in range(2):
        for i in range(len(arena['objects'])):pieces.append(f'<weld name="grab-{a}-{i}" body1="agent-{a}" body2="object-{i}" active="false" solref=".025 1"/>')
    for i in range(len(arena['objects'])):pieces.append(f'<weld name="lock-{i}" body1="object-{i}" active="false" solref=".015 1"/>')
    pieces.append('</equality></mujoco>');return ''.join(pieces)

def local_xy(v,angle):
    c=math.cos(angle);s=math.sin(angle);return [c*v[0]+s*v[1],-s*v[0]+c*v[1]]

class PhysicsEnv:
    def __init__(self,seed=1,scenario='shelter',size=8,n_boxes=3,n_ramps=1,arena=None,disable_tools=False,immovable=False,prep=PREP,play=PLAY,**kwargs):
        self.config=dict(scenario=scenario,size=size,n_boxes=n_boxes,n_ramps=n_ramps);self.disable_tools=disable_tools;self.immovable=immovable;self.prep=prep;self.play=play
        self.reset(seed,arena)
    def reset(self,seed=None,arena=None):
        self.arena=arena or generate_arena(seed if seed is not None else self.arena['seed'],**self.config)
        self.model=mujoco.MjModel.from_xml_string(arena_xml(self.arena));self.data=mujoco.MjData(self.model);self.n_obj=len(self.arena['objects']);self.rng=Random(self.arena['seed']^0x9e3779b9)
        self.agent_bodies=np.array([self.model.body(f'agent-{i}').id for i in range(2)]);self.agent_geoms=np.array([self.model.geom(f'agent-geom-{i}').id for i in range(2)])
        self.object_bodies=np.array([self.model.body(f'object-{i}').id for i in range(self.n_obj)],dtype=int);self.object_geoms=np.array([self.model.geom(f'object-geom-{i}').id for i in range(self.n_obj)],dtype=int)
        for i,a in enumerate(self.arena['agents']):self.data.qpos[i*4:i*4+4]=[*a['position'],a.get('yaw',0)]
        for i,o in enumerate(self.arena['objects']):self.data.qpos[8+i*4:12+i*4]=[*o['position'],o.get('yaw',0)]
        self.t=0;self.returns=np.zeros(2);self.hidden=0;self.grips=[-1,-1];self.locks=np.full(self.n_obj,-1);self.lock_events=np.zeros(2,dtype=int);self.grab_events=np.zeros(2,dtype=int);self.release_events=np.zeros(2,dtype=int);self.short_holds=np.zeros(2,dtype=int);self.grip_started=[0,0];self.path=np.zeros(2);self.collisions=np.zeros(2,dtype=int);self.shielded=0;self.visible=False;self.done=False;self.actions=np.zeros((2,6));self.jump_events=np.zeros(2,dtype=int);self.last_seen=[None,None]
        mujoco.mj_forward(self.model,self.data)
        if self.immovable:
            for i in range(self.n_obj):self._set_weld(2*self.n_obj+i,self.object_bodies[i],0);self.locks[i]=2
        self.initial_objects=self.data.qpos[8:].reshape(-1,4)[:,:3].copy();self._senses();return self.observe()
    def _ray(self,origin,direction,exclude=-1,groups=None):
        hit=np.array([-1],dtype=np.int32);dist=mujoco.mj_ray(self.model,self.data,np.asarray(origin,dtype=float),np.asarray(direction,dtype=float),np.array(groups or [1,1,1,1,1,1],dtype=np.uint8),True,int(exclude),hit)
        return float(dist),int(hit[0])
    def _sees(self,a,target_body,ignore_props=False):
        origin=self.data.xpos[self.agent_bodies[a]];target=self.data.xpos[target_body];d=target-origin;dist=np.linalg.norm(d);yaw=self.data.qpos[a*4+3]
        if dist>VISION_RANGE or dist<1e-8:return dist<1e-8
        if abs(math.atan2(math.sin(math.atan2(d[1],d[0])-yaw),math.cos(math.atan2(d[1],d[0])-yaw)))>FOV/2:return False
        _,hit=self._ray(origin,d/dist,self.agent_bodies[a],[1,1,1,0 if ignore_props else 1,1,1]);return hit>=0 and self.model.geom_bodyid[hit]==target_body
    def _senses(self):
        self.seen=np.zeros((2,2),dtype=bool);self.object_seen=np.zeros((2,self.n_obj),dtype=bool)
        for a in range(2):
            if a==1 and self.t<self.prep:continue
            self.seen[a,1-a]=self._sees(a,self.agent_bodies[1-a]);self.object_seen[a]=[self._sees(a,b) for b in self.object_bodies]
            if self.seen[a,1-a]:self.last_seen[a]=dict(position=self.data.xpos[self.agent_bodies[1-a]].tolist(),t=self.t)
        self.visible=bool(self.seen[1,0])
    def _set_weld(self,eq,b1,b2):
        p1=self.data.xpos[b1];p2=self.data.xpos[b2];R1=self.data.xmat[b1].reshape(3,3);R2=self.data.xmat[b2].reshape(3,3);q=np.empty(4);mujoco.mju_mat2Quat(q,(R1.T@R2).flatten())
        self.model.eq_data[eq,:3]=0;self.model.eq_data[eq,3:6]=R1.T@(p2-p1);self.model.eq_data[eq,6:10]=q;self.model.eq_data[eq,10]=1;self.data.eq_active[eq]=True
    def _distance(self,a,i):
        # Distance from sphere surface to oriented object bounding cuboid. Ramp contact still uses its actual mesh in MuJoCo.
        p=self.data.xpos[self.agent_bodies[a]]-self.data.xpos[self.object_bodies[i]];q=self.data.qpos[8+i*4+3];xy=local_xy(p,q);half=np.asarray(self.arena['objects'][i]['size'])/2
        return np.linalg.norm(np.maximum(np.abs([*xy,p[2]])-half,0))-.25
    def _tools(self,actions):
        for a in range(2):
            candidates=[i for i in range(self.n_obj) if self.object_seen[a,i] and self._distance(a,i)<.22]
            if a==1 and self.t<self.prep:candidates=[]
            old=self.grips[a];chosen=-1
            if not self.disable_tools and actions[a,3]>.5:
                if old>=0 and self.locks[old]<0:chosen=old
                else:
                    available=[i for i in candidates if self.locks[i]<0 and i not in self.grips]
                    if available:chosen=min(available,key=lambda i:self._distance(a,i))
            if old!=chosen:
                if old>=0:self._release(a)
                if chosen>=0:self._set_weld(a*self.n_obj+chosen,self.agent_bodies[a],self.object_bodies[chosen]);self.grab_events[a]+=1;self.grip_started[a]=self.t
                self.grips[a]=chosen
            for i in candidates:
                if self.disable_tools or self.locks[i]==2:continue
                if actions[a,4]>.5 and self.locks[i]<0:self._set_weld(2*self.n_obj+i,self.object_bodies[i],0);self.locks[i]=a;self.lock_events[a]+=1
                elif actions[a,4]<=.5 and self.locks[i]==a:self.data.eq_active[2*self.n_obj+i]=False;self.locks[i]=-1
        # Locking a prop must not weld its carrier to the world. This is an
        # interaction constraint, not a reward for choosing a tool strategy.
        for a,i in enumerate(self.grips):
            if i>=0 and self.locks[i]>=0:self._release(a)
    def _release(self,a):
        i=self.grips[a]
        if i<0:return
        self.data.eq_active[a*self.n_obj+i]=False
        self.release_events[a]+=1
        self.short_holds[a]+=self.t-self.grip_started[a]<=3
        self.grips[a]=-1
    def _jump(self,a,request):
        if request<=.5:return
        pos=self.data.qpos[4*a:4*a+3]
        dist,geom=self._ray(pos,[0,0,-1],self.agent_bodies[a],[1,1,0,1,0,0])
        if dist<0 or dist>.29 or abs(self.data.qvel[4*a+2])>.35:return
        if self.model.geom_group[geom]==1:return
        # 0.8m rise reaches 0.7m props. Cap the absolute apex below the
        # shortest wall even when launching from a prop. No midair jumps.
        ceiling=min(w['position'][2]+w['size'][2]/2 for w in self.arena['walls'])-.15
        rise=min(.8,ceiling-(pos[2]-.25))
        if rise<.1:return
        carried=self.grips[a]
        mass=1+(self.arena['objects'][carried]['mass'] if carried>=0 else 0)
        self.data.qvel[4*a+2]=math.sqrt(2*15*rise)/mass
        self.jump_events[a]+=1
    def step(self,actions):
        if self.done:return self.observe(),np.zeros(2),True,self.info()
        actions=np.asarray(actions,dtype=float).reshape(2,6).copy();actions[:,:3]=np.clip(actions[:,:3],-1,1)
        if not np.isfinite(actions).all():raise ValueError('Actions must be finite')
        if self.t<self.prep:actions[1]=0
        self.actions=actions;before=self.data.qpos[:8].reshape(2,4)[:,:3].copy();self._tools(actions)
        for a in range(2):
            self._jump(a,actions[a,5])
        for a in range(2):
            # Acceleration inputs are body-local, then rotate into world XY.
            yaw=self.data.qpos[4*a+3];c=math.cos(yaw);s=math.sin(yaw);x,y=actions[a,:2];self.data.ctrl[a*3:a*3+3]=[c*x-s*y,s*x+c*y,actions[a,2]]
        # MuJoCo runs the identical fixed-control substeps inside C.
        mujoco.mj_step(self.model,self.data,nstep=SUBSTEPS)
        hit=[False,False]
        for c in self.data.contact:
            if c.dist>0:continue
            for a in range(2):
                if self.agent_geoms[a] in (c.geom1,c.geom2) and (c.geom2 if c.geom1==self.agent_geoms[a] else c.geom1)!=0:hit[a]=True
        mujoco.mj_forward(self.model,self.data);self.path+=np.linalg.norm(self.data.qpos[:8].reshape(2,4)[:,:3]-before,axis=1);self.collisions+=hit
        was_prep=self.t<self.prep;self.t+=1;self._senses();reward=np.zeros(2)
        if not was_prep:
            reward[:]=[-1 if self.visible else 1,1 if self.visible else -1];self.hidden+=not self.visible
            shield=not self.visible and self._sees(1,self.agent_bodies[0],ignore_props=True);self.shielded+=shield
        self.returns+=reward;self.done=self.t>=self.prep+self.play
        if not np.isfinite(self.data.qpos).all() or np.max(np.abs(self.data.qvel))>150:raise RuntimeError('Physics diverged')
        return self.observe(),reward,self.done,self.info()
    def observe(self):
        rows=[]
        for a in range(2):
            pos=self.data.qpos[a*4:a*4+3];yaw=self.data.qpos[a*4+3];vel=self.data.qvel[a*4:a*4+4];lv=local_xy(vel,yaw)
            row=[lv[0]/5,lv[1]/5,vel[2]/5,vel[3]/8,pos[2]/2,min(1,self.t/max(1,self.prep)),max(0,self.t-self.prep)/self.play,1-a,float(self.grips[a]>=0),float(self.grips[a]>=0 and self.locks[self.grips[a]]>=0)]
            b=1-a
            if self.seen[a,b]:
                p=self.data.qpos[b*4:b*4+3]-pos;v=local_xy(self.data.qvel[b*4:b*4+2],yaw);xy=local_xy(p,yaw);ang=self.data.qpos[b*4+3]-yaw
                row += [1,xy[0]/6,xy[1]/6,p[2]/2,v[0]/5,v[1]/5,math.cos(ang),math.sin(ang)]
            else:row += [0]*8
            ids=[i for i in range(self.n_obj) if self.object_seen[a,i]];ids.sort(key=lambda i:np.linalg.norm(self.data.qpos[8+i*4:11+i*4]-pos))
            for k in range(OBJECT_SLOTS):
                if k>=len(ids):row += [0]*16;continue
                i=ids[k];o=self.arena['objects'][i];p=self.data.qpos[8+i*4:11+i*4]-pos;v=self.data.qvel[8+i*4:12+i*4];xy=local_xy(p,yaw);lv=local_xy(v,yaw);ang=self.data.qpos[11+i*4]-yaw
                row += [1,xy[0]/6,xy[1]/6,p[2]/2,lv[0]/5,lv[1]/5,v[2]/5,math.cos(ang),math.sin(ang),o['size'][0]/2,o['size'][1]/2,o['size'][2]/2,float(o['kind']=='ramp'),float(self.locks[i]==a),float(self.locks[i]>=0 and self.locks[i]!=a),float(self.grips[a]==i)]
            if a==1 and self.t<self.prep:row += [1]*LIDAR
            else:
                angles=yaw+np.arange(LIDAR)*math.tau/LIDAR
                rays=np.stack((np.cos(angles),np.sin(angles),np.zeros(LIDAR)),axis=-1).ravel()
                hits=np.empty(LIDAR,dtype=np.int32);distances=np.empty(LIDAR)
                mujoco.mj_multiRay(self.model,self.data,pos,rays,np.array([0,1,0,1,0,0],dtype=np.uint8),True,int(self.agent_bodies[a]),hits,distances,None,LIDAR,1e6)
                row += (np.where(distances<0,6,np.clip(distances,0,6))/6).tolist()
            rows.append(row)
        return np.asarray(rows,dtype=np.float32)
    def info(self):
        displacement=np.linalg.norm(self.data.qpos[8:].reshape(-1,4)[:,:3]-self.initial_objects,axis=1) if self.n_obj else np.zeros(0)
        return dict(t=self.t,hidden=int(self.hidden),visible=bool(self.visible),shielded=int(self.shielded),returns=self.returns.tolist(),grabs=self.grab_events.tolist(),jumps=self.jump_events.tolist(),releases=self.release_events.tolist(),short_holds=self.short_holds.tolist(),locks=self.lock_events.tolist(),locked=int(np.sum(self.locks>=0)),object_displacement=displacement.tolist(),path=self.path.tolist(),collisions=self.collisions.tolist(),prep=self.t<self.prep,play_steps=max(0,self.t-self.prep),reward_terms={'visibility':[-1 if self.visible else 1,1 if self.visible else -1],'preparation':self.t<=self.prep})
    def trace(self):
        return dict(t=self.t,qpos=self.data.qpos.tolist(),qvel=self.data.qvel.tolist(),grips=list(self.grips),locks=self.locks.tolist(),actions=self.actions.tolist(),info=self.info())
    def close(self):pass

if __name__=='__main__':
    e=PhysicsEnv();print('MuJoCo',mujoco.__version__,'OBS_DIM',OBS_DIM,'shape',e.observe().shape,'objects',e.n_obj)
    for _ in range(240):e.step(np.zeros((2,6)))
    print(e.info())
