"""Ground-plane hide-and-seek simulator; constants mirrored by src/core/arena.js.
The policies receive only observe(), never grids or the opponent's hidden position.
"""
import math, hashlib, json
import numpy as np

MAX_SIZE = 24
VISION = 7.0
MEMORY_STEPS = 60
PREP_STEPS = 24
PLAY_STEPS = 180
SPEEDS = np.array([.30, .34], dtype=np.float32)
DIRECTIONS = np.array([[0,0],[0,-1],[1,0],[0,1],[-1,0]], dtype=np.float32)
RAYS = np.array([[math.cos(a*math.pi/4),math.sin(a*math.pi/4)] for a in range(8)], dtype=np.float32)
OBS_SIZE = 23

class Random:
    def __init__(self, seed): self.n = int(seed) & 0xffffffff
    def next(self):
        n=self.n; n ^= (n << 13)&0xffffffff; n ^= n >> 17; n ^= (n << 5)&0xffffffff; self.n=n&0xffffffff
        return self.n / 4294967296
    def integer(self,n): return min(n-1, int(self.next()*n))

def connected(grid):
    free=np.argwhere(grid==0)
    if len(free)==0:return False
    stack=[tuple(free[0])];seen={stack[0]}
    while stack:
        y,x=stack.pop()
        for dy,dx in [(0,1),(1,0),(0,-1),(-1,0)]:
            cell=(y+dy,x+dx)
            if 0<=cell[0]<grid.shape[0] and 0<=cell[1]<grid.shape[1] and grid[cell]==0 and cell not in seen:seen.add(cell);stack.append(cell)
    return len(seen)==len(free)

def line_clear(grid,a,b):
    # Exact closed-cell slab intersections; touching a solid corner occludes.
    ys,xs=np.where(grid>0)
    low=np.stack([xs,ys],-1);high=low+1
    delta=np.asarray(b)-np.asarray(a)
    safe=np.where(np.abs(delta)<1e-12,1e-12,delta)
    t1=(low-a)/safe;t2=(high-a)/safe
    near=np.maximum(np.minimum(t1,t2).max(-1),0)
    far=np.minimum(np.maximum(t1,t2).min(-1),1)
    return not (near<=far).any()

def generate_arena(seed,width=14,height=12,count=6,spawn_seed=None):
    rng=Random(seed or 1);grid=np.zeros((height,width),dtype=np.uint8);grid[[0,-1],:]=1;grid[:,[0,-1]]=1;blocks=[]
    for _ in range(count):
        for _attempt in range(24):
            bw=1+rng.integer(3);bh=1+rng.integer(3);x=2+rng.integer(max(1,width-bw-3));y=2+rng.integer(max(1,height-bh-3))
            if grid[y:y+bh,x:x+bw].any():continue
            grid[y:y+bh,x:x+bw]=1
            if connected(grid):blocks.append({'x':x,'y':y,'width':bw,'height':bh});break
            grid[y:y+bh,x:x+bw]=0
    if spawn_seed is not None:rng=Random(spawn_seed)
    free=np.argwhere(grid==0)
    seeker=np.array(free[rng.integer(len(free))][::-1],dtype=np.float32)+.5;hider=None
    for _ in range(200):
        candidate=np.array(free[rng.integer(len(free))][::-1],dtype=np.float32)+.5;distance=np.linalg.norm(candidate-seeker)
        if 3<=distance<=6 and line_clear(grid,candidate,seeker):hider=candidate;break
    if hider is None:
        candidates=[p for p in free if np.linalg.norm(np.array(p[::-1])+.5-seeker)>2]
        hider=np.array(candidates[rng.integer(len(candidates))][::-1],dtype=np.float32)+.5
    return {'seed':int(seed),'width':width,'height':height,'blocks':blocks,'spawns':[hider.tolist(),seeker.tolist()]},grid

def arena_hash(arena):
    geometry=[arena['width'],arena['height'],sorted([[b['x'],b['y'],b['width'],b['height']] for b in arena['blocks']])]
    return hashlib.sha256(json.dumps(geometry,separators=(',',':')).encode()).hexdigest()

class BatchArena:
    def __init__(self,n=32,seed=123,stage=0):
        self.map_hashes=set();self.n=n;self.rng=np.random.default_rng(seed);self.stage=stage
        self.grid=np.ones((n,MAX_SIZE,MAX_SIZE),dtype=np.uint8);self.pos=np.zeros((n,2,2),np.float32);self.vel=np.zeros_like(self.pos)
        self.size=np.zeros((n,2),np.float32);self.t=np.zeros(n,np.int32);self.prep=np.zeros(n,np.int32)
        self.memory=np.zeros_like(self.pos);self.age=np.full((n,2),MEMORY_STEPS,np.int32);self.known=np.zeros((n,2),bool);self.visible=np.zeros(n,bool)
        self.ever=np.zeros(n,bool);self.visited=np.zeros((n,MAX_SIZE,MAX_SIZE),bool);self.novelty=np.zeros(n,np.float32)
        self.hidden=np.zeros(n,np.int32);self.collisions=np.zeros((n,2),np.int32);self.distance=np.zeros((n,2),np.float32);self.returns=np.zeros((n,2),np.float32)
        self.rects=np.zeros((n,12,4),np.float32);self.rect_count=np.zeros(n,np.int32)
        self.arenas=[None]*n
        for i in range(n):self.reset(i)
    def reset(self,i,arena=None):
        if arena is None:
            if self.stage==0:w=int(self.rng.integers(9,13));h=int(self.rng.integers(9,13));count=int(self.rng.integers(0,3));prep=0
            else:w=int(self.rng.integers(10,19));h=int(self.rng.integers(10,17));count=int(self.rng.integers(2,10));prep=PREP_STEPS
            arena,g=generate_arena(int(self.rng.integers(1,1000000)),w,h,count)
        else:
            w=arena['width'];h=arena['height'];prep=arena.get('prep',PREP_STEPS);g=np.zeros((h,w),np.uint8);g[[0,-1],:]=1;g[:,[0,-1]]=1
            for block in arena['blocks']:g[block['y']:block['y']+block['height'],block['x']:block['x']+block['width']]=1
        self.map_hashes.add(arena_hash(arena))
        self.rects[i]=0;self.rect_count[i]=len(arena['blocks'])
        for j,b in enumerate(arena['blocks']):self.rects[i,j]=[b['x'],b['y'],b['x']+b['width'],b['y']+b['height']]
        self.grid[i]=1;self.grid[i,:h,:w]=g;self.size[i]=[w,h];self.pos[i]=arena['spawns'];self.vel[i]=0;self.t[i]=0;self.prep[i]=prep
        self.memory[i]=0;self.age[i]=MEMORY_STEPS;self.known[i]=False;self.ever[i]=False;self.visited[i]=False;self.novelty[i]=0;self.hidden[i]=0;self.collisions[i]=0;self.distance[i]=0;self.returns[i]=0;self.arenas[i]=arena
        self.update_sight(indices=np.array([i]), age=False)
    def occupancy(self,points):
        # points shape [B,...,2]
        indices=np.floor(points).astype(np.int32);indices=np.clip(indices,0,MAX_SIZE-1)
        batch=np.arange(self.n).reshape((self.n,)+(1,)*(points.ndim-2))
        return self.grid[batch,indices[...,1],indices[...,0]].astype(bool)
    def sight(self):
        delta=self.pos[:,1]-self.pos[:,0];distance=np.linalg.norm(delta,axis=-1)
        safe=np.where(np.abs(delta)<1e-12,1e-12,delta)
        t1=(self.rects[:,:,:2]-self.pos[:,0,None,:])/safe[:,None,:]
        t2=(self.rects[:,:,2:]-self.pos[:,0,None,:])/safe[:,None,:]
        near=np.maximum(np.minimum(t1,t2).max(-1),0)
        far=np.minimum(np.maximum(t1,t2).min(-1),1)
        hits=(near<=far)&(np.arange(12)[None,:]<self.rect_count[:,None])
        return (distance<=VISION)&~hits.any(axis=1),distance

    def update_sight(self, indices=None, age=True):
        self.visible,_=self.sight()
        for role in range(2):
            seen=self.visible & ((self.t>=self.prep) if role==1 else True)
            if indices is not None:
                mask=np.zeros(self.n,bool);mask[indices]=True;seen &= mask
            if age:self.age[:,role]=np.minimum(MEMORY_STEPS,self.age[:,role]+1)
            self.memory[seen,role]=self.pos[seen,1-role];self.known[seen,role]=True;self.age[seen,role]=0
    def observe(self):
        distances=np.arange(1,13,dtype=np.float32)*.35
        raypoints=self.pos[:,:,None,None,:]+RAYS[None,None,:,None,:]*distances[None,None,None,:,None]
        hits=self.occupancy(raypoints);first=np.where(hits.any(-1),hits.argmax(-1)+1,12).astype(np.float32)/12
        outputs=[]
        for role in range(2):
            seen=self.visible&((self.t>=self.prep) if role==1 else True);known=self.known[:,role]&(self.age[:,role]<MEMORY_STEPS)
            if role==1:known&=self.t>=self.prep
            relative=np.clip((self.pos[:,1-role]-self.pos[:,role])/VISION,-1,1)*seen[:,None]
            memory=np.clip((self.memory[:,role]-self.pos[:,role])/VISION,-1,1)*known[:,None]
            remaining=np.clip((PLAY_STEPS-(self.t-self.prep))/PLAY_STEPS,0,1)
            prep=np.maximum(0,self.prep-self.t)/PREP_STEPS
            out=np.concatenate([first[:,role],seen[:,None],relative,known[:,None],memory,self.age[:,role,None]/MEMORY_STEPS,self.vel[:,role]/SPEEDS[role],self.pos[:,role]/self.size,self.size/MAX_SIZE,remaining[:,None],prep[:,None]],axis=1)
            outputs.append(out.astype(np.float32))
        return np.stack(outputs,axis=1)
    def step(self,actions,auto_reset=True):
        old_visible,old_distance=self.sight();active=self.t>=self.prep;oldpos=self.pos.copy();oldknown=self.ever.copy()
        motion=DIRECTIONS[actions]*SPEEDS[None,:,None];motion[~active,1]=0
        corners=np.array([[-.24,-.24],[-.24,.24],[.24,-.24],[.24,.24]],np.float32)
        for axis in range(2):
            target=self.pos.copy();target[:,:,axis]+=motion[:,:,axis]
            blocked=self.occupancy(target[:,:,None,:]+corners[None,None]).any(-1)
            self.pos[:,:,axis]=np.where(blocked,self.pos[:,:,axis],target[:,:,axis]);self.collisions+=blocked&(np.abs(motion[:,:,axis])>0)
        self.vel=self.pos-oldpos;self.distance+=np.linalg.norm(self.vel,axis=-1)
        visible,distance=self.sight();tag=active&visible&(distance<.65)
        rewards=np.zeros((self.n,2),np.float32)
        both=old_visible&visible&active
        delta=(old_distance-distance)/VISION
        rewards[:,1]=np.where(active,-.001,0)+both*delta*.15
        rewards[:,1]+=(active&visible&~oldknown)*.03
        self.ever|=visible&active
        cell=np.floor(self.pos[:,1]).astype(int);new=~self.visited[np.arange(self.n),cell[:,1],cell[:,0]]&active&(self.novelty<.15)
        rewards[:,1]+=new*.003;self.novelty+=new*.003;self.visited[np.arange(self.n),cell[:,1],cell[:,0]]=True
        rewards[:,0]=np.where(active,np.where(visible,-.002,.002),0)-both*delta*.05
        self.hidden+=active&~visible
        self.t+=1;timeout=self.t>=self.prep+PLAY_STEPS;done=tag|timeout
        rewards[tag,0]-=2;rewards[tag,1]+=2;rewards[timeout&~tag,0]+=2;rewards[timeout&~tag,1]-=2
        self.returns+=rewards;self.update_sight()
        infos=[]
        for i in np.flatnonzero(done):
            infos.append({'index':int(i),'capture':bool(tag[i]),'hidden_fraction':float(self.hidden[i]/max(1,self.t[i]-self.prep[i])),'play_steps':int(self.t[i]-self.prep[i]),'returns':self.returns[i].tolist(),'distance':self.distance[i].tolist(),'collisions':self.collisions[i].tolist(),'arena':self.arenas[i]})
            if auto_reset:self.reset(int(i))
        return self.observe(),rewards,done,infos
