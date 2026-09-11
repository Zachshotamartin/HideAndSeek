import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {createHash} from 'node:crypto';
import load from '../src/vendor/mujoco-csp.js';import{PhysicsSimulation,generateArena,validateArena,editObject,OBS_DIM}from'../src/core/physics.js';
const mj=await load({wasmBinary:fs.readFileSync(new URL('../node_modules/@mujoco/mujoco/mujoco.wasm',import.meta.url))});
const near=(a,b,tol=3e-6)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<tol,`${i}: ${a[i]} versus ${b[i]}`);};
test('native and browser MuJoCo share real action, collision, grab, lock, observation trajectories',()=>{
 const f=JSON.parse(fs.readFileSync(new URL('../training/fixtures/native-physics.json',import.meta.url)));
 assert.equal(f.observationSize,OBS_DIM);assert.equal(f.actionSize,6);
 for(const [name,hash]of Object.entries(f.sources))assert.equal(createHash('sha256').update(fs.readFileSync(new URL('../training_v4/'+name,import.meta.url))).digest('hex'),hash,`Stale native source: ${name}`);
 const s=new PhysicsSimulation(mj,f.arena,{prep:f.prep,play:f.play});
 try{f.actions.forEach((actions,i)=>{s.step(actions);near(s.data.qpos,f.frames[i].qpos);near(s.data.qvel,f.frames[i].qvel,1e-5);for(let a=0;a<2;a++)near(s.observe(a),f.frames[i].obs[a],3e-5);assert.deepEqual(s.grips,f.frames[i].grips);assert.deepEqual(s.locks,f.frames[i].locks);});assert.ok(s.grabEvents[0]>0);assert.ok(s.lockEvents[0]>0);}finally{s.dispose();}
});
test('seeker is blind during prep and hidden opponent coordinates cannot leak into observations',()=>{
 const arena=generateArena(1,'open',8,0,0),s=new PhysicsSimulation(mj,arena);
 try{const before=s.observe(1);assert.equal(before.length,OBS_DIM);s.data.qpos[0]=3.1;s.data.qpos[1]=3.4;mj.mj_forward(s.model,s.data);s.senses();near(s.observe(1),before);assert.equal(s.lastSeen[1],null);
 s.t=s.prep;s.data.qpos[7]=0;s.data.qpos[0]=1;s.senses();const hidden=s.observe(1);s.data.qpos[0]=2;mj.mj_forward(s.model,s.data);s.senses();near(s.observe(1),hidden);assert.equal(s.seen[1][0],false);}finally{s.dispose();}
});
test('continuous diagonal acceleration is bounded and cannot tunnel through walls',()=>{
 const s=new PhysicsSimulation(mj,generateArena(7,'open',8,0,0),{prep:0,play:240});
 try{const start=Array.from(s.data.qpos);for(let i=0;i<120;i++)s.step([[1,1,.03,0,0],[0,0,0,0,0]]);assert.ok(s.path[0]>2);assert.ok(s.data.qpos[0]<=7.76&&s.data.qpos[1]<=7.76);assert.ok(Math.abs(s.data.qpos[0]-start[0])>.1);assert.ok(s.collisions[0]>0);assert.ok(Math.hypot(s.data.qvel[0],s.data.qvel[1])<3);}finally{s.dispose();}
});
test('physical lock, ownership, release, and no-tool ablation keep real push contacts',()=>{
 const f=JSON.parse(fs.readFileSync(new URL('../training/fixtures/native-physics.json',import.meta.url)));const s=new PhysicsSimulation(mj,f.arena,{disableTools:true,prep:12,play:52});
 try{for(const a of f.actions)s.step(a);assert.deepEqual(s.grabEvents,[0,0]);assert.deepEqual(s.lockEvents,[0,0]);assert.ok(s.info().object_displacement.some(x=>x>.05),'push contacts still displace a prop');}finally{s.dispose();}
});
test('seeded generation, arena validation and unsafe imports',()=>{
 assert.deepEqual(generateArena(15),generateArena(15));assert.notDeepEqual(generateArena(15).objects,generateArena(16).objects);
 const a=generateArena(15);assert.equal(validateArena(a).seed,15);assert.throws(()=>validateArena({...a,objects:Array(11).fill(a.objects[0])}));assert.throws(()=>validateArena({...a,width:Infinity}));
 assert.throws(()=>editObject(a,0,{...a.objects[0],position:a.agents[0].position}));
});
test('a supported agent climbs an actual solid ramp, with height and support contacts',()=>{
 const a=generateArena(4,'open',8,0,0);a.agents=[{position:[2.6,4,.25],yaw:0},{position:[6.5,6.5,.25],yaw:0}];a.objects=[{id:'ramp-a',kind:'ramp',position:[4,4,.353],size:[1.7,1.2,.7],yaw:0,mass:1.2}];const s=new PhysicsSimulation(mj,a,{prep:0,play:100,immovable:true});let supportedHigh=false,maxHeight=0;
 try{for(let i=0;i<50;i++){s.step([[1,0,0,0,0],[0,0,0,0,0]]);maxHeight=Math.max(maxHeight,s.agents[0].position[1]);supportedHigh ||= s.agents[0].grounded&&s.agents[0].position[1]>.55;}assert.ok(maxHeight>.65,`maximum height ${maxHeight}`);assert.ok(supportedHigh,'ramp support is recognized above floor level');}finally{s.dispose();}
});

test('presentation support recognizes tiny floor clearance without inventing physical contacts',()=>{
 const s=new PhysicsSimulation(mj,generateArena(2709,'open',8,0,0));
 try{
  for(const height of [.2505,.2545,.264]){
   s.data.qpos[2]=height;s.data.qvel.fill(0);mj.mj_forward(s.model,s.data);
   const qpos=Array.from(s.data.qpos),qvel=Array.from(s.data.qvel),obs=Array.from(s.observe(0)),ncon=s.data.ncon;
   s.syncView();const a=s.agents[0];assert.equal(a.grounded,true);assert.equal(a.support.source,'proximity');near(a.support.point,[a.position[0],0,a.position[2]]);near(a.support.normal,[0,1,0]);assert.ok(Math.abs(a.support.gap-(height-.25))<1e-8);
   assert.deepEqual(Array.from(s.data.qpos),qpos);assert.deepEqual(Array.from(s.data.qvel),qvel);assert.deepEqual(Array.from(s.observe(0)),obs);assert.equal(s.data.ncon,ncon,'presentation query does not generate contacts');
  }
  for(const [height,vz]of[[.28,0],[.75,-1],[.254,1]]){s.data.qpos[2]=height;s.data.qvel[2]=vz;mj.mj_forward(s.model,s.data);s.syncView();assert.equal(s.agents[0].grounded,false,'real separation or rising motion remains airborne');}
 }finally{s.dispose();}
});

test('presentation support uses the actual box top and the slope-normal sphere gap of a rotated ramp',()=>{
 for(const kind of ['box','ramp']){
  const a=generateArena(4,'open',8,0,0),yaw=.7;
  a.objects=[{id:'support',kind,position:[4,4,.353],size:[1.7,1.2,.7],yaw,mass:1.2}];
  const s=new PhysicsSimulation(mj,a,{immovable:true});
  try{
   const slope=kind==='ramp'?.7/1.7:0,nz=1/Math.sqrt(1+slope*slope),surface=kind==='ramp'?.353:.703;
   s.data.qpos.set([4,4,surface+.258/nz],0);s.data.qvel.fill(0);mj.mj_forward(s.model,s.data);s.syncView();
   const agent=s.agents[0];assert.equal(agent.grounded,true);assert.equal(agent.support.source,'proximity');assert.ok(Math.abs(agent.support.gap-.008)<1e-6);assert.ok(Math.abs(agent.support.point[1]-surface)<1e-6);assert.ok(Math.abs(agent.support.normal[1]-nz)<1e-6);
   if(kind==='ramp'){assert.ok(Math.abs(agent.support.normal[0]+slope*nz*Math.cos(yaw))<1e-6);assert.ok(Math.abs(agent.support.normal[2]+slope*nz*Math.sin(yaw))<1e-6);}
  }finally{s.dispose();}
 }
});
test('locks have ownership and physical tools cannot be acquired through a wall',()=>{
 const a=generateArena(4,'open',8,0,0);a.agents=[{position:[3.39,4,.25],yaw:0},{position:[4.61,4,.25],yaw:Math.PI}];a.objects=[{id:'box-a',kind:'box',position:[4,4,.353],size:[.7,.7,.7],yaw:0,mass:1.2}];const s=new PhysicsSimulation(mj,a,{prep:0,play:100});
 try{s.step([[0,0,0,0,1],[0,0,0,0,0]]);assert.equal(s.locks[0],0);s.step([[0,0,0,0,1],[0,0,0,0,0]]);assert.equal(s.locks[0],0,'other agent cannot unlock');s.step([[0,0,0,0,0],[0,0,0,0,0]]);assert.equal(s.locks[0],-1);}finally{s.dispose();}
 a.walls.push({position:[3.58,4,.45],size:[.08,1,.9],yaw:0});const blocked=new PhysicsSimulation(mj,a,{prep:0,play:100});try{assert.equal(blocked.objectSeen[0][0],false);blocked.step([[0,0,0,1,1],[0,0,0,0,0]]);assert.equal(blocked.grips[0],-1);assert.equal(blocked.locks[0],-1);}finally{blocked.dispose();}
});

test('native and WASM agree on airborne jump trajectories and landing on a box',()=>{
 const f=JSON.parse(fs.readFileSync(new URL('./fixtures/jump-native.json',import.meta.url)));
 const s=new PhysicsSimulation(mj,f.arena,{prep:f.prep,play:f.play});
 try{
  for(const row of f.frames){s.step(row.action);near(s.data.qpos,row.qpos);near(s.data.qvel,row.qvel,1e-5);for(let a=0;a<2;a++)near(s.observe(a),row.obs[a],3e-5);assert.deepEqual([...s.jumpEvents],row.jumps);}
  assert.equal(s.jumpEvents[0],1);assert(s.data.qpos[2]>.9,'lands on the prop');
 }finally{s.dispose();}
});
