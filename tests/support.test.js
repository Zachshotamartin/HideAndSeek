import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createCharacter } from '../src/visuals/character.js';
import { findFoothold, interpolateFrame, interpolateSupport, toLocal, toWorld, toWorldDirection } from '../src/visuals/support.js';
const vec=a=>new THREE.Vector3(...a), scale=.7/1.743;
const frame=(t,kind='box')=>({objectId:'carrier',kind,size:[2,.7,1.4],position:[2+t*.3,.35,3+t*.12],quaternion:new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),t*.7).toArray()});
function state(transform, localPosition=null, localHeading=0){
 const slope=transform.kind==='ramp'?.7/2:0,nz=1/Math.sqrt(1+slope*slope);
 localPosition ??= [.1,transform.kind==='ramp'?slope*.1+.25/nz:.6,0];
 const position=toWorld(vec(localPosition),transform),normal=toWorldDirection(vec([-slope*nz,nz,0]),transform),point=position.clone().add(new THREE.Vector3(0,-.25/nz,0));
 const forward=toWorldDirection(new THREE.Vector3(Math.sin(localHeading),0,Math.cos(localHeading)),transform);
 return{position:position.toArray(),forward:forward.toArray(),grounded:true,gripId:null,contacts:[],supportFrames:[transform],support:{objectId:transform.objectId,geomId:10,point:point.toArray(),normal:normal.toArray(),transform}};
}
const distance=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
function localFoot(pose,side,transform){return ['ankle','toe'].map(name=>toLocal(vec(pose.landmarks[name+side]),transform).toArray());}
function assertLegs(pose){for(const side of['L','R'])for(const [a,b]of[['hip','knee'],['knee','ankle']])assert.ok(Math.abs(distance(pose.landmarks[a+side],pose.landmarks[b+side])-.42*scale)<1e-6);}

test('passive translating and yawing support carries planted ankles and toes without starting a walk',()=>{
 for(const kind of ['box','ramp']){const character=createCharacter(0);let original;
 try{for(let i=0;i<=180;i++){const t=i/60,transform=frame(t,kind);character.update(state(transform),t);const pose=character.root.userData.pose;assertLegs(pose);
  const feet=['R','L'].map(side=>localFoot(pose,side,transform));if(!original)original=feet;
  for(let j=0;j<2;j++){assert.equal(pose.feet[j].swing,false);assert.equal(pose.feet[j].supportId,'carrier');for(let k=0;k<2;k++)assert.ok(distance(feet[j][k],original[j][k])<1e-7,'rendered foot drifted in carrier coordinates');}
 }}finally{character.dispose();}}
});

test('walking and turning on a moving support unlocks only the swinging foot',()=>{
 for (const kind of ['box', 'ramp']) {
 const character=createCharacter(0);let previous,previousTransform;const plants=[new Set(),new Set()];let swings=0;
 try{for(let i=0;i<210;i++){const t=i/60,transform=frame(t*.7,kind);character.update(state(transform,[.1,kind==='ramp'?.035+.25*Math.sqrt(1+.35**2):.6,Math.sin(t*.8)*.22],Math.sin(t)*.3),t);const pose=structuredClone(character.root.userData.pose);assertLegs(pose);
  for(let j=0;j<2;j++){const foot=pose.feet[j],side=j?'L':'R';if(foot.swing){swings++;assert.equal(foot.supportId,null);}else{plants[j].add(foot.plantId);if(previous&&!previous.feet[j].swing&&previous.feet[j].plantId===foot.plantId){const a=localFoot(previous,side,previousTransform),b=localFoot(pose,side,transform);for(let k=0;k<2;k++)assert.ok(distance(a[k],b[k])<1e-7);}}}
  previous=pose;previousTransform=transform;
 }assert.ok(swings>30);assert.ok(plants.every(set=>set.size>=3));}finally{character.dispose();}
 }
});

test('support interpolation keeps the plane on the same rotating prop transform used to draw the mesh',()=>{
 const a=frame(0,'ramp'),b=frame(1,'ramp'),localPoint=vec([.3,.7/2*.3,0]);
 const localNormal=vec([-.35,1,0]).normalize();
 for(const alpha of[0,.25,.5,.75,1]){const rendered=interpolateFrame(a,b,alpha),support={objectId:b.objectId,geomId:10,transform:b,point:toWorld(localPoint,b).toArray(),normal:toWorldDirection(localNormal,b).toArray()};const result=interpolateSupport(null,support,alpha,[rendered]);assert.ok(distance(toLocal(vec(result.point),rendered).toArray(),localPoint.toArray())<1e-10);assert.deepEqual(result.transform,rendered);}
});

test('finite ramp and box footholds stay on their surface at an edge, or choose reachable floor',()=>{
 for(const kind of['box','ramp']){const transform={...frame(0,kind),position:[0,.35,0],quaternion:[0,0,0,1]};const top=.7;
  const hit=findFoothold(vec([1.12,top,.3]),{frames:[transform],heading:Math.PI/2,scale,minHeight:top-.14,maxHeight:top+.14});assert.ok(hit,'a nearby finite foothold exists');assert.equal(hit.frame?.objectId,'carrier');const point=toLocal(hit.point,transform);assert.ok(point.x<1&&point.x>-1);assert.ok(Math.abs(point.z)<.7);assert.ok(Math.abs(point.y-(kind==='ramp'?.35*point.x:.35))<1e-9);
  const floor=findFoothold(vec([1.25,0,0]),{frames:[transform],heading:0,scale,minHeight:0,maxHeight:.1});assert.ok(floor);assert.equal(floor.frame,null);assert.equal(floor.point.y,0);
 }
});

test('floor footholds are not selected through a solid prop and attachment releases in actual flight',()=>{
 const transform=frame(0),blocked=findFoothold(vec([2,.0,3]),{frames:[transform],heading:0,scale,minHeight:0,maxHeight:.1});assert.equal(blocked,null,'the floor beneath a tall solid box is not a reachable foothold');
 const character=createCharacter(0);try{character.update(state(transform),0);const airborne={...state(transform),position:[2,1.3,3],grounded:false,support:null};character.update(airborne,.08);assert.ok(character.root.userData.pose.feet.every(f=>f.supportId===null));const before=character.root.userData.pose.landmarks;character.update({...airborne,supportFrames:[frame(1)]},.1);for(const side of['L','R'])assert.ok(distance(before['ankle'+side],character.root.userData.pose.landmarks['ankle'+side])<1e-7);}finally{character.dispose();}
});

 test('unreachable finite surfaces never become invented planted feet on reset or stop',()=>{
 const transform={...frame(0),size:[4,.7,4]}, character=createCharacter(0);
 try {for(let i=0;i<10;i++){character.update({...state(transform),position:[2,.25,3],support:null,groundHeight:0},i/60);const pose=character.root.userData.pose;assert.ok(pose.feet.every(f=>f.swing&&f.supportId===null));}}finally{character.dispose();}
 });
