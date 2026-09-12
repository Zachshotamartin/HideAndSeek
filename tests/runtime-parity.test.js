import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createPolicyControllers} from '../src/core/policyController.js';
const root=new URL('../',import.meta.url);
const read=file=>readFileSync(new URL(file,root));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const manifest=JSON.parse(read('public/models/MANIFEST.json'));
const near=(a,b,name)=>{
  assert.equal(a.length,b.length,name);
  a.forEach((v,i)=>assert(Number.isFinite(v)&&Math.abs(v-b[i])<=1e-5,`${name}[${i}]: ${v} versus ${b[i]}`));
};
for(const entry of manifest.checkpoints)test(`${entry.id} v4 policy matches independent PyTorch recurrence and sampled jump/tool decisions`,()=>{
  const bytes=read(entry.parityFixture);
  assert.equal(hash(bytes),entry.parityFixtureSHA256);
  const fixture=JSON.parse(bytes),modelBytes=read('public/models/'+entry.file);
  assert.equal(fixture.modelSHA256,hash(modelBytes));
  for(const [file,sha]of Object.entries(fixture.sources))assert.equal(hash(read('training_v4/'+file)),sha,`Native fixture source changed: ${file}`);
  const policies=createPolicyControllers(JSON.parse(modelBytes));let rows=0,blind=0,resets=0;
  for(const rollout of fixture.rollouts){
    const states=policies.map(p=>p.initialState());
    for(const row of rollout.rows){
      const policy=policies[row.role];if(row.reset){states[row.role]=policy.initialState();resets++;}
      let used=0;
      const result=policy.act(Float32Array.from(row.observation),states[row.role],{deterministic:rollout.deterministic,random:()=>{assert(used<row.tape.length);return row.tape[used++];}});
      assert.equal(used,rollout.deterministic?0:10);
      for(const key of ['action','memory','mean','toolLogits'])near(result[key],row[key],key);
      assert.deepEqual([...result.commands],row.commands);assert.deepEqual([...result.buttons],row.buttons);
      assert.equal(result.action.length,6);
      if(row.observation[7]<.5&&row.observation[5]<1){assert(result.action.every(v=>v===0));assert(result.buttons.every(v=>v===0));blind++;}
      states[row.role]=result.state;rows++;
    }
  }
  assert.deepEqual({rows,blind,resets},fixture.counts);
  assert(rows>=500&&blind>=40&&resets>=8,'parity coverage must include hundreds of rows, dozens of blind rows and repeated resets');
});
