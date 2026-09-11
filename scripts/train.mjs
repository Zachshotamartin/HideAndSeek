import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
const candidates=[process.env.PYTHON_BIN,'.venv-physics/bin/python','/opt/homebrew/bin/python3','python3'].filter(Boolean);
const python=candidates.find(bin=>{if(bin.includes('/')&&!existsSync(bin))return false;const check=spawnSync(bin,['-c','import mujoco,torch,numpy; print("physical-training-ready")'],{encoding:'utf8'});return check.status===0&&check.stdout.includes('physical-training-ready');});
if(!python){console.error('Install training/requirements.txt into a Python virtual environment, then set PYTHON_BIN to its Python executable.');process.exit(1);}
const args=['training/train_physics.py',...process.argv.slice(2)];const result=spawnSync(python,args,{stdio:'inherit'});if(result.error||result.status!==0){console.error(result.error||`Training exited ${result.status}`);process.exit(result.status||1);}
