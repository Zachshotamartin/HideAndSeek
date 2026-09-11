import createMuJoCo from '../vendor/mujoco-csp.js';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
let enginePromise;
/** Single shared compiled engine; each mount owns and deletes its Model and Data. */
export function loadPhysics() {
  return (enginePromise ??= createMuJoCo({
    locateFile: (path) => (path.endsWith('.wasm') ? wasmUrl : path),
  }).catch((error) => {
    enginePromise = null;
    throw error;
  }));
}
