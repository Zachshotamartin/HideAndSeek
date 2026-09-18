# Dependencies and research references

The experiment's environment, renderer, training code, and policy weights are original project work.

- **MuJoCo 3.13.0** — Google DeepMind, Apache License 2.0. The native training environment and the browser's `@mujoco/mujoco` WebAssembly dependency use the same physics engine. [Source and license](https://github.com/google-deepmind/mujoco).
- **Emscripten binding glue** — Emscripten Authors, MIT / University of Illinois-NCSA dual license. Two invokers use the standard non-dynamic calling approach to support the site's Content Security Policy. The generated module, full licenses, pinned source references, and reproduction instructions are in `src/vendor/`; the physics WASM is unmodified.
- **Three.js** — MIT License. Used to render the simulated state. [Source and license](https://github.com/mrdoob/three.js).
- **PyTorch** — BSD-style license. Offline policy training only. [Source and license](https://github.com/pytorch/pytorch).

The OpenAI paper [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) inspired the combination of visibility rewards, a preparation phase, limited observations, grabbing, and locking. Its source implementation, released policies, and trained parameters are not distributed or used by this project.
