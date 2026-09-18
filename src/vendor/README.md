# MuJoCo CSP-compatible JavaScript bindings

`mujoco-csp.js` is generated from the exact `@mujoco/mujoco@3.13.0` npm JavaScript
module. Its physics WASM is imported from that pinned package without modification.
The upstream glue compiles two binding wrappers from strings. This version uses
ordinary closures for those wrappers so `script-src 'self' 'wasm-unsafe-eval'`
works without permitting JavaScript string compilation.

Run `node scripts/build-mujoco-csp.mjs` to reproduce it; add `--check` to verify
the checked-in result. The generator rejects a different upstream JS or WASM hash.
This is a source-level adaptation of the standard Emscripten non-dynamic path,
not a claim that the WASM was rebuilt with `DYNAMIC_EXECUTION=0`.

Reference: Emscripten commit `a1fe3902bf73a3802eae0357d273d0e37ea79898` (3.1.64),
[Embind lines 822–890](https://github.com/emscripten-core/emscripten/blob/a1fe3902bf73a3802eae0357d273d0e37ea79898/src/embind/embind.js#L822)
and [Emval lines 344–358](https://github.com/emscripten-core/emscripten/blob/a1fe3902bf73a3802eae0357d273d0e37ea79898/src/embind/emval.js#L344).
The adapters retain this npm build's optional argument validation, synchronous
binding assertion, void-return handling, destructor order and argument packing.
They use per-call arrays so callbacks can safely reenter a binding. Function names
are assigned with `Object.defineProperty`, and constructors use `Reflect.construct`.

MuJoCo is Apache-2.0; Emscripten's invokers are MIT / University of Illinois-NCSA
dual licensed. Complete license texts are included here. Changes are marked above
and in the adapter templates under `scripts/vendor/`.
