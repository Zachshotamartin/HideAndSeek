/** Frozen PPO actor inference. Inputs come only from Simulation.observe(role). */
export function validateModel(data) {
  if (data?.format !== 'hide-seek-ppo-v1' || !Array.isArray(data.policies) || data.policies.length !== 2) throw new Error('This is not a Hide and Seek model.');
  for (const p of data.policies) {
    if (!Array.isArray(p.layers) || p.layers.length !== 3) throw new Error('A policy needs three layers.');
    const sizes = [[23, 48], [48, 48], [48, 5]];
    p.layers.forEach((l, i) => {
      if (!l || l.input !== sizes[i][0] || l.output !== sizes[i][1] || !Array.isArray(l.weights) || l.weights.length !== l.input * l.output || !Array.isArray(l.bias) || l.bias.length !== l.output || ![...l.weights, ...l.bias].every(v => Number.isFinite(v) && Math.abs(v) <= 100)) throw new Error('Model layer dimensions or values are invalid.');
    });
  }
  return { format: data.format, label: typeof data.label === 'string' ? data.label.slice(0, 80) : 'Imported model', policies: data.policies.map(p => ({ layers: p.layers.map(l => ({ input: l.input, output: l.output, weights: l.weights.slice(), bias: l.bias.slice() })) })) };
}
export function logits(policy, observation) {
  if (observation.length !== 23) throw new Error('Policy requires 23 observations.');
  let x = observation;
  policy.layers.forEach((l, index) => { const y = new Float64Array(l.output); for (let o = 0; o < l.output; o++) { let v = l.bias[o]; for (let i = 0; i < l.input; i++) v += l.weights[o * l.input + i] * x[i]; y[o] = index < 2 ? Math.tanh(v) : v; } x = y; }); return Array.from(x);
}
export function chooseAction(policy, observation, random) {
  const values = logits(policy, observation), max = Math.max(...values), probs = values.map(x => Math.exp(x - max)), total = probs.reduce((a, b) => a + b, 0); let n = random.next() * total;
  for (let i = 0; i < probs.length; i++) { n -= probs[i]; if (n < 0) return i; } return 4;
}
