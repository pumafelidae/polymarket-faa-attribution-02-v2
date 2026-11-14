// Polymarket FAA outage attribution forecaster — v2
// Node 22, stdlib only, no network, deterministic.
//
// Model (tuned 2026-11-14 on the NVIDIA practice cases + FAA backtest checkpoints
// cut from Polymarket's own public price history):
//  - One-sided EWMA volatility (half-life 30 min) over up-moves and down-moves
//    separately, floored at 0.0003 and capped at 0.05 per minute. Prices move in
//    steps and hug the 0 boundary, so up and down risk are not symmetric.
//  - Drift from the endpoint slope over the last 180 minutes, shrunk x0.3 and
//    capped at 0.00015/min, then gated by recent volatility (scaled by
//    min(1, sigma/0.002)): in quiet stretches the drift is mostly noise, so it
//    is suppressed; when the market is actually moving, the drift is kept.
//    The drift's effect saturates with horizon (tau = 180 min), so a 4-hour
//    trend is followed but not extrapolated linearly forever.
//  - Quantiles use narrow z-scores (-0.85, -0.4, 0, 0.4, 0.85) times the
//    one-sided sigma * sqrt(h). The EWMA sigma is inflated for hours after a
//    big jump (a jump is a level shift, not a volatility regime), so the
//    z-scores are tightened to compensate; the backtest preferred this over
//    winsorizing the diffs or blending in empirical quantiles.
//  - No empirical-quantile blend: with only ~10 hours of FAA history, the
//    empirical tails are dominated by the Nov-13 evening jumps and widen every
//    later forecast; the pure parametric form scored better on both backtests.
//  - Rows are clamped to [0, 1], forced monotone, rounded to 6 decimals.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';

function slash(p) {
  return p.endsWith('/') ? p : p + '/';
}
const INPUT_DIR = slash(process.env.MAKEFUN_INPUT_DIR || '/input/');
const OUTPUT_DIR = slash(process.env.MAKEFUN_OUTPUT_DIR || '/output/');
const HORIZON = 240;
const Z = [-0.85, -0.4, 0, 0.4, 0.85];
const HL = 30; // EWMA half-life, minutes
const VOL_FLOOR = 0.0003;
const VOL_CAP = 0.05;
const DRIFT_WIN = 180; // minutes
const DRIFT_SHRINK = 0.3;
const DRIFT_CAP = 0.00015; // per minute
const DRIFT_TAU = 180; // minutes, horizon decay of the drift effect
const DRIFT_GATE_REF = 0.002; // sigma at/above which drift is ungated

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function round6(v) {
  return Math.round(v * 1e6) / 1e6;
}

// EWMA variance of a diff series (diffs include zeros). alpha from half-life.
function ewmaVar(diffs, hl) {
  const a = 1 - Math.pow(0.5, 1 / hl);
  let v = 0;
  let first = true;
  for (const d of diffs) {
    const sq = d * d;
    if (first) { v = sq; first = false; }
    else v = (1 - a) * v + a * sq;
  }
  return v;
}

// Forecast one market: prices is the 1-minute grid up to and including the checkpoint.
function forecastMarket(prices) {
  const n = prices.length;
  const x = prices[n - 1];

  // One-sided 1-minute diffs.
  const up = new Array(n - 1);
  const dn = new Array(n - 1);
  for (let i = 1; i < n; i++) {
    const d = prices[i] - prices[i - 1];
    up[i - 1] = d > 0 ? d : 0;
    dn[i - 1] = d < 0 ? d : 0;
  }
  let sigUp = Math.sqrt(ewmaVar(up, HL));
  let sigDn = Math.sqrt(ewmaVar(dn, HL));
  sigUp = Math.min(VOL_CAP, Math.max(VOL_FLOOR, sigUp));
  sigDn = Math.min(VOL_CAP, Math.max(VOL_FLOOR, sigDn));
  const sig = (sigUp + sigDn) / 2;

  // Drift: endpoint slope over the last DRIFT_WIN minutes.
  let drift = 0;
  const w = Math.min(DRIFT_WIN, n - 1);
  if (w > 0) {
    drift = (prices[n - 1] - prices[n - 1 - w]) / w;
    drift *= DRIFT_SHRINK;
    if (drift > DRIFT_CAP) drift = DRIFT_CAP;
    else if (drift < -DRIFT_CAP) drift = -DRIFT_CAP;
    const gate = sig / DRIFT_GATE_REF;
    if (gate < 1) drift *= gate;
  }
  // Drift effect at horizon h: saturating, drift * tau * (1 - exp(-h/tau)).
  const driftAt = (h) => drift * DRIFT_TAU * (1 - Math.exp(-h / DRIFT_TAU));

  const rows = [];
  for (let h = 1; h <= HORIZON; h++) {
    const c = x + driftAt(h);
    const s = Math.sqrt(h);
    const row = [
      c + Z[0] * sigDn * s,
      c + Z[1] * sigDn * s,
      c,
      c + Z[3] * sigUp * s,
      c + Z[4] * sigUp * s,
    ];
    for (let j = 0; j < 5; j++) row[j] = clamp01(row[j]);
    for (let j = 1; j < 5; j++) if (row[j] < row[j - 1]) row[j] = row[j - 1];
    rows.push(row.map(round6));
  }
  return rows;
}

function main() {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  for (const f of files) {
    const id = f.slice(0, -3);
    const c = JSON.parse(readFileSync(INPUT_DIR + f, 'utf8'));
    const forecasts = c.markets.map((m) => ({ id: m.id, q: forecastMarket(m.prices) }));
    writeFileSync(`${OUTPUT_DIR}${id}.out`, JSON.stringify({ forecasts }));
  }
}

main();
