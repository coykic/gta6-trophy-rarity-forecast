// gta6-trophy-rarity-forecast-01 entry: "weekone-1", a forecasting program for the earn rate of every
// Grand Theft Auto VI PS5 trophy on Exophase at a case's target, written as five quantiles per trophy
// and scored by pinball loss on the logit scale.
//
// What a run does: read every <id>.in in the input directory (the grader runs one case at a time),
// where a case is every read the house made of the game's Exophase trophy page before the case's
// checkpoint, and write <id>.out: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.
//
// The model, in one paragraph. Exophase refreshes its counts in steps, so the history is first
// collapsed to the distinct page states (a read identical to the one before it is the same page
// standing, not a new observation), each weighted by how long it stood and how recent it is. On
// the last 48 hours of states, each trophy's trajectory is fitted in ceiling space,
// z = ln(r / (A - r)) against ln(1 + hours since release), over a grid of ceilings A: a trophy
// approaching the share of players who will ever earn it flattens as it nears its ceiling instead
// of extrapolating through it, which is where a plain logit line goes wrong in week one. The
// fitted slope is precision-shrunk toward the trophy's tier pool, clamped, damped toward
// persistence, and anchored at the last observed rate. The spread combines the fit's residual
// scatter (floored at the rates' own quantisation noise), the extrapolation leverage at the
// target, a small refresh-risk term whose volatility is measured from the history's own page
// changes, and a doubt term proportional to the move itself. Finally, subset relations that hold
// in every state of the world (a later story chapter's earners are a subset of the earlier one's;
// the platinum's are a subset of everyone's) cap the dominated trophies' quantiles at the
// dominator's.
//
// The program is deterministic, standard library only, and never throws: any unexpected shape
// falls back to a persistence forecast, because an invalid case ranks below every valid one.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/* ---------------- constants (see MODEL notes in README; tuned on synthetic week-one histories) --- */

/** The trend is fitted on the distinct states of this last window; earlier states are launch chaos. */
const WINDOW_H = 48;
/** Recency decay inside the window: a state tau hours old counts exp(-age / this) as much. */
const WINDOW_DECAY_H = 72;
/** A page state's weight is its standing time in hours, clamped to this range before recency. */
const DUR_MIN = 0.2;
const DUR_MAX = 24;
/** Extrapolated moves are damped toward persistence by this factor. */
const DAMP = 0.9;
/** A fitted move over the horizon is clamped to this many logit units before damping. */
const MOVE_CAP = 1.6;
/** Slope shrinkage: pseudo-precision of the tier pool, in units of one fit's precision. */
const SHRINK_K = 6;
/** Residual-variance shrinkage toward the pooled variance, in pseudo-observations. */
const RESID_K = 6;
/** Share of the measured refresh variance added to the predictive variance. Small on purpose:
    the per-refresh moves already carry the trend the median extrapolates, so most of the
    dispersion must come from the fit's own residuals, not from re-charging the trend as noise. */
const JUMP_SHARE = 0.1;
/** Extra predictive SD charged per unit of extrapolated move, in logit units. */
const MOVE_UNCERTAINTY = 0.3;
/** Floors and caps for the predictive spread, in logit units. */
const SPREAD_FLOOR = 0.32;
const SPREAD_CAP = 1.7;
/** Per-square-root-day volatility floor/cap for the refresh-risk term, in logit units. */
const VOL_FLOOR = 0.15;
const VOL_CAP = 0.9;
/** A history with a single distinct page state identifies no dynamics: this spread floor applies. */
const FROZEN_SPREAD = 1.15;
/** ...and this dilution drift per ln(1 + horizon days) is taken off the median, in logit units. */
const FROZEN_DRIFT = 0.5;

/* ---------------- small math helpers ------------------------------------------------------------ */

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const logisticPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** The standard normal's quantile (Acklam's rational approximation, |error| < 1.2e-9). */
function normQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - plow) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

interface Fit {
  a: number; // intercept
  b: number; // slope
  seB: number; // standard error of the slope (Infinity when unidentified)
  residVar: number; // weighted residual variance
  xbar: number; // weighted mean of x
  sxx: number; // weighted sum of (x - xbar)^2
  neff: number; // effective sample size, (sum w)^2 / sum w^2
  sumw: number;
}

/** Weighted least squares of y on x. Falls back to a flat line at the weighted mean. */
function wls(xs: number[], ys: number[], ws: number[]): Fit {
  let sumw = 0, sx = 0, sy = 0;
  for (let i = 0; i < xs.length; i++) { sumw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; }
  const xbar = sx / sumw, ybar = sy / sumw;
  let sxx = 0, sxy = 0, sw2 = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - xbar;
    sxx += ws[i] * dx * dx; sxy += ws[i] * dx * (ys[i] - ybar); sw2 += ws[i] * ws[i];
  }
  const neff = (sumw * sumw) / Math.max(sw2, 1e-12);
  if (sxx < 1e-9 || xs.length < 2) {
    let rv = 0;
    for (let i = 0; i < xs.length; i++) rv += ws[i] * (ys[i] - ybar) * (ys[i] - ybar);
    return { a: ybar, b: 0, seB: Infinity, residVar: Math.max(rv / sumw, 0.0009), xbar, sxx, neff, sumw };
  }
  const b = sxy / sxx;
  const a = ybar - b * xbar;
  let rv = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = ys[i] - (a + b * xs[i]);
    rv += ws[i] * e * e;
  }
  // Rates are printed to 2 decimals, so a fit's residuals can read as exactly zero on a handful
  // of states; floor the variance at the quantisation noise (about 0.03 logit units) so a lucky
  // line through two points never earns infinite precision downstream.
  rv = Math.max(rv / sumw, 0.0009);
  const dof = Math.max(neff - 2, 0.5);
  const seB = Math.max(Math.sqrt(rv / dof / sxx * neff), 0.02);
  return { a, b, seB, residVar: rv, xbar, sxx, neff, sumw };
}

/** Precision-weighted shrinkage of a fitted slope toward its pool's slope. */
function shrinkSlope(b: number, seB: number, poolB: number): number {
  if (!Number.isFinite(seB)) return poolB;
  const prec = 1 / (seB * seB);
  return (b * prec + poolB * SHRINK_K) / (prec + SHRINK_K);
}

/* ---------------- case model -------------------------------------------------------------------- */

interface TrophyMeta { id: number; tier: string; description: string }
interface Read { atMs: number; players: number; rates: Array<number | null> }
interface CaseData {
  levels: number[];
  horizon_minutes: number;
  atMs: number;
  targetMs: number;
  anchorMs: number;
  trophies: TrophyMeta[];
  history: Read[];
}

interface State {
  tau: number; // hours since the anchor (release midnight UTC)
  players: number;
  rates: Array<number | null>;
  dur: number; // hours this page state stood (filled in once the next state is known)
}

/** Collapse the read history to the distinct page states, oldest first. */
function distinctStates(c: CaseData): State[] {
  const states: State[] = [];
  let prevKey = '';
  for (const r of c.history) {
    const key = `${r.players}|${r.rates.map((v) => (v === null ? 'x' : v)).join(',')}`;
    if (key === prevKey) continue;
    prevKey = key;
    states.push({ tau: Math.max(0, (r.atMs - c.anchorMs) / 3_600_000), players: r.players, rates: r.rates, dur: DUR_MIN });
  }
  for (let k = 0; k < states.length; k++) {
    const nextTau = k + 1 < states.length ? states[k + 1].tau : Math.max(0, (c.atMs - c.anchorMs) / 3_600_000);
    states[k].dur = Math.min(DUR_MAX, Math.max(DUR_MIN, nextTau - states[k].tau));
  }
  return states;
}

/** The pooled per-sqrt-day volatility of logit rates between distinct states (the refresh risk). */
function pooledVol(states: State[], nTrophies: number): number {
  let num = 0, den = 0;
  for (let k = 1; k < states.length; k++) {
    const dDays = (states[k].tau - states[k - 1].tau) / 24;
    if (dDays <= 0) continue;
    for (let i = 0; i < nTrophies; i++) {
      const r0 = states[k - 1].rates[i], r1 = states[k].rates[i];
      if (r0 === null || r1 === null) continue;
      const d = logit(r1) - logit(r0);
      num += d * d; den += dDays;
    }
  }
  if (den === 0) return VOL_FLOOR;
  return Math.min(VOL_CAP, Math.max(VOL_FLOOR, Math.sqrt(num / den)));
}

const clampMove = (x: number): number => Math.min(MOVE_CAP, Math.max(-MOVE_CAP, x));

/** Forecast one case; returns one quantile row per trophy, in the case's trophy order. */
function forecastCase(c: CaseData): Array<{ id: number; q: number[] }> {
  const zs = c.levels.map(normQuantile);
  const states = distinctStates(c);
  const lastRead = c.history[c.history.length - 1];
  const xStar = Math.log(1 + Math.max(0, (c.targetMs - c.anchorMs) / 3_600_000));
  const horizonDays = c.horizon_minutes / 1440;
  const frozen = states.length <= 1;

  // Persistence fallback rows, used for any trophy the model cannot see at all.
  const fallback = (i: number): number[] => {
    const r = lastRead?.rates[i] ?? null;
    const l = logit(r ?? 0.01);
    const s = frozen ? FROZEN_SPREAD : 0.5 * Math.sqrt(horizonDays);
    return zs.map((z) => logisticPct(l + z * s));
  };
  if (states.length === 0 || lastRead === undefined) {
    return c.trophies.map((t, i) => ({ id: t.id, q: enforceMonotone(fallback(i).map(round4)) }));
  }

  const vol = pooledVol(states, c.trophies.length);
  const jumpVar = JUMP_SHARE * vol * vol * horizonDays;

  // The fitting window: the last WINDOW_H of distinct states (at least the last five).
  const tauLast = states[states.length - 1].tau;
  let wStart = states.length - 1;
  while (wStart > 0 && tauLast - states[wStart - 1].tau <= WINDOW_H) wStart--;
  while (states.length - wStart < 5 && wStart > 0) wStart--;
  const win = states.slice(wStart);
  const wws = win.map((s) => s.dur * Math.exp(-(tauLast - s.tau) / WINDOW_DECAY_H));
  const wxs = win.map((s) => Math.log(1 + s.tau));
  const xLast = wxs[wxs.length - 1];
  const dx = Math.max(0, xStar - xLast);

  // Per-trophy window fits, each trophy anchored at its last observed state. The fit runs in
  // ceiling space: z = ln(r / (A - r)) for a ceiling A on a grid, so a trophy near its ceiling
  // flattens instead of extrapolating through it. A plain logit fit is kept beside it for the
  // residual scatter that feeds the spread.
  interface PerTrophy {
    curve: Fit; ceil: Fit; ceilA: number;
    yLast: number; zLast: number; lastRate: number | null; nObs: number;
  }
  const per: PerTrophy[] = c.trophies.map((t, i) => {
    const ox: number[] = [], oy: number[] = [], ow: number[] = [];
    let yLast = NaN, maxRate = 0;
    win.forEach((s, k) => {
      const r = s.rates[i];
      if (r === null) return;
      ox.push(wxs[k]); oy.push(logit(r)); ow.push(wws[k]);
      yLast = logit(r);
    });
    for (const s of states) { const r = s.rates[i]; if (r !== null && r > maxRate) maxRate = r; }
    // Ceiling grid: fractions of 100% above the highest rate this trophy has ever shown.
    const grid = [100, 99, 97, 94, 90, 84, 76, 66, 55, 45, 36, 28, 21, 15, 10, 7, 4.5, 3, 2, 1.2]
      .filter((A) => A > maxRate * 1.001);
    if (grid.length === 0) grid.push(Math.min(100, maxRate * 1.02 + 0.01));
    let ceil: Fit | null = null, ceilA = 100, ceilSSE = Infinity, plainSSE = Infinity;
    for (const A of grid) {
      const oz = oy.map((_, k) => {
        const r = Math.min(logisticPct(oy[k]), A * 0.9995);
        return Math.log(r / (A - r));
      });
      const f = wls(ox, oz, ow);
      let sse = 0;
      for (let k = 0; k < ox.length; k++) { const e = oz[k] - (f.a + f.b * ox[k]); sse += ow[k] * e * e; }
      if (A === grid[0]) plainSSE = sse;
      if (sse < ceilSSE) { ceilSSE = sse; ceil = f; ceilA = A; }
    }
    // Accept a lower ceiling only when it clearly fits better than the plain logit line.
    if (ceil === null || ceilSSE > 0.85 * plainSSE) {
      const A = grid[0];
      const oz = oy.map((_, k) => {
        const r = Math.min(logisticPct(oy[k]), A * 0.9995);
        return Math.log(r / (A - r));
      });
      ceil = wls(ox, oz, ow); ceilA = A;
    }
    const rLast = lastRead.rates[i];
    const zLast = rLast === null ? NaN : Math.log(Math.min(rLast, ceilA * 0.9995) / (ceilA - Math.min(rLast, ceilA * 0.9995)));
    return { curve: wls(ox, oy, ow), ceil, ceilA, yLast, zLast, lastRate: rLast, nObs: ox.length };
  });

  // Tier pools of window slopes and residual variances, plus a global pool.
  const poolOf = (key: (t: TrophyMeta) => string) => {
    const pools = new Map<string, { bCeil: number; rv: number }>();
    const groups = new Map<string, number[]>();
    c.trophies.forEach((t, i) => {
      const g = key(t);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(i);
    });
    for (const [g, idxs] of groups) {
      let cw = 0, cSum = 0, rvSum = 0, rvW = 0;
      for (const i of idxs) {
        const f = per[i].curve, fc = per[i].ceil;
        if (Number.isFinite(fc.seB) && fc.seB > 0) { const p = 1 / (fc.seB * fc.seB); cw += p; cSum += p * fc.b; }
        rvSum += f.residVar * Math.max(per[i].nObs, 1); rvW += Math.max(per[i].nObs, 1);
      }
      pools.set(g, {
        bCeil: cw > 0 ? cSum / cw : 0,
        rv: rvW > 0 ? rvSum / rvW : 0.04,
      });
    }
    return pools;
  };
  const tierPools = poolOf((t) => t.tier);
  const globalPool = poolOf(() => 'all').get('all')!;

  const rows: Array<{ id: number; q: number[] }> = c.trophies.map((t, i) => {
    const p = per[i];
    if (p.nObs === 0 || p.lastRate === null || !Number.isFinite(p.yLast) || !Number.isFinite(p.zLast)) return { id: t.id, q: fallback(i) };
    const pool = tierPools.get(t.tier) ?? globalPool;

    // The forecast: the shrunk slope in ceiling space, damped, anchored at the last rate and
    // mapped back through the ceiling, so a flattening trophy stays flat.
    const bCeil = shrinkSlope(p.ceil.b, p.ceil.seB, pool.bCeil);
    const zHat = p.zLast + clampMove(bCeil * dx) * DAMP;
    let yHat = logit(p.ceilA / (1 + Math.exp(-zHat)));
    if (frozen) yHat = logit(p.lastRate) - FROZEN_DRIFT * Math.log(1 + horizonDays);
    if (!Number.isFinite(yHat)) yHat = p.yLast;

    // Predictive spread: residual scatter with extrapolation leverage, refresh risk, move doubt.
    const rv = (p.curve.neff * p.curve.residVar + RESID_K * pool.rv) / (p.curve.neff + RESID_K);
    const leverage = p.curve.sxx > 1e-9
      ? 1 + 1 / Math.max(p.curve.neff, 1) + ((xStar - p.curve.xbar) ** 2) / p.curve.sxx
      : 3;
    const move = yHat - p.yLast;
    let spread = Math.sqrt(Math.max(0, rv) * leverage + jumpVar + (MOVE_UNCERTAINTY * move) ** 2);
    spread = Math.min(SPREAD_CAP, Math.max(SPREAD_FLOOR, spread));
    if (frozen) spread = Math.max(spread, FROZEN_SPREAD);

    const q = zs.map((z) => logisticPct(yHat + z * spread));
    return { id: t.id, q };
  });

  applyDominanceCaps(c, rows);
  // Last line of defence: a row with a non-finite quantile is an invalid case, so any row that
  // still carries one is replaced by its persistence fallback.
  return rows.map((r, i) => ({
    id: r.id,
    q: enforceMonotone((r.q.every(Number.isFinite) ? r.q : fallback(i)).map(round4)),
  }));
}

/** Roundoff guard: quantiles must be non-decreasing. */
function enforceMonotone(q: number[]): number[] {
  const out = q.slice();
  for (let j = 1; j < out.length; j++) out[j] = Math.max(out[j], out[j - 1]);
  return out;
}

/**
 * Subset caps that hold in every state of the world: a later story chapter's earners are a subset
 * of the earlier chapter's, and the platinum's earners are a subset of every other trophy's. Where
 * X's earners are a subset of Y's, X's rate never exceeds Y's, so X's quantile at each level is
 * capped at Y's quantile at that level.
 */
function applyDominanceCaps(c: CaseData, rows: Array<{ id: number; q: number[] }>): void {
  const byId = new Map(rows.map((r) => [r.id, r]));
  const cap = (sub: { q: number[] }, sup: { q: number[] }) => {
    for (let j = 0; j < sub.q.length; j++) sub.q[j] = Math.min(sub.q[j], sup.q[j]);
  };
  // The story chain, in order: the opening mission, chapters 1..4, the final mission.
  const chain: Array<{ id: number; ord: number }> = [];
  for (const t of c.trophies) {
    const ch = /complete chapter\s+(\d+)/i.exec(t.description);
    if (ch) chain.push({ id: t.id, ord: Number(ch[1]) });
    else if (/complete the final mission\./i.test(t.description)) chain.push({ id: t.id, ord: 99 });
  }
  chain.sort((x, y) => x.ord - y.ord);
  for (let k = 1; k < chain.length; k++) {
    const sub = byId.get(chain[k].id), sup = byId.get(chain[k - 1].id);
    if (sub && sup) cap(sub, sup);
  }
  // The platinum is capped by every other trophy.
  const plat = c.trophies.find((t) => t.tier === 'platinum');
  if (plat) {
    const prow = byId.get(plat.id);
    if (prow) for (const r of rows) if (r.id !== plat.id) cap(prow, r);
  }
}

/* ---------------- driver ------------------------------------------------------------------------ */

function parseCase(raw: string): CaseData {
  const c = JSON.parse(raw) as {
    levels: number[]; horizon_minutes: number; at: string; target: string;
    game: { release_date: string };
    trophies: Array<{ id: number; tier: string; description: string }>;
    history: Array<{ at: string; players_tracked: number; rates: Array<number | null> }>;
  };
  const anchorMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  return {
    levels: c.levels,
    horizon_minutes: c.horizon_minutes,
    atMs: Date.parse(c.at),
    targetMs: Date.parse(c.target),
    anchorMs: Number.isFinite(anchorMs) ? anchorMs : Date.parse(c.history[0]?.at ?? c.at),
    trophies: c.trophies.map((t) => ({ id: t.id, tier: t.tier, description: t.description ?? '' })),
    history: c.history.map((h) => ({ atMs: Date.parse(h.at), players: h.players_tracked, rates: h.rates })),
  };
}

/** Persistence rows for a case that cannot be parsed at all, keyed off nothing but its text. */
function emergencyRows(raw: string): Array<{ id: number; q: number[] }> | null {
  const ids = [...raw.matchAll(/"id"\s*:\s*(\d+)/g)].map((m) => Number(m[1]));
  if (ids.length === 0) return null;
  const unique = [...new Set(ids)].slice(0, 200);
  return unique.map((id) => ({ id, q: [0.01, 0.01, 0.01, 0.01, 0.01] }));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const raw = readFileSync(join(INPUT_DIR, file), 'utf8');
  let rows: Array<{ id: number; q: number[] }> | null = null;
  try {
    rows = forecastCase(parseCase(raw));
  } catch {
    rows = emergencyRows(raw);
  }
  if (rows !== null && rows.length > 0) {
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
  }
}
