# Week One — GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/coykic/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/coykic/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecasting program for make.fun's "Forecast how many GTA VI players earn
each trophy in week one" bounty. It reads one case file — every read the house
made of GTA VI's Exophase trophy page before the case's checkpoint — and
writes, for each of the game's 55 PlayStation 5 trophies, five quantiles of
its earn rate at the case's target, scored by pinball loss on the logit scale.

## The model (weekone-1)

One function forecasts every trophy; no trophy is special-cased by name.

- **Distinct states.** Exophase refreshes its counts in steps, so consecutive
  identical reads are one observation, not many. The history is collapsed to
  the moments the page actually changed, each weighted by how long it stood
  and how recent it is.
- **Ceiling space.** On the last 48 hours of states, each trophy's trajectory
  is fitted as z = ln(r / (A − r)) against ln(1 + hours since release), over a
  grid of ceilings A, accepting a ceiling below 100% only when it clearly fits
  better than the plain logit line. A trophy approaching the share of players
  who will ever earn it flattens as it nears its ceiling instead of
  extrapolating through it — where a plain logit line goes wrong in week one.
  (An earlier two-view design that also extrapolated earner counts against
  the tracked pool was dropped: the pool's growth saturates, its log-log
  extrapolation overshoots, and the blend scored worse on every synthetic
  world tested.)
- **Anchored, damped, shrunk.** The forecast starts from the last observed
  rate, never from a fitted intercept. The fitted slope is precision-shrunk
  toward the trophy's tier pool (platinum / gold / silver / bronze), clamped,
  and damped toward persistence, so a quiet trophy borrows its peers'
  behaviour and no trophy extrapolates its launch-week slope at full strength
  for four and a half days.
- **Honest spread.** The predictive spread combines the fit's residual
  scatter (floored at the rates' own quantisation noise — the page prints 2
  decimals, so a perfect-looking fit through a few states is luck, not
  knowledge), the extrapolation leverage at the target, a small refresh-risk
  term whose volatility is measured from the history's own page changes, and
  a doubt term proportional to the move itself.
- **Dominance caps.** Earners of a later story chapter are a subset of the
  earlier chapter's, and the platinum's earners are a subset of every other
  trophy's, so those quantiles are capped at the dominating trophy's, level
  by level — true in every state of the world, so it can only help.
- **Frozen pages.** If a case's history is a single unchanging page, no
  dynamics are identifiable: the forecast stays near persistence but with a
  wide, refresh-aware spread and a mild dilution drift.

The program is deterministic (two runs byte-identical), standard library
only, and never throws: any unexpected case shape falls back to a persistence
forecast rather than an invalid case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` at the repository root reads every `*.in` case file in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<id>.out` forecast per
case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=hw6n5dxn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
