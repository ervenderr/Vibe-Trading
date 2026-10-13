// Run: node --test src/desk/static/indicators.test.mjs
// Expected numbers come from an independent numpy implementation of the documented definitions
// (Pine-style: EMA seeded with the SMA, Wilder RSI seeded with the SMA, population-stdev Bollinger).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const { sma, ema, rsi, macd, bollinger } = createRequire(import.meta.url)("./indicators.js");

const close = Array.from({ length: 60 }, (_, i) => Math.round((100 + 5 * Math.sin(i / 3) + i * 0.1) * 1e4) / 1e4);
const near = (got, want, eps = 1e-7) => assert.ok(Math.abs(got - want) < eps, `${got} vs ${want}`);

test("sma", () => {
  const s = sma(close, 20);
  assert.equal(s[18], null);
  near(s[19], 100.9572); near(s[30], 101.89986); near(s[59], 105.154405);
});
test("ema is seeded with the SMA and starts at index n-1", () => {
  const e = ema(close, 12);
  assert.equal(e[10], null);
  near(e[19], 99.934273321); near(e[30], 103.31566719); near(e[59], 105.065164612);
});
test("rsi uses Wilder smoothing", () => {
  const r = rsi(close, 14);
  assert.equal(r[13], null);
  near(r[19], 58.953023416); near(r[30], 42.936137209); near(r[59], 67.775811099);
});
test("rsi is 100 on a series that only rises", () => {
  assert.equal(rsi(Array.from({ length: 30 }, (_, i) => i), 14)[29], 100);
});
test("macd line and signal", () => {
  const m = macd(close, 12, 26, 9);
  assert.equal(m.macd[19], null); assert.equal(m.signal[30], null);
  near(m.macd[30], 0.885484004); near(m.macd[59], 0.811533342);
  near(m.signal[45], 1.23629519); near(m.signal[59], 0.259833519);
  near(m.hist[59], 0.811533342 - 0.259833519);
});
test("bollinger uses the population standard deviation", () => {
  const b = bollinger(close, 20, 2);
  near(b.mid[59], 105.154405); near(b.upper[59], 105.154405 + 2 * 3.292706051); near(b.lower[19], 100.9572 - 2 * 3.054754576);
});
test("inputs are not mutated and short series give nulls", () => {
  const copy = [...close]; sma(close, 5); ema(close, 5); rsi(close, 5);
  assert.deepEqual(close, copy);
  assert.deepEqual(sma([1, 2], 5), [null, null]);
});
