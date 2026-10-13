"use strict";
/* Chart indicators, computed in the browser from candle closes. Pine-style definitions:
   EMA and Wilder RSI are seeded with the simple average of the first n values; Bollinger uses
   the population standard deviation. Every function returns a new array (null while warming up). */
const Indicators = (() => {
  const blank = (n) => new Array(n).fill(null);

  function sma(values, n) {
    const out = blank(values.length);
    let sum = 0;
    values.forEach((v, i) => {
      sum += v;
      if (i >= n) sum -= values[i - n];
      if (i >= n - 1) out[i] = sum / n;
    });
    return out;
  }

  function ema(values, n) {
    const out = blank(values.length);
    if (values.length < n) return out;
    const alpha = 2 / (n + 1);
    out[n - 1] = values.slice(0, n).reduce((a, b) => a + b, 0) / n;
    for (let i = n; i < values.length; i++) out[i] = alpha * values[i] + (1 - alpha) * out[i - 1];
    return out;
  }

  function wilder(values, n) {  // RMA: alpha = 1/n, seeded with the SMA
    const out = blank(values.length);
    if (values.length < n) return out;
    out[n - 1] = values.slice(0, n).reduce((a, b) => a + b, 0) / n;
    for (let i = n; i < values.length; i++) out[i] = (out[i - 1] * (n - 1) + values[i]) / n;
    return out;
  }

  function rsi(values, n = 14) {
    const out = blank(values.length);
    if (values.length <= n) return out;
    const gain = [], loss = [];
    for (let i = 1; i < values.length; i++) {
      const d = values[i] - values[i - 1];
      gain.push(d > 0 ? d : 0); loss.push(d < 0 ? -d : 0);
    }
    const up = wilder(gain, n), down = wilder(loss, n);
    up.forEach((u, k) => {
      if (u === null) return;
      out[k + 1] = down[k] === 0 ? 100 : 100 - 100 / (1 + u / down[k]);
    });
    return out;
  }

  function macd(values, fast = 12, slow = 26, signalN = 9) {
    const f = ema(values, fast), s = ema(values, slow);
    const line = f.map((v, i) => (v === null || s[i] === null ? null : v - s[i]));
    const first = line.findIndex((v) => v !== null);
    const signal = blank(values.length);
    if (first >= 0) ema(line.slice(first), signalN).forEach((v, k) => { signal[first + k] = v; });
    const hist = line.map((v, i) => (v === null || signal[i] === null ? null : v - signal[i]));
    return { macd: line, signal, hist };
  }

  function bollinger(values, n = 20, k = 2) {
    const mid = sma(values, n);
    const sd = mid.map((m, i) => {
      if (m === null) return null;
      const w = values.slice(i - n + 1, i + 1);
      return Math.sqrt(w.reduce((a, v) => a + (v - m) ** 2, 0) / n);
    });
    return {
      mid,
      upper: mid.map((m, i) => (m === null ? null : m + k * sd[i])),
      lower: mid.map((m, i) => (m === null ? null : m - k * sd[i])),
    };
  }

  return { sma, ema, rsi, macd, bollinger };
})();
if (typeof module !== "undefined") module.exports = Indicators;
