"use strict";
/* Chart terminal: live candles on TradingView's open-source Lightweight Charts, computed indicators,
   drawing tools and a persistent watchlist. Read-only: it only reads prices and never places orders. */

const TF_SECONDS = { "1Min": 60, "5Min": 300, "15Min": 900 };
const DEFAULT_WATCH = ["BTC/USD", "ETH/USD", "GLD", "SPY", "QQQ", "AAPL", "NVDA"];
const LIVE_SYMBOL_RE = /^[A-Z][A-Z0-9.\-]{0,9}(\/USD)?$/;
const MAX_WATCH = 20;
const INDICATORS = { sma20: "SMA 20", ema50: "EMA 50", boll: "Bollinger", rsi: "RSI 14", macd: "MACD" };
const WATCH_REFRESH_MS = 5000;

let liveSym = store.get("desk.live", "BTC/USD");
let liveTf = store.get("desk.liveTf", "1Min");
let watchlist = store.get("desk.watch", DEFAULT_WATCH).filter((s) => LIVE_SYMBOL_RE.test(s)).slice(0, MAX_WATCH);
let activeInd = new Set(store.get("desk.ind", ["sma20"]).filter((k) => k in INDICATORS));
let chart = null, candles = null, volume = null, markers = null;
let bars = [];                       // [{time, open, high, low, close, volume}], oldest first
let indSeries = [];                  // [{key, series, pick}]: pick(computed) -> values aligned with bars
let drawn = { lines: [], trends: [] };
let draw = { mode: null, pending: null };

const nyTime = (opts) => (sec) => new Date(sec * 1000).toLocaleString("en-US", { timeZone: NY, ...opts });
const tickLabel = (time, type) => (type >= 3
  ? nyTime({ hour: "numeric", minute: "2-digit", hour12: false })(time)
  : nyTime({ month: "short", day: "numeric" })(time));
const size = () => TF_SECONDS[liveTf];
const drawKey = () => "desk.draw." + liveSym;

/* ---------- chart ---------- */
function ensureChart() {
  if (chart) return;
  const box = $("live");
  box.replaceChildren();
  chart = LightweightCharts.createChart(box, {
    autoSize: true,
    layout: { background: { color: css("--paper") }, textColor: css("--muted"), fontFamily: css("--num"), panes: { separatorColor: css("--ink") } },
    grid: { vertLines: { color: css("--rule") }, horzLines: { color: css("--rule") } },
    rightPriceScale: { borderColor: css("--ink") },
    timeScale: { borderColor: css("--ink"), timeVisible: true, secondsVisible: false, tickMarkFormatter: tickLabel },
    localization: { timeFormatter: nyTime({ month: "short", day: "numeric", hour: "numeric", minute: "2-digit", hour12: false }) },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
    upColor: css("--buy"), downColor: css("--sell"), borderVisible: false,
    wickUpColor: css("--buy"), wickDownColor: css("--sell"),
  });
  volume = chart.addSeries(LightweightCharts.HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "vol", color: css("--rule") });
  chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
  chart.subscribeClick(onChartClick);
}
function showError(message) {
  if (chart) { chart.remove(); chart = candles = volume = markers = null; indSeries = []; drawn = { lines: [], trends: [] }; }
  $("live").replaceChildren(h("div", { class: "empty", text: message }));
}
function sizePanes() {
  try { chart.panes().forEach((pane, i) => pane.setStretchFactor(i === 0 ? 3 : 1)); } catch { /* panes are cosmetic */ }
}

async function loadCandles(keepView = false) {
  try {
    const d = await api(`/api/bars?symbol=${encodeURIComponent(liveSym)}&tf=${liveTf}`);
    if (!d.bars.length) { showError("No price history for " + liveSym + " yet."); return; }
    ensureChart();
    bars = d.bars.map((b) => ({ ...b }));
    candles.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    volume.setData(bars.map((b) => ({ time: b.time, value: b.volume, color: b.close >= b.open ? css("--buy-tint") : css("--sell-tint") })));
    paintIndicators();
    restoreDrawings();
    if (!keepView) chart.timeScale().scrollToRealTime();
    await loadFills(bars[0].time);
  } catch (e) { showError("Price history: " + e.message); }
}
async function loadFills(firstTime) {
  try {
    const d = await api("/api/fills?symbol=" + encodeURIComponent(liveSym));
    const pts = d.fills.filter((f) => f.time >= firstTime).map((f) => ({
      time: Math.floor(f.time / size()) * size(), position: f.side === "buy" ? "belowBar" : "aboveBar",
      shape: f.side === "buy" ? "arrowUp" : "arrowDown", color: f.side === "buy" ? css("--buy") : css("--sell"),
      text: `${f.side === "buy" ? "Buy" : "Sell"} ${f.price.toLocaleString("en-US", { maximumFractionDigits: 2 })}`,
    })).sort((a, b) => a.time - b.time);
    if (markers) markers.setMarkers(pts); else markers = LightweightCharts.createSeriesMarkers(candles, pts);
  } catch { /* markers are optional */ }
}

/* ---------- indicators ---------- */
const withTimes = (values) => values.map((v, i) => (v === null ? null : { time: bars[i].time, value: v })).filter(Boolean);
function indicatorDefs() {
  const L = LightweightCharts.LineSeries, H = LightweightCharts.HistogramSeries;
  const thin = { lineWidth: 1, lastValueVisible: false, priceLineVisible: false };
  return {
    sma20: [{ type: L, opts: { ...thin, color: css("--wait"), lineWidth: 2 }, pane: 0, pick: (c) => Indicators.sma(c, 20) }],
    ema50: [{ type: L, opts: { ...thin, color: css("--stamp"), lineWidth: 2 }, pane: 0, pick: (c) => Indicators.ema(c, 50) }],
    boll: ["upper", "mid", "lower"].map((k) => ({ type: L, opts: { ...thin, color: css("--muted"), lineStyle: k === "mid" ? 2 : 0 }, pane: 0, pick: (c) => Indicators.bollinger(c, 20, 2)[k] })),
    rsi: [{ type: L, opts: { ...thin, color: css("--stamp"), lineWidth: 2, lastValueVisible: true }, pane: 1, pick: (c) => Indicators.rsi(c, 14), levels: [30, 70] }],
    macd: [
      { type: H, opts: { priceLineVisible: false, lastValueVisible: false }, pane: 2, pick: (c) => Indicators.macd(c).hist, histogram: true },
      { type: L, opts: { ...thin, color: css("--stamp"), lineWidth: 2 }, pane: 2, pick: (c) => Indicators.macd(c).macd },
      { type: L, opts: { ...thin, color: css("--wait"), lineWidth: 2 }, pane: 2, pick: (c) => Indicators.macd(c).signal },
    ],
  };
}
function paintIndicators() {
  if (!chart) return;
  indSeries.forEach(({ series }) => chart.removeSeries(series));
  indSeries = [];
  const defs = indicatorDefs();
  const closes = bars.map((b) => b.close);
  for (const key of Object.keys(INDICATORS).filter((k) => activeInd.has(k))) {
    for (const def of defs[key]) {
      const series = chart.addSeries(def.type, def.opts, def.pane);
      (def.levels || []).forEach((price) => series.createPriceLine({ price, color: css("--rule"), lineWidth: 1, lineStyle: 2, axisLabelVisible: false }));
      const values = def.pick(closes);
      series.setData(def.histogram
        ? withTimes(values).map((p) => ({ ...p, color: p.value >= 0 ? css("--buy-tint") : css("--sell-tint") }))
        : withTimes(values));
      indSeries.push({ key, series, def });
    }
  }
  sizePanes();
}
function updateIndicatorsLast() {
  if (!indSeries.length || !bars.length) return;
  const closes = bars.map((b) => b.close), i = bars.length - 1;
  for (const { series, def } of indSeries) {
    const v = def.pick(closes)[i];
    if (v === null) continue;
    series.update(def.histogram ? { time: bars[i].time, value: v, color: v >= 0 ? css("--buy-tint") : css("--sell-tint") } : { time: bars[i].time, value: v });
  }
}
function renderIndicatorTools() {
  $("indicator-tools").replaceChildren(...Object.entries(INDICATORS).map(([key, label]) => {
    const btn = h("button", { type: "button", "aria-pressed": String(activeInd.has(key)), text: label });
    btn.addEventListener("click", () => {
      activeInd = activeInd.has(key) ? new Set([...activeInd].filter((k) => k !== key)) : new Set([...activeInd, key]);
      store.set("desk.ind", [...activeInd]);
      btn.setAttribute("aria-pressed", String(activeInd.has(key)));
      paintIndicators();
    });
    return btn;
  }));
}

/* ---------- drawings ---------- */
function loadDrawings() { return store.get(drawKey(), { levels: [], trends: [] }); }
function saveDrawings(d) { store.set(drawKey(), d); }
function paintLevel(price) {
  drawn.lines.push(candles.createPriceLine({ price, color: css("--stamp"), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" }));
}
function paintTrend([t1, p1, t2, p2]) {
  const s = chart.addSeries(LightweightCharts.LineSeries, { color: css("--stamp"), lineWidth: 2, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
  s.setData([{ time: t1, value: p1 }, { time: t2, value: p2 }]);
  drawn.trends.push(s);
}
function clearPainted() {
  drawn.lines.forEach((l) => candles.removePriceLine(l));
  drawn.trends.forEach((s) => chart.removeSeries(s));
  drawn = { lines: [], trends: [] };
}
function restoreDrawings() {
  clearPainted();
  const d = loadDrawings();
  d.levels.forEach(paintLevel);
  d.trends.forEach(paintTrend);
}
function setMode(mode) {
  draw = { mode: draw.mode === mode ? null : mode, pending: null };
  document.querySelectorAll("[data-draw]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.draw === draw.mode)));
  $("draw-hint").textContent = draw.mode === "level" ? "Click the chart to place a price level. Esc to stop."
    : draw.mode === "trend" ? "Click two points on the chart. Esc to stop." : "";
}
function onChartClick(param) {
  if (!draw.mode || !param.point || param.time == null) return;
  const price = candles.coordinateToPrice(param.point.y);
  if (price == null) return;
  const d = loadDrawings();
  if (draw.mode === "level") {
    saveDrawings({ ...d, levels: [...d.levels, price] });
    paintLevel(price);
  } else if (!draw.pending) {
    draw = { ...draw, pending: [param.time, price] };
    $("draw-hint").textContent = "Now click the second point.";
  } else {
    const [t1, p1] = draw.pending;
    if (param.time === t1) return;            // two points on one candle would stack and break the series
    const line = t1 < param.time ? [t1, p1, param.time, price] : [param.time, price, t1, p1];
    saveDrawings({ ...d, trends: [...d.trends, line] });
    paintTrend(line);
    draw = { ...draw, pending: null };
    $("draw-hint").textContent = "Trend line added. Click two more points, or press Esc.";
  }
}

/* ---------- live updates ---------- */
async function pollLive() {
  if (document.hidden || !candles || !bars.length) return;
  try {
    const d = await api("/api/tick?symbol=" + encodeURIComponent(liveSym));
    const px = midPrice(d);
    if (px == null) return;
    const last = bars[bars.length - 1];
    const bucket = Math.floor(Date.now() / 1000 / size()) * size();
    const next = bucket > last.time
      ? { time: bucket, open: last.close, high: Math.max(last.close, px), low: Math.min(last.close, px), close: px, volume: 0 }
      : { ...last, high: Math.max(last.high, px), low: Math.min(last.low, px), close: px };
    bars = bucket > last.time ? [...bars, next] : [...bars.slice(0, -1), next];
    candles.update({ time: next.time, open: next.open, high: next.high, low: next.low, close: next.close });
    updateIndicatorsLast();
  } catch { /* the next tick will try again */ }
}
function midPrice(d) {
  const bid = num(d.bid), ask = num(d.ask);
  const sane = bid > 0 && ask > 0 && ask >= bid && (ask - bid) / ask < 0.01;   // closed markets report junk quotes
  return sane ? (bid + ask) / 2 : num(d.last);
}

/* ---------- watchlist ---------- */
const watchPrices = new Map();
function renderWatch() {
  $("watch-list").replaceChildren(...watchlist.map((sym) => {
    const p = watchPrices.get(sym);
    const open = h("button", { type: "button", class: "wl-open", "aria-current": String(sym === liveSym) },
      h("b", { text: sym }), h("span", { class: "num", text: p ? p.px.toLocaleString("en-US", { maximumFractionDigits: 4 }) : "..." }),
      h("small", { text: p ? (p.spread === "no live quote" ? p.spread : `spread ${p.spread}`) : "" }));
    open.addEventListener("click", () => { $("live-sym").value = sym; watch(sym, liveTf); renderWatch(); });
    const rm = h("button", { type: "button", class: "wl-rm", "aria-label": "Remove " + sym, text: "x" });
    rm.addEventListener("click", () => { watchlist = watchlist.filter((s) => s !== sym); store.set("desk.watch", watchlist); renderWatch(); });
    return h("li", {}, open, rm);
  }));
}
async function refreshWatch() {
  if (document.hidden) return;
  await Promise.allSettled(watchlist.map(async (sym) => {
    const d = await api("/api/tick?symbol=" + encodeURIComponent(sym));
    const px = midPrice(d), bid = num(d.bid), ask = num(d.ask);
    const gap = bid > 0 && ask >= bid ? (ask - bid) / ask : null;
    const spread = gap !== null && gap < 0.01 ? (gap * 100).toFixed(3) + "%" : "no live quote";   // closed markets quote junk
    if (px != null) watchPrices.set(sym, { px, spread });
  }));
  renderWatch();
}

/* ---------- wiring ---------- */
function watch(sym, tf) {
  liveSym = sym; liveTf = tf; store.set("desk.live", sym); store.set("desk.liveTf", tf);
  markers = null; bars = []; setMode(null); loadCandles();
}
$("live-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const v = $("live-sym").value.trim().toUpperCase();
  if (!LIVE_SYMBOL_RE.test(v)) { $("live-sym").setCustomValidity("Use a ticker like AAPL or a pair like BTC/USD"); $("live-sym").reportValidity(); return; }
  $("live-sym").setCustomValidity("");
  watch(v, $("live-tf").value);
  renderWatch();
});
$("live-tf").addEventListener("change", () => watch(liveSym, $("live-tf").value));
$("live-sym").addEventListener("input", () => $("live-sym").setCustomValidity(""));
$("watch-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const v = $("watch-sym").value.trim().toUpperCase();
  if (!LIVE_SYMBOL_RE.test(v)) { $("watch-sym").setCustomValidity("Use a ticker like AAPL or a pair like BTC/USD"); $("watch-sym").reportValidity(); return; }
  $("watch-sym").setCustomValidity(""); $("watch-sym").value = "";
  if (!watchlist.includes(v) && watchlist.length < MAX_WATCH) {
    watchlist = [...watchlist, v]; store.set("desk.watch", watchlist); renderWatch(); refreshWatch();
  }
});
$("watch-sym").addEventListener("input", () => $("watch-sym").setCustomValidity(""));
document.querySelectorAll("[data-draw]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.draw)));
$("clear-draw").addEventListener("click", () => { saveDrawings({ levels: [], trends: [] }); if (chart) clearPainted(); setMode(null); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setMode(null); });
$("live-sym").value = liveSym;
$("live-tf").value = liveTf;
renderIndicatorTools();
renderWatch();
refreshWatch();
setInterval(pollLive, 1000);
setInterval(refreshWatch, WATCH_REFRESH_MS);
setInterval(() => { if (!document.hidden) loadCandles(true); }, 60000);   // re-sync with the exchange candles
