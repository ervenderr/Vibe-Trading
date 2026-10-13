"use strict";
/* Strategy lab: runs /api/backtest and shows the result. Read-only; nothing here places an order. */

const STRATS = {
  sma_cross: { label: "Moving-average cross", params: [["fast", 20, "Fast average"], ["slow", 100, "Slow average"]] },
  rsi_reversion: { label: "RSI dip buyer", params: [["period", 14, "RSI length"], ["low", 30, "Buy below"], ["high", 70, "Sell above"]] },
  breakout: { label: "Channel breakout", params: [["window", 20, "Window"]] },
};
const pctS = (n, digits = 1) => (n == null ? "n/a" : (n * 100).toFixed(digits) + "%");
let labChart = null;

function renderLabParams() {
  const spec = STRATS[$("lab-strat").value];
  $("lab-params").replaceChildren(...spec.params.map(([name, def, label]) =>
    h("label", {}, label + " ", h("input", { type: "number", min: "1", step: "1", value: String(def), "data-param": name }))));
}
function labQuery() {
  const q = new URLSearchParams({
    symbol: $("lab-sym").value.trim().toUpperCase(), strategy: $("lab-strat").value, interval: $("lab-interval").value,
    years: $("lab-years").value, fee_bps: $("lab-fee").value, slippage_bps: $("lab-slip").value,
  });
  document.querySelectorAll("[data-param]").forEach((i) => q.set(i.dataset.param, i.value));
  return q.toString();
}

function verdictText(d) {
  const m = d.metrics, b = d.benchmark, years = d.years;
  const beat = m.total_return > b.total_return;
  const calmer = m.max_drawdown < b.max_drawdown;
  const first = d.halves.first.sharpe, second = d.halves.second.sharpe;
  const parts = [`Over about ${years} years the rule returned ${pctS(m.total_return, 0)} after costs, against ${pctS(b.total_return, 0)} for simply holding.`];
  parts.push(beat ? "It beat holding." : "It did not beat holding.");
  parts.push(calmer ? `It had a smaller worst drop (${pctS(m.max_drawdown)} against ${pctS(b.max_drawdown)}).` : `Its worst drop was no smaller than holding (${pctS(m.max_drawdown)} against ${pctS(b.max_drawdown)}).`);
  if (m.trades < 30) parts.push(`Only ${m.trades} trades, so the result says little.`);
  if (first > 0 && second < first * 0.5) parts.push("The second half was much weaker than the first, a sign of curve fitting.");
  return parts.join(" ");
}

function metricRow(label, cells) { return h("tr", {}, h("th", { scope: "row", text: label }), ...cells.map((c) => h("td", { class: "num", text: c }))); }
function metricsTable(d) {
  const m = d.metrics, b = d.benchmark, f = d.halves.first, s = d.halves.second;
  const col = (x) => [pctS(x.total_return), pctS(x.cagr), x.sharpe.toFixed(2), pctS(x.max_drawdown)];
  const rows = [
    ["Total return", 0], ["Per year", 1], ["Sharpe ratio", 2], ["Worst drop", 3],
  ].map(([label, i]) => metricRow(label, [col(m)[i], col(b)[i], col(f)[i], col(s)[i]]));
  rows.push(metricRow("Trades", [String(m.trades), "1", "", ""]),
    metricRow("Win rate", [pctS(m.win_rate, 0), "", "", ""]),
    metricRow("Time invested", [pctS(m.exposure, 0), "100%", "", ""]));
  return h("table", { class: "lab-table" }, h("thead", {}, h("tr", {}, h("th", { text: "" }), ...["Rule", "Buy and hold", "First half", "Second half"].map((t) => h("th", { scope: "col", text: t })))), h("tbody", {}, ...rows));
}

function drawLabChart(d) {
  const box = h("div", { id: "lab-chart", role: "img", "aria-label": "Rule equity against buy and hold, with drawdown below" });
  if (labChart) { labChart.remove(); labChart = null; }
  $("lab-out").replaceChildren(h("p", { class: "verdict-line", text: verdictText(d) }), metricsTable(d), box, tradesTable(d));
  labChart = LightweightCharts.createChart(box, {
    autoSize: true, height: 360,
    layout: { background: { color: css("--paper") }, textColor: css("--muted"), fontFamily: css("--num"), panes: { separatorColor: css("--ink") } },
    grid: { vertLines: { color: css("--rule") }, horzLines: { color: css("--rule") } },
    rightPriceScale: { borderColor: css("--ink") }, timeScale: { borderColor: css("--ink") },
  });
  const line = (key, color, title) => {
    const s = labChart.addSeries(LightweightCharts.LineSeries, { color, lineWidth: 2, title, priceLineVisible: false });
    s.setData(d.equity.map((p) => ({ time: p.time, value: (p[key] - 1) * 100 })));
  };
  line("buy_hold", css("--muted"), "Hold %");
  line("strategy", css("--stamp"), "Rule %");
  const dd = labChart.addSeries(LightweightCharts.AreaSeries, { lineColor: css("--sell"), topColor: "transparent", bottomColor: css("--sell-tint"), title: "Drop %", priceLineVisible: false }, 1);
  dd.setData(d.drawdown.map((p) => ({ time: p.time, value: -p.value * 100 })));
  try { labChart.panes()[0].setStretchFactor(3); } catch { /* cosmetic */ }
  labChart.timeScale().fitContent();
}
function tradesTable(d) {
  const rows = d.trades.slice(-12).reverse().map((t) => h("tr", {},
    h("td", { text: new Date(t.entry_time * 1000).toLocaleDateString("en-US", { timeZone: NY, year: "numeric", month: "short", day: "numeric" }) }),
    h("td", { class: "num", text: `${t.entry_price.toLocaleString("en-US", { maximumFractionDigits: 2 })} to ${t.exit_price.toLocaleString("en-US", { maximumFractionDigits: 2 })}` }),
    h("td", { class: "num", text: t.bars + " bars" }),
    h("td", { class: "num " + (t.return >= 0 ? "up" : "down"), text: pctS(t.return, 2) + (t.open ? " (open)" : "") })));
  if (!rows.length) return h("p", { class: "hint", text: "This rule made no trades in this period." });
  return h("div", {}, h("h3", { text: "Latest trades, after costs" }),
    h("table", { class: "lab-table" }, h("thead", {}, h("tr", {}, ...["Entered", "Price", "Held", "Result"].map((t) => h("th", { scope: "col", text: t })))), h("tbody", {}, ...rows)));
}

$("lab-strat").replaceChildren(...Object.entries(STRATS).map(([k, v]) => h("option", { value: k, text: v.label })));
$("lab-strat").addEventListener("change", renderLabParams);
renderLabParams();
$("lab-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("lab-run");
  btn.disabled = true;
  $("lab-out").replaceChildren(h("div", { class: "skeleton", text: "Running the backtest..." }));
  try { drawLabChart(await api("/api/backtest?" + labQuery())); }
  catch (e) { $("lab-out").replaceChildren(h("div", { class: "empty", text: "Backtest failed: " + e.message })); }
  finally { btn.disabled = false; }
});
