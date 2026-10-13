"use strict";
/* AI research desk: shows TradingAgents' saved reports. The text is model output, so it is only ever
   inserted with textContent (a tiny Markdown subset is rebuilt as DOM nodes, never parsed as HTML). */

const REPORT_LABELS = {
  market_report: "Price and indicators", sentiment_report: "Sentiment", news_report: "News", fundamentals_report: "Fundamentals",
  investment_plan: "Bull vs bear debate", trader_investment_plan: "Trader", risk_report: "Risk review", final_trade_decision: "Final decision",
};
let runs = [], current = null, currentName = null;
const labelOf = (name) => REPORT_LABELS[name] || name.replace(/_/g, " ");

function inline(text) {   // **bold** and `code`, as DOM nodes
  const out = [];
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) out.push(h("strong", { text: part.slice(2, -2) }));
    else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) out.push(h("code", { text: part.slice(1, -1) }));
    else if (part) out.push(part);
  });
  return out;
}
function renderMarkdown(src) {
  const nodes = [], lines = src.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) { nodes.push(h(heading[1].length <= 2 ? "h3" : "h4", {}, ...inline(heading[2]))); continue; }
    if (/^\s*\|/.test(line)) {                                   // table block
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { if (!/^\s*\|[\s:|-]+\|?\s*$/.test(lines[i])) rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim())); i++; }
      i--;
      if (rows.length) nodes.push(h("div", { class: "table-wrap" }, h("table", { class: "lab-table" },
        h("thead", {}, h("tr", {}, ...rows[0].map((c) => h("th", { scope: "col" }, ...inline(c))))),
        h("tbody", {}, ...rows.slice(1).map((r) => h("tr", {}, ...r.map((c) => h("td", {}, ...inline(c)))))))));
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {                              // bullet list block
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(h("li", {}, ...inline(lines[i].replace(/^\s*[-*]\s+/, "")))); i++; }
      i--; nodes.push(h("ul", {}, ...items)); continue;
    }
    if (line.trim()) nodes.push(h("p", {}, ...inline(line)));
  }
  return nodes;
}

function renderRunPick() {
  const pick = $("run-pick");
  pick.replaceChildren(...runs.map((r) => h("option", { value: `${r.ticker}|${r.date}`, text: `${r.ticker}  ${r.date}${r.complete ? "" : "  (in progress)"}` })));
  if (current) pick.value = `${current.ticker}|${current.date}`;
}
function renderRun() {
  if (!current) return;
  $("run-status").replaceChildren(h("span", { class: "chip " + (current.complete ? "done" : "wip"), text: current.complete ? "Finished" : "Still running: more reports will appear" }));
  $("report-tabs").replaceChildren(...current.reports.map((r) => {
    const tab = h("button", { type: "button", role: "tab", "aria-selected": String(r.name === currentName), text: labelOf(r.name) });
    tab.addEventListener("click", () => { currentName = r.name; renderRun(); });
    return tab;
  }));
  const report = current.reports.find((r) => r.name === currentName) || current.reports[current.reports.length - 1];
  if (!report) { $("report-body").replaceChildren(h("p", { class: "hint", text: "No reports yet." })); return; }
  currentName = report.name;
  const kids = renderMarkdown(report.text);
  if (report.reasoning) kids.push(h("details", {}, h("summary", { text: "Model's raw reasoning (hidden by default)" }), h("pre", { text: report.reasoning })));
  $("report-body").replaceChildren(...kids);
}
async function loadRun(ticker, date) {
  try { current = await api(`/api/agents/run?ticker=${encodeURIComponent(ticker)}&date=${encodeURIComponent(date)}`); renderRun(); }
  catch (e) { $("report-body").replaceChildren(h("div", { class: "empty", text: "Could not load this run: " + e.message })); }
}
async function loadRuns() {
  if (document.hidden) return;
  try {
    runs = (await api("/api/agents")).runs;
    if (!runs.length) {
      $("run-pick").replaceChildren();
      $("report-tabs").replaceChildren();
      $("report-body").replaceChildren(h("div", { class: "empty" }, "No research runs yet. Start one in a terminal: ",
        h("code", { text: "cd ~/Projects/TradingAgents && .venv/bin/tradingagents --ticker NVDA --date 2026-10-13 --save --no-show" })));
      return;
    }
    const keep = current && runs.find((r) => r.ticker === current.ticker && r.date === current.date);
    const target = keep || runs[0];
    renderRunPick();
    if (!current || !keep || (!current.complete) || current.reports.length !== target.reports.length) await loadRun(target.ticker, target.date);
    current && ($("run-pick").value = `${current.ticker}|${current.date}`);
  } catch (e) { $("report-body").replaceChildren(h("div", { class: "empty", text: "Research desk: " + e.message })); }
}
$("run-pick").addEventListener("change", () => { const [t, d] = $("run-pick").value.split("|"); currentName = null; loadRun(t, d); });
loadRuns();
setInterval(loadRuns, 10000);
