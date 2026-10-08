"use strict";
/* Paper Desk. Read-only: this script only ever issues GET requests. All text from the
   server (including news headlines) is inserted with textContent, never as HTML. */

const SYMBOL_RE = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMD", "META"];
const FLATTEN_MINUTES = 10;
const NY = "America/New_York";

const $ = (id) => document.getElementById(id);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid);
  return el;
}
const usd = (n, signed) => (n == null ? "n/a" : (signed && n > 0 ? "+" : "") + n.toLocaleString("en-US", { style: "currency", currency: "USD" }));
const pct = (n) => (n == null ? "n/a" : (n * 100).toFixed(1) + "%");
const num = (v) => { const n = Number(v); return Number.isFinite(n) ? n : null; };
const clock = (iso, opts) => new Date(iso).toLocaleString("en-US", { timeZone: NY, ...opts });
const ago = (iso) => {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + " min ago";
  if (s < 129600) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
};
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
async function api(path) {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || "request failed");
  return data;
}

/* ---------- header: market clock + autopilot pulse ---------- */
let lastState = null;
function renderClock() {
  const box = $("clock");
  const c = lastState && lastState.clock;
  if (!c) { box.replaceChildren(h("div", { class: "big", text: "Market clock unavailable" })); return; }
  const now = Date.now();
  if (c.is_open) {
    const close = new Date(c.next_close).getTime();
    const flatten = close - FLATTEN_MINUTES * 60000;
    const left = Math.max(0, close - now);
    box.replaceChildren(
      h("div", { class: "big", text: "Market open. Closes in " + fmtDur(left) }),
      h("div", { class: "sub", text: "The autopilot stops entering about 15 minutes before the close and closes everything it opened by " + clock(new Date(flatten).toISOString(), { hour: "numeric", minute: "2-digit" }) + " New York time." }));
  } else {
    box.replaceChildren(
      h("div", { class: "big", text: "Market closed. Opens in " + fmtDur(new Date(c.next_open).getTime() - now) }),
      h("div", { class: "sub", text: "Next open: " + clock(c.next_open, { weekday: "short", hour: "numeric", minute: "2-digit" }) + " New York time. Nothing trades while the market is closed." }));
  }
}
function fmtDur(ms) {
  const m = Math.max(0, Math.floor(ms / 60000));
  const d = Math.floor(m / 1440), hh = Math.floor((m % 1440) / 60), mm = m % 60;
  return d ? `${d}d ${hh}h` : hh ? `${hh}h ${String(mm).padStart(2, "0")}m` : `${mm}m`;
}
function renderPilot(state) {
  const beat = state.last_beat;
  const box = $("pilot");
  const age = beat ? (Date.now() - new Date(beat.t).getTime()) / 1000 : Infinity;
  const grace = (beat && beat.next_s ? beat.next_s : 300) + 120;   // it may sleep up to 5 minutes between looks
  const running = age <= grace;
  const dot = h("span", { class: "dot " + (state.halted ? "off" : running ? "live" : "") });
  let text;
  if (state.halted) text = "Kill switch is on. The autopilot will not trade.";
  else if (!beat) text = "No autopilot heartbeat yet.";
  else if (running) text = `Autopilot is running. Last look ${ago(beat.t)}: ${beat.msg}.`;
  else text = `Autopilot has gone quiet. Last heard ${ago(beat.t)}.`;
  const kids = [dot, text];
  if (!running && !state.halted) kids.push(h("div", {}, "Start it in a terminal: ", h("code", { text: "vibe-trading autopilot NVDA TSLA --execute" })));
  box.replaceChildren(h("div", {}, ...kids));
}
function renderBanner(state) {
  const b = $("banner");
  if (state.error) { b.hidden = false; b.className = "banner"; b.textContent = state.error + ". Check ~/.vibe-trading/alpaca.json and your connection."; }
  else if (state.paper === false) { b.hidden = false; b.className = "banner"; b.textContent = "This account is not marked as paper. Do not use it with this tool."; }
  else if (state.halted) { b.hidden = false; b.className = "banner info"; b.textContent = "Kill switch is on. Remove the HALT file under ~/.vibe-trading/live to let the autopilot trade again."; }
  else b.hidden = true;
}

/* ---------- account + positions ---------- */
function fig(label, value, cls) { return h("div", { class: "fig" }, h("b", { class: cls || "", text: value }), h("span", { text: label })); }
function renderAccount(state) {
  const a = state.account;
  if (!a) { $("account").replaceChildren(h("div", { class: "empty", text: "No account data yet." })); return; }
  const dc = a.day_change;
  $("account").replaceChildren(
    fig("Account value", usd(a.equity)),
    fig("Change since yesterday's close", usd(dc, true), dc > 0 ? "up" : dc < 0 ? "down" : ""),
    fig("Buying power", usd(a.buying_power)),
  );
}
function renderPositions(state) {
  const box = $("positions");
  const rows = [];
  for (const p of state.positions || []) {
    const qty = num(p.qty), pl = num(p.unrealized_pl);
    const side = qty != null && qty < 0 ? "short" : "long";
    rows.push(h("div", { class: "pos-row " + side },
      h("span", { class: "s", text: String(p.symbol || "?").slice(0, 10) }),
      h("span", {}, `${side === "short" ? "Short" : "Long"} ${Math.abs(qty ?? 0)} shares`,
        h("small", { text: "Bought at " + usd(num(p.avg_entry_price)) + ", now " + usd(num(p.current_price)) })),
      h("b", { class: "num " + (pl > 0 ? "up" : pl < 0 ? "down" : ""), text: usd(pl, true) })));
  }
  const orders = state.orders || [];
  if (orders.length) rows.push(h("p", { class: "hint", text: `${orders.length} waiting order${orders.length === 1 ? "" : "s"}: stops and targets that close a position when price reaches them.` }));
  box.replaceChildren(...(rows.length ? rows : [h("div", { class: "empty", text: "Nothing open. The account is flat." })]));
}

/* ---------- promotion gate + trace ---------- */
function renderGate(track) {
  const box = $("gate");
  if (track.error || !track.promotion) { box.replaceChildren(h("div", { class: "empty", text: track.error || "No data." })); return; }
  const items = track.promotion.checks.map((c) =>
    h("li", {}, h("span", { class: "mark " + (c.ok ? "ok" : ""), text: c.ok ? "✓" : "" }),
      h("span", {}, c.name.charAt(0).toUpperCase() + c.name.slice(1), h("small", { class: "num", text: c.detail }))));
  box.replaceChildren(h("ul", { class: "checks" }, ...items),
    h("p", { class: "verdict-line", text: track.promotion.eligible
      ? "All six tests pass. That only makes a conversation about live trading possible. It does not promise profit."
      : "Not ready, and these tests were fixed before any results existed. Going live always stays a manual decision." }));
}
function renderTrace(track) {
  const box = $("trace");
  const pts = track.curve || [];
  if (track.error || !pts.length) {
    box.replaceChildren(h("div", { class: "empty", text: track.error || "No closed trades yet. When the autopilot closes its first trade, the profit and loss line starts here." }));
    return;
  }
  const W = 900, H = 260, L = 56, R = 70, T = 16, B = 28;
  const ys = [0, ...pts.map((p) => p.cum_pnl)];
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < 1) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
  const X = (i) => L + (pts.length === 1 ? (W - L - R) / 2 : (i / (pts.length - 1)) * (W - L - R));
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ns = "http://www.w3.org/2000/svg";
  const s = (tag, attrs, text) => { const e = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Profit and loss across ${pts.length} closed trades, ending at ${usd(pts[pts.length - 1].cum_pnl, true)}` });
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = Y(v);
    svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y, y2: y }), s("text", { x: L - 8, y: y + 4, "text-anchor": "end" }, usd(v)));
  }
  for (let i = 0; i < pts.length; i += Math.max(1, Math.ceil(pts.length / 10))) svg.append(s("line", { class: "grid", x1: X(i), x2: X(i), y1: T, y2: H - B }));
  svg.append(s("line", { class: "zero", x1: L, x2: W - R, y1: Y(0), y2: Y(0) }));
  svg.append(s("polyline", { class: "pen", points: pts.map((p, i) => `${X(i)},${Y(p.cum_pnl)}`).join(" ") }));
  const last = pts[pts.length - 1];
  svg.append(s("circle", { cx: X(pts.length - 1), cy: Y(last.cum_pnl), r: 4, fill: "currentColor" }),
    s("text", { x: X(pts.length - 1), y: Y(last.cum_pnl) - 12, "text-anchor": "middle" }, usd(last.cum_pnl, true)));
  const sm = track.summary;
  const pf = sm.profit_factor == null ? "no losing trades yet" : sm.profit_factor.toFixed(2);
  box.replaceChildren(svg, h("p", { class: "cap", text:
    `${sm.trades} closed trade${sm.trades === 1 ? "" : "s"} over ${sm.trading_days} day${sm.trading_days === 1 ? "" : "s"}. Won ${pct(sm.win_rate)} of them. Profit factor ${pf} (above 1 means winners outweigh losers). The dashed line is break-even.` }));
}

/* ---------- logbook ---------- */
function renderLog(state) {
  const rows = (state.journal || []).slice().reverse();
  $("log").replaceChildren(...(rows.length
    ? rows.map((e) => h("li", {}, h("time", { datetime: e.t, text: clock(e.t, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", second: "2-digit" }) }), h("span", { text: e.msg })))
    : [h("li", {}, h("span", {}), h("span", { text: "Nothing logged yet. Every decision the autopilot makes, including why it skipped a trade, is written here." }))]));
}

/* ---------- signal strips ---------- */
const VERDICT = {
  BUY: ["Buy", "buy", "Looks like a buy"],
  SELL: ["Sell", "sell", "Looks like a sell"],
  WAIT: ["Wait", "wait", "Waiting for a clearer picture"],
  AVOID: ["Avoid", "avoid", "Staying out"],
};
let symbols = store.get("desk.symbols", DEFAULT_SYMBOLS).filter((s) => SYMBOL_RE.test(s));
const open = new Set(store.get("desk.open", []));
const signals = new Map();

function ladder(l) {
  const e = num(l.entry), st = num(l.stop), t = num(l.target);
  if (e == null || st == null || t == null) return null;
  const lo = Math.min(st, t), hi = Math.max(st, t), span = hi - lo || 1;
  const at = (v) => 4 + ((v - lo) / span) * 92;
  const tick = (label, v, cls) => h("div", { class: "tick", style: `left:${at(v)}%` }, h("b", { text: label }), h("span", { class: "num", text: usd(v) }));
  return h("div", {}, h("p", { class: "hint", text: "If it trades, the entry, the stop that cuts a loss, and the target that locks a gain go in together as one order." }),
    h("div", { class: "ladder" }, h("div", { class: "line" }), tick("Stop", st), tick("Entry", e), tick("Target", t)));
}
function mood(s) {
  if (s == null) return ["No read", ""];
  return s > 0.15 ? ["Positive", "pos"] : s < -0.15 ? ["Negative", "neg"] : ["Neutral", ""];
}
function newsList(headlines) {
  if (!headlines.length) return h("p", { class: "hint", text: "No recent news in the last two days for this symbol." });
  return h("ul", { class: "news" }, ...headlines.slice(0, 6).map((n) => {
    const [label, cls] = mood(n.sentiment);
    const head = n.url ? h("a", { href: n.url, target: "_blank", rel: "noopener noreferrer", text: n.headline }) : h("span", { text: n.headline });
    const bits = [n.source, n.published ? ago(n.published) : null, n.event_type, n.origin === "fallback" ? "keyword read only, ignored for the verdict" : null].filter(Boolean);
    return h("li", {}, h("span", { class: "mood " + cls, text: label }), h("div", {}, head, h("div", { class: "meta", text: bits.join(", ") })));
  }));
}
function stripFor(sym) {
  const sig = signals.get(sym);
  const root = h("div", { class: "strip " + (sig && sig.action ? (VERDICT[sig.action] || VERDICT.AVOID)[1] : "avoid"), "data-sym": sym });
  const isOpen = open.has(sym);
  const head = h("button", { class: "head", type: "button", "aria-expanded": String(isOpen), "aria-label": `${sym} details` });
  head.append(h("span", { class: "sym", text: sym }));
  if (!sig) {
    head.append(h("span", { class: "verdict", text: "Checking" }), h("span", { class: "gist", text: "Reading news and price. The first read can take a minute." }), h("span"));
  } else if (sig.error) {
    head.append(h("span", { class: "verdict", text: "No read" }), h("span", { class: "gist", text: sig.error }), h("span"));
  } else {
    const [name, , gist] = VERDICT[sig.action] || VERDICT.AVOID;
    const pctConf = Math.round((sig.confidence || 0) * 100);
    head.append(h("span", { class: "verdict", text: name }),
      h("span", { class: "gist", text: (sig.reasons && sig.reasons[0]) || gist }),
      h("span", { class: "meter" }, h("div", { class: "track", role: "img", "aria-label": `Confidence ${pctConf} percent` }, h("div", { class: "fill", style: `width:${pctConf}%` })), h("div", { class: "cap", text: pctConf + "% sure" })));
  }
  head.addEventListener("click", () => {
    if (open.has(sym)) open.delete(sym); else open.add(sym);
    store.set("desk.open", [...open]);
    paintStrip(sym);
  });
  root.append(head);
  if (sig && !sig.error) {
    const body = h("div", { class: "body", hidden: !isOpen });
    const lad = sig.action === "BUY" || sig.action === "SELL" ? ladder(sig.levels || {}) : null;
    body.append(h("div", {}, h("p", { text: `${(VERDICT[sig.action] || VERDICT.AVOID)[2]}. Read ${ago(sig.generated_at)}${sig.stale ? ", but the price data is stale, so it will not act on it" : ""}.` }),
      h("ul", { class: "reasons" }, ...(sig.reasons || []).map((r) => h("li", { text: r }))),
      h("p", { class: "hint", text: `News score ${(sig.news_score ?? 0).toFixed(2)}, price score ${(sig.tech_score ?? 0).toFixed(2)}. Both run from -1 (bad) to +1 (good).` })),
      lad, newsList(sig.headlines || []),
      sig.news_error ? h("p", { class: "hint", text: "News source problem: " + sig.news_error }) : null);
    root.append(body);
  }
  const remove = h("button", { type: "button", class: "rm", text: "Remove " + sym + " from this board" });
  remove.addEventListener("click", () => { symbols = symbols.filter((s) => s !== sym); store.set("desk.symbols", symbols); root.remove(); });
  const body = root.querySelector(".body");
  if (body) body.append(remove); else root.append(h("div", { class: "body" }, remove));
  return root;
}
function paintStrip(sym, fresh) {
  const old = document.querySelector(`[data-sym="${CSS.escape(sym)}"]`);
  const next = stripFor(sym);
  if (fresh) next.classList.add("fresh");
  if (old) old.replaceWith(next); else $("strips").append(next);
}
async function loadSignals() {
  const btn = $("refresh");
  btn.disabled = true;
  if (!$("strips").children.length) $("strips").replaceChildren(...symbols.map((s) => stripFor(s)));
  for (const sym of [...symbols]) {           // one at a time: each read calls the news and AI services
    try { signals.set(sym, await api("/api/signal?symbol=" + encodeURIComponent(sym))); }
    catch (e) { signals.set(sym, { error: e.message }); }
    if (symbols.includes(sym)) paintStrip(sym, true);
  }
  btn.disabled = false;
}

/* ---------- polling ---------- */
async function pollState() {
  if (document.hidden) return;
  try { lastState = await api("/api/state"); } catch (e) { lastState = { error: e.message, journal: [], positions: [], orders: [] }; }
  renderBanner(lastState); renderClock(); renderPilot(lastState); renderAccount(lastState); renderPositions(lastState); renderLog(lastState);
}
async function pollTrack() {
  if (document.hidden) return;
  let t; try { t = await api("/api/track?days=30"); } catch (e) { t = { error: e.message }; }
  renderGate(t); renderTrace(t);
}
$("add").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const sym = $("sym").value.trim().toUpperCase();
  if (!SYMBOL_RE.test(sym)) { $("sym").setCustomValidity("Use a ticker such as MSFT or BRK.B"); $("sym").reportValidity(); return; }
  $("sym").setCustomValidity("");
  $("sym").value = "";
  if (symbols.includes(sym)) return;
  symbols = [...symbols, sym]; store.set("desk.symbols", symbols);
  paintStrip(sym);
  api("/api/signal?symbol=" + encodeURIComponent(sym)).then((r) => signals.set(sym, r), (e) => signals.set(sym, { error: e.message })).then(() => paintStrip(sym, true));
});
$("sym").addEventListener("input", () => $("sym").setCustomValidity(""));
$("refresh").addEventListener("click", loadSignals);

pollState(); pollTrack(); loadSignals();
setInterval(pollState, 5000);
setInterval(pollTrack, 60000);
setInterval(renderClock, 1000);
setInterval(() => { if (!document.hidden) loadSignals(); }, 5 * 60000);
