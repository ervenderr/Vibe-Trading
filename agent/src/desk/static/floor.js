"use strict";
/* Agent floor: the desk's agents as live cards. Read-only. Briefings come from the Claude Code agents,
   bot status from the journal, prices from the tick endpoint. All text is inserted with textContent. */

const COINS = ["BTC/USD", "ETH/USD", "SOL/USD"];
const BOTS = [
  { name: "Executor", role: "SOL active bot", tag: "CRYPTO DEMO" },
  { name: "Trend Bot", role: "Daily BTC/ETH rules", tag: "CRYPTO BOT", daily: true },
];
const prices = new Map();
const feedSeen = new Set();
let briefings = [];

function chipClass(call) { return /LONG|BUY/.test(call) ? "buy" : /WAIT/.test(call) ? "wait" : "avoid"; }

function briefingCard(b) {
  return h("article", { class: "agent-card" },
    h("header", {}, h("span", { class: "dot live" }), h("h3", { text: b.agent }), h("span", { class: "role", text: b.role })),
    h("p", { class: "headline", text: b.headline }),
    h("div", { class: "chips" }, b.calls.map((c) => h("span", { class: "chip " + chipClass(c.call), title: c.note, text: c.symbol + " " + c.call }))),
    h("ul", {}, b.bullets.map((t) => h("li", { text: t }))),
    h("footer", { class: "sub", text: b.t ? "Briefed " + ago(b.t) : "" }));
}

function botStatus(bot, journal) {
  const rows = journal.filter((r) => (r.msg || "").startsWith(bot.tag));
  const last = rows[rows.length - 1];
  const msg = last ? last.msg : "";
  const state = /entered|holding|bought/.test(msg) && !/closed|sold/.test(msg) ? "HOLDING"
    : /closed|sold/.test(msg) ? "FLAT" : /no entry|flat|waiting/i.test(msg) ? "SCANNING" : last ? "IDLE" : "OFF";
  const fresh = last && Date.now() - new Date(last.t).getTime() < 45 * 60000;
  return { last, msg, state, live: state !== "OFF" && fresh };
}

function botCard(bot, journal) {
  const s = botStatus(bot, journal);
  return h("article", { class: "agent-card bot " + (s.live ? "working" : "") },
    h("header", {}, h("span", { class: "dot " + (s.live ? "live" : "") }), h("h3", { text: bot.name }), h("span", { class: "role", text: bot.role })),
    h("p", { class: "headline", text: s.state + (s.live ? "" : bot.daily ? " (checks once a day)" : " (not running)") }),
    h("div", { class: "scan", "aria-hidden": "true" }),
    h("p", { class: "sub type", text: s.msg || "No activity yet." }),
    h("footer", { class: "sub", text: s.last ? "Last action " + ago(s.last.t) : "" }));
}

function renderTicker() {
  $("ticker").replaceChildren(...COINS.map((c) => {
    const p = prices.get(c);
    const cell = h("div", { class: "tick" }, h("span", { class: "sym", text: c.split("/")[0] }), h("span", { class: "px num", text: p ? p.last.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "..." }));
    if (p && p.dir) cell.classList.add(p.dir);
    return cell;
  }));
}

async function pollPrices() {
  if (document.hidden) return;
  for (const c of COINS) {
    try {
      const t = await api("/api/tick?symbol=" + encodeURIComponent(c)), last = num(t.last);
      if (last == null) continue;
      const old = prices.get(c);
      prices.set(c, { last, dir: old && last !== old.last ? (last > old.last ? "up" : "down") : old && old.dir });
    } catch { /* keep the previous price */ }
  }
  renderTicker();
}

function renderFeed(journal) {
  const items = [
    ...briefings.map((b) => ({ t: b.t, who: b.agent, text: b.headline })),
    ...journal.filter((r) => /^CRYPTO/.test(r.msg || "")).map((r) => ({ t: r.t, who: r.msg.startsWith("CRYPTO DEMO") ? "Executor" : "Trend Bot", text: r.msg.replace(/^CRYPTO (DEMO|BOT)[^:]*:?\s*/, "") })),
  ].filter((i) => i.t).sort((a, b) => new Date(b.t) - new Date(a.t)).slice(0, 14);
  $("feed").replaceChildren(...items.map((i) => {
    const key = i.t + i.who + i.text;
    const li = h("li", { class: feedSeen.has(key) ? "" : "new" }, h("span", { class: "who", text: i.who }), h("span", { class: "what", text: i.text }), h("time", { class: "sub", text: ago(i.t) }));
    feedSeen.add(key);
    return li;
  }));
}

async function pollFloor() {
  if (document.hidden) return;
  try { briefings = (await api("/api/floor")).briefings; } catch { briefings = []; }
  const journal = (lastState && lastState.journal) || [];
  $("floor").replaceChildren(...briefings.slice(-3).reverse().map(briefingCard), ...BOTS.map((b) => botCard(b, journal)));
  $("floor-note").textContent = "Paper money. Briefings are the Claude Code agents' latest read; bots update from the journal.";
  renderFeed(journal);
}

renderTicker(); pollPrices(); pollFloor();
setInterval(pollPrices, 8000);
setInterval(pollFloor, 5000);
