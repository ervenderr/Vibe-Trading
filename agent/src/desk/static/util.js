"use strict";
/* Shared helpers for the dashboard and the chart terminal. Text from the server is always inserted
   with textContent, never as HTML. */

const SYMBOL_RE = /^[A-Z][A-Z0-9.\-]{0,9}$/;
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

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
