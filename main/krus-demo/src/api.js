const API_BASE = (process.env.REACT_APP_API_BASE || "http://localhost:8000").replace(/\/$/, "");

export async function apiAsk(question, reset_memory = false) {
  const r = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question, reset_memory })
  });
  if (!r.ok) throw new Error("ask failed");
  return await r.json(); // { answer, citations: [...] }
}

export async function apiFollowUp() {
  const r = await fetch(`${API_BASE}/followup`, { method: "POST" });
  if (!r.ok) throw new Error("followup failed");
  return await r.json(); // { ok: true, ...}
}

export async function apiReset() {
  const r = await fetch(`${API_BASE}/reset`, { method: "POST" });
  if (!r.ok) throw new Error("reset failed");
  return await r.json(); // { ok: true }
}
