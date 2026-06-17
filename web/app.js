/* ============================================================
   OASIS Clinical Console — frontend controller.
   Plain ES2020 (no framework / no CDN) so it deploys to any
   browser on any device, including hospital displays and TVs.
   Talks to the FastAPI backend on the same origin.
   ============================================================ */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { imageB64: null, busy: false };

/* ---------------- utilities ---------------- */
function toast(msg, isErr = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => (t.hidden = true), 3800);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function colorForClass(label) {
  const l = (label || "").toLowerCase();
  if (l.includes("non")) return "var(--good)";
  if (l.includes("very mild")) return "var(--warn)";
  if (l.includes("moderate")) return "var(--bad)";
  if (l.includes("mild")) return "var(--warn)";
  return "var(--accent)";
}

/* ---------------- theme ---------------- */
const THEMES = ["midnight", "ocean", "slate", "amber", "clinical", "t6", "t7", "t8", "hc"];
function applyTheme(name) {
  if (!THEMES.includes(name)) name = "midnight";
  if (name === "midnight") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = name;
  try { localStorage.setItem("oasis-theme", name); } catch (_) {}
  const sel = $("themeSelect");
  if (sel) sel.value = name;
}
function initTheme() {
  let saved = "midnight";
  try {
    saved = localStorage.getItem("oasis-theme")
      || new URLSearchParams(location.search).get("theme") || "midnight";
  } catch (_) {}
  applyTheme(saved);
  const sel = $("themeSelect");
  if (sel) sel.addEventListener("change", () => applyTheme(sel.value));
}

/* ---------------- status / clock ---------------- */
function startClock() {
  const tick = () => ($("clock").textContent = new Date().toLocaleTimeString());
  tick(); setInterval(tick, 1000);
}

async function refreshStatus() {
  try {
    const h = await api("/health");
    const pill = $("pillHealth");
    const ok = h.status === "healthy";
    pill.innerHTML = `<span class="dot ${ok ? "dot-ok" : "dot-idle"}"></span><span>${ok ? "System ready" : "Initializing…"}</span>`;
    if (!ok) { setTimeout(refreshStatus, 2500); return; }
    try {
      const info = await api("/models/info");
      const provs = (info.acceleration || []).map((p) => p.replace("ExecutionProvider", ""));
      $("pillAccel").textContent = "Accel: " + (provs[0] || "CPU");
      $("pillAccel").title = "Provider chain: " + (info.acceleration || []).join(" → ");
      const llm = (info.llm || "");
      const m = llm.match(/edge=([^@\s]+)/);
      $("pillLLM").textContent = "LLM: " + (m ? m[1] : (llm.includes("llm=on") ? "Ollama" : "template"));
      $("footMeta").textContent = `device ${info.device} · ${(info.acceleration || []).length} providers`;
    } catch (_) {}
  } catch (e) {
    $("pillHealth").innerHTML = `<span class="dot dot-bad"></span><span>API offline</span>`;
    setTimeout(refreshStatus, 4000);
  }
}

/* ---------------- image loading ---------------- */
function setBaseImage(b64) {
  state.imageB64 = b64;
  $("imgBase").src = "data:image/png;base64," + b64;
  $("imgBase").style.display = "block";
  $("viewportEmpty").style.display = "none";
  $("btnRun").disabled = false;
}

async function loadSample() {
  try {
    const s = await api("/api/sample");
    setBaseImage(s.image_base64);
    $("imgOverlay").src = "";
    $("scanMeta").textContent = `sample · ${s.filename} · truth: ${s.true_label}`;
    toast(`Loaded sample (ground truth: ${s.true_label})`);
  } catch (e) { toast("Could not load sample: " + e.message, true); }
}

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const b64 = String(reader.result).split(",")[1];
    setBaseImage(b64);
    $("imgOverlay").src = "";
    $("scanMeta").textContent = `uploaded · ${file.name}`;
  };
  reader.readAsDataURL(file);
}

/* ---------------- agent rail animation ---------------- */
const AGENT_SEQ = ["vision", "explainer", "biomarker", "temporal", "volumetry", "rag", "ethicist", "reasoner"];
function railReset() {
  document.querySelectorAll("#agentRail li").forEach((li) => (li.className = ""));
}
function railRun() {
  railReset();
  let i = 0;
  const tick = () => {
    if (i > 0) document.querySelector(`#agentRail li[data-agent="${AGENT_SEQ[i - 1]}"]`)?.classList.replace("active", "done");
    if (i < AGENT_SEQ.length) {
      document.querySelector(`#agentRail li[data-agent="${AGENT_SEQ[i]}"]`)?.classList.add("active");
      i++; state._rail = setTimeout(tick, 260);
    }
  };
  tick();
}
function railFinish() {
  clearTimeout(state._rail);
  document.querySelectorAll("#agentRail li").forEach((li) => { li.classList.remove("active"); li.classList.add("done"); });
}

/* ---------------- run analysis ---------------- */
async function runAnalysis(evt) {
  evt?.preventDefault();
  if (state.busy) return;
  if (!state.imageB64) { toast("Load or upload a scan first.", true); return; }
  state.busy = true;
  $("btnRun").disabled = true;
  $("btnRun").querySelector(".run-label").textContent = "Analyzing…";
  $("btnRun").querySelector(".spinner").hidden = false;
  railRun();

  const payload = {
    patient_data: {
      patient_id: $("patientId").value || "UNKNOWN",
      age: parseFloat($("age").value) || 0,
      mmse: parseFloat($("mmse").value) || 0,
      gender: $("sex").value,
      education: parseInt($("education").value) || null,
    },
    image_base64: state.imageB64,
    longitudinal_id: $("longId").value || null,
  };

  try {
    const r = await api("/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    railFinish();
    renderResult(r);
  } catch (e) {
    toast("Analysis failed: " + e.message, true);
    railReset();
  } finally {
    state.busy = false;
    $("btnRun").disabled = false;
    $("btnRun").querySelector(".run-label").textContent = "▶ Run Multi-Agent Analysis";
    $("btnRun").querySelector(".spinner").hidden = true;
  }
}

/* ---------------- rendering ---------------- */
function renderResult(r) {
  const v = r.vision_prediction || {};
  const expl = r.explainability || {};

  // Grad-CAM overlay
  if (expl.overlay_png) {
    $("imgBase").src = "data:image/png;base64," + (expl.base_png || state.imageB64);
    $("imgOverlay").src = "data:image/png;base64," + expl.overlay_png;
    $("viewportLegend").hidden = false;
    applyOverlay();
  }

  // Verdict + confidence ring
  const verdict = $("verdict");
  verdict.hidden = false;
  $("verdictClass").textContent = r.final_diagnosis === "DIAGNOSIS_BLOCKED" ? "Withheld — human review" : (v.class || "—");
  $("verdictSub").textContent = r.approved ? "authorized by ethics guardrail" : "blocked by ethics guardrail";
  verdict.className = "verdict " + (r.approved ? "ok" : "flag");
  const conf = Math.round(v.confidence || 0);
  $("confVal").textContent = conf + "%";
  const c = colorForClass(v.class);
  $("confRing").style.background = `conic-gradient(${c} ${conf * 3.6}deg, var(--line) 0deg)`;

  // Ethics banner
  const eb = $("ethicsBanner");
  const ethics = r.ethics_audit || {};
  eb.hidden = false;
  eb.className = "ethics-banner " + (ethics.approved ? "ok" : "flag");
  eb.innerHTML = `<b>${ethics.approved ? "✓ Ethics: cleared" : "⚠ Ethics: flagged"}</b>&nbsp;${ethics.message || ""}`;

  // Probability bars
  renderProbBars(v.probabilities || [], v.class);

  // Volumetry
  renderVolumetry(r.regional_volumetry || {});

  // ATN biomarker profile
  renderATN(r.atn_profile || {});

  // Differential diagnosis + anatomical brain map
  renderDifferential(r.differential || {});
  updateBrainMap(r.regional_volumetry || {}, r.atn_profile || {});

  // Temporal + biomarker
  const t = r.temporal_analysis || {};
  $("tTrend").textContent = t.trend || "N/A";
  $("tAtrophy").textContent = (t.atrophy_velocity != null ? t.atrophy_velocity.toFixed(2) : "0.00") + " %/yr";
  const b = r.biomarker_analysis || {};
  $("bAge").textContent = b.age_risk || "—";
  $("bMmse").textContent = (b.mmse_category || "—").replace(/_/g, " ");

  // Reasoner narrative
  $("reasonerTier").textContent = r.reasoning_tier || "—";
  const narr = $("narrative");
  narr.textContent = r.clinical_narrative || "(no narrative)";
  narr.className = "narrative" + (r.approved ? "" : " flagged");

  // RAG evidence
  const ev = $("evidence");
  ev.innerHTML = "";
  (r.rag_context || []).forEach((doc) => {
    const li = document.createElement("li");
    li.textContent = doc;
    ev.appendChild(li);
  });
  if (!ev.children.length) ev.innerHTML = '<li class="muted">No evidence retrieved.</li>';

  toast(`Analysis complete — ${v.class} (${conf}%)`);
}

const CLASS_ORDER = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"];
function renderProbBars(probs, topClass) {
  const wrap = $("probBars");
  if (!probs.length) { wrap.innerHTML = '<p class="muted">No image prediction.</p>'; return; }
  wrap.innerHTML = "";
  probs.forEach((p, i) => {
    const label = CLASS_ORDER[i] || `Class ${i}`;
    const pct = (p * 100);
    const row = document.createElement("div");
    row.className = "bar-row" + (label === topClass ? " top" : "");
    row.innerHTML = `
      <div class="bar-head"><span>${label}</span><b>${pct.toFixed(1)}%</b></div>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>`;
    wrap.appendChild(row);
    requestAnimationFrame(() => (row.querySelector(".bar-fill").style.width = pct.toFixed(1) + "%"));
  });
}

function renderVolumetry(vol) {
  const wrap = $("zBars");
  const summary = $("volSummary");
  const regions = vol.regions || [];
  summary.textContent = vol.summary || "";
  if (!regions.length) {
    wrap.innerHTML = `<p class="muted">${vol.source === "estimated"
      ? "Whole-brain estimate only — supply a FreeSurfer subject for regional detail."
      : "No regional segmentation available (provide a longitudinal/FreeSurfer ID)."}</p>`;
    return;
  }
  wrap.innerHTML = "";
  regions.forEach((reg) => {
    const z = Math.max(-3, Math.min(3, reg.z_score));
    const half = 50; // percent half-width
    const mag = Math.abs(z) / 3 * half;
    const left = z < 0 ? half - mag : half;
    const color = reg.abnormal ? "var(--bad)" : (Math.abs(z) > 1 ? "var(--warn)" : "var(--accent-2)");
    const row = document.createElement("div");
    row.className = "zrow";
    row.innerHTML = `
      <span class="zlab" title="${reg.structure}">${reg.structure.replace(/-/g, " ")}</span>
      <div class="zaxis"><div class="zfill" style="left:${left}%;width:${mag}%;background:${color}"></div></div>
      <span class="zval" style="color:${color}">${reg.z_score >= 0 ? "+" : ""}${reg.z_score.toFixed(1)}</span>`;
    wrap.appendChild(row);
  });
}

function renderATN(atn) {
  const map = { positive: ["+", "pos"], negative: ["-", "neg"], indeterminate: ["?", "ind"] };
  const set = (id, status) => {
    const el = $(id);
    const [sym, cls] = map[status] || ["?", "ind"];
    el.textContent = sym;
    el.parentElement.className = "atn-chip " + cls;
  };
  set("atnA", atn.a_status || "indeterminate");
  set("atnT", atn.t_status || "indeterminate");
  set("atnN", atn.n_status || "indeterminate");
  $("atnProfile").textContent = atn.profile || "—";
  let cat = atn.category || "—";
  const pet = atn.pet;
  if (pet && (pet.amyloid_suvr != null || pet.tau_suvr != null)) {
    const bits = [];
    if (pet.amyloid_suvr != null) bits.push(`amyloid ${pet.amyloid_tracer} ${pet.amyloid_centiloid} CL`);
    if (pet.tau_suvr != null) bits.push(`tau ${pet.tau_tracer} SUVR ${pet.tau_suvr}`);
    cat += `  ·  PET: ${bits.join(" · ")}`;
  } else if (atn.centiloid != null) {
    cat += `  ·  ${atn.centiloid} CL`;
  }
  $("atnCategory").textContent = cat;
}

/* ---------------- differential diagnosis ---------------- */
function renderDifferential(diff) {
  const wrap = $("diffBars");
  const ranking = diff.ranking || [];
  $("diffProvider").textContent = diff.provider ? diff.provider.split(":")[0] : "Agent 11";
  if (!ranking.length) { wrap.innerHTML = '<p class="muted">No differential.</p>'; return; }
  wrap.innerHTML = "";
  const max = Math.max(...ranking.map((r) => r.likelihood || 0), 1);
  ranking.forEach((r, i) => {
    const pct = r.likelihood || 0;
    const row = document.createElement("div");
    row.className = "bar-row" + (i === 0 ? " top" : "");
    row.title = r.rationale || "";
    row.innerHTML = `
      <div class="bar-head"><span>${r.etiology}</span><b>${pct}%</b></div>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>`;
    wrap.appendChild(row);
    requestAnimationFrame(() => (row.querySelector(".bar-fill").style.width = (pct / max * 100) + "%"));
  });
  $("diffWorkup").textContent = (diff.recommended_workup || []).length
    ? "Work-up: " + diff.recommended_workup.slice(0, 3).join("; ") : "";
}

/* ---------------- anatomical brain map (glowing regions) ---------------- */
const REGION_MAP = {
  "Left-Hippocampus": "r-hippo-l", "Right-Hippocampus": "r-hippo-r",
  "Left-Amygdala": "r-amyg-l", "Right-Amygdala": "r-amyg-r",
  "Left-Lateral-Ventricle": "r-vent-l", "Right-Lateral-Ventricle": "r-vent-r",
  "3rd-Ventricle": "r-vent3",
};
function glowColor(sev) {
  // teal -> amber -> red as severity rises
  if (sev < 0.34) return "#21c7b6";
  if (sev < 0.67) return "#f0a93b";
  return "#ef5a5a";
}
function setRegionGlow(el, sev, label, detail) {
  const color = glowColor(sev);
  el.style.fill = sev > 0.05
    ? `color-mix(in srgb, ${color} ${Math.round(20 + sev * 70)}%, #18212c)`
    : "#1d2733";
  el.style.filter = sev > 0.05 ? `drop-shadow(0 0 ${4 + sev * 22}px ${color})` : "none";
  el.dataset.detail = detail || "";
}
function updateBrainMap(vol, atn) {
  state.brainData = { vol, atn };
  // reset all
  document.querySelectorAll("#brainMap .region").forEach((el) => setRegionGlow(el, 0));
  const regions = vol.regions || [];
  if (regions.length) {
    regions.forEach((reg) => {
      const id = REGION_MAP[reg.structure];
      if (!id) return;
      const el = $(id);
      if (!el) return;
      // oriented z: negative = abnormal. severity from how negative.
      const sev = Math.max(0, Math.min(1, -reg.z_score / 3));
      setRegionGlow(el, sev, reg.structure, `${reg.structure.replace(/-/g, " ")}: z=${reg.z_score >= 0 ? "+" : ""}${reg.z_score.toFixed(1)}${reg.abnormal ? " (abnormal)" : ""}`);
    });
    $("r-cortex").dataset.detail = vol.summary || "";
  } else {
    // estimated / whole-brain only: glow the cortex by MTA risk
    const sev = Math.max(0, Math.min(1, (vol.mta_risk_score || 0) / 2));
    setRegionGlow($("r-cortex"), sev, "Cortex", vol.summary || "Whole-brain estimate");
  }
}
function brainTip(evt) {
  const el = evt.target.closest(".region");
  const tip = $("brainTip");
  if (!el || !el.dataset.detail) { tip.hidden = true; return; }
  const vp = $("viewport").getBoundingClientRect();
  tip.textContent = el.dataset.detail;
  tip.style.left = (evt.clientX - vp.left + 12) + "px";
  tip.style.top = (evt.clientY - vp.top + 12) + "px";
  tip.hidden = false;
}

/* ---------------- cure-research lab ---------------- */
async function runResearch() {
  const btn = $("btnResearch");
  btn.disabled = true;
  btn.textContent = "🧪 Mining cohort…";
  try {
    const data = await api("/research");
    const report = data.report || {};
    const insight = data.insight || {};
    const fwrap = $("researchFindings");
    fwrap.innerHTML = "";
    (report.findings || []).slice(0, 6).forEach((f) => {
      const d = document.createElement("div");
      d.className = "finding";
      const p = f.p_value != null ? ` (p=${f.p_value})` : "";
      d.innerHTML = `<b>${f.name}</b> — ${f.metric}=${f.value}${p}<br>${f.interpretation}`;
      fwrap.appendChild(d);
    });
    $("researchProvider").textContent = insight.provider ? insight.provider.split(":")[0] : "";
    const hwrap = $("researchHyps");
    hwrap.innerHTML = "";
    (insight.hypotheses || []).forEach((h) => {
      const li = document.createElement("li");
      li.className = "hyp";
      li.innerHTML = `
        <div class="hyp-title"><span>${h.title}</span><span class="hyp-badge ${h.strength || 'low'}">${h.strength || ''}</span></div>
        <p><b>Evidence:</b> ${h.evidence || ''}</p>
        <p><b>Mechanism:</b> ${h.mechanism || ''}</p>
        <p class="hyp-next"><b>Next experiment:</b> ${h.next_experiment || ''}</p>`;
      hwrap.appendChild(li);
    });
    $("researchOut").hidden = false;
    toast("Cure-research complete — " + (insight.hypotheses || []).length + " hypotheses");
  } catch (e) {
    toast("Research failed: " + e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "🧪 Run cure-research experiments";
  }
}

/* ---------------- view toggle (scan / brain map) ---------------- */
function setView(view) {
  const brain = view === "brain";
  $("brainMap").hidden = !brain;
  $("brainLegend").hidden = !brain;
  $("imgBase").style.visibility = brain ? "hidden" : "visible";
  $("imgOverlay").style.visibility = brain ? "hidden" : "visible";
  $("heatmapCtl").style.opacity = brain ? 0.35 : 1;
  $("viewportLegend").hidden = brain || !$("imgOverlay").src;
  document.querySelectorAll("#viewSeg .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view));
}

/* ---------------- viewer controls ---------------- */
function applyOverlay() {
  const on = $("overlayToggle").checked;
  const op = parseInt($("opacity").value, 10);
  $("imgOverlay").style.opacity = on ? op / 100 : 0;
  $("opacityVal").textContent = op + "%";
}

/* ---------------- init ---------------- */
function init() {
  initTheme();
  startClock();
  refreshStatus();
  $("btnSample").addEventListener("click", loadSample);
  $("fileInput").addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0]));
  $("intakeForm").addEventListener("submit", runAnalysis);
  $("overlayToggle").addEventListener("change", applyOverlay);
  $("opacity").addEventListener("input", applyOverlay);
  $("btnInvert").addEventListener("click", () => $("imgBase").classList.toggle("invert"));
  document.querySelectorAll("#viewSeg .seg-btn").forEach((b) =>
    b.addEventListener("click", () => setView(b.dataset.view)));
  $("brainMap").addEventListener("mousemove", brainTip);
  $("brainMap").addEventListener("mouseleave", () => ($("brainTip").hidden = true));
  $("btnResearch").addEventListener("click", runResearch);

  // Kiosk / auto mode (?auto=1): for unattended hospital displays & demos.
  // Loads a sample scan and runs the full analysis on load, then optionally loops.
  const params = new URLSearchParams(location.search);
  if (params.has("auto")) {
    const loop = params.has("loop");
    const cycle = async () => {
      await loadSample();
      await runAnalysis();
      if (loop) setTimeout(cycle, 12000);
    };
    setTimeout(cycle, 1200);
  }
}
document.addEventListener("DOMContentLoaded", init);
