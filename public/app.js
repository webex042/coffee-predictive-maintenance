/* =========================================================================
   Coffee Line — Predictive Maintenance dashboard (app.js)
   -------------------------------------------------------------------------
   Vanilla JS, no framework, no build step. It:
     1. builds the sensor input form per machine type,
     2. POSTs a reading to the real serverless model at /api/predict,
     3. renders the model's REAL output (probability, health, risk,
        recommendation) as gauges + a deviation chart.

   The dashboard degrades gracefully: if Chart.js (loaded from a CDN) is
   unavailable, every visual falls back to a plain numeric/table readout.
   ========================================================================= */
"use strict";

/* -------------------------------------------------------------------------
   Machine baselines — mirror of config.py MACHINE_BASELINES.
   DUPLICATED ON PURPOSE: the browser can't import the Python config, and
   these numbers only drive the *client-side convenience* of pre-filling a
   healthy reading per machine type + the input placeholders. They are NOT
   used for scoring — every prediction is computed server-side by the trained
   model against config.py. If you change config.py, update this to match.
   ------------------------------------------------------------------------- */
const MACHINE_BASELINES = {
  Roaster:    { temperature: 210, vibration: 2.5, pressure: 1.5, rpm: 30,   motor_current: 45 },
  Grinder:    { temperature: 55,  vibration: 4.0, pressure: 1.2, rpm: 1450, motor_current: 30 },
  Extraction: { temperature: 95,  vibration: 3.0, pressure: 9.0, rpm: 120,  motor_current: 38 },
  SprayDryer: { temperature: 180, vibration: 2.0, pressure: 3.5, rpm: 8000, motor_current: 60 },
  Conveyor:   { temperature: 35,  vibration: 1.5, pressure: 1.0, rpm: 200,  motor_current: 15 },
  Packaging:  { temperature: 40,  vibration: 2.2, pressure: 2.0, rpm: 600,  motor_current: 20 },
};

/* One-line "what is normal" descriptor per type, shown under the selector. */
const MACHINE_NOTES = {
  Roaster:    "Runs hot and slow — high roast temperature, low rpm.",
  Grinder:    "Spins fast and runs cool — high rpm, moderate vibration.",
  Extraction: "High-pressure brew stage — warm, high pressure, low rpm.",
  SprayDryer: "Very hot, very high rpm atomiser — powder drying stage.",
  Conveyor:   "Cool and steady transport — low everything.",
  Packaging:  "Cool, mid-speed fill/seal line.",
};

/* Default whole-machine context (not type-specific). A healthy machine at a
   typical duty point: ~70% load, some hours on the clock, freshly maintained. */
const CONTEXT_DEFAULTS = {
  operating_hours: 2000,
  machine_load: 70,            // percent (the API converts to a fraction)
  hours_since_maintenance: 0,
  maintenance_count: 0,
  previous_failures: 0,
};

/* Field definitions. `key` matches the JSON the API expects. `advanced: true`
   fields live in the collapsible section and default to healthy values. */
const REQUIRED_FIELDS = [
  { key: "temperature",      label: "Temperature",      unit: "°C",   step: "0.1", min: "-50" },
  { key: "vibration",        label: "Vibration",        unit: "mm/s", step: "0.1", min: "0" },
  { key: "pressure",         label: "Pressure",         unit: "bar",  step: "0.1", min: "0" },
  { key: "rotational_speed", label: "Rotational speed", unit: "rpm",  step: "1",   min: "0" },
  { key: "operating_hours",  label: "Operating hours",  unit: "h",    step: "1",   min: "0" },
  { key: "machine_load",     label: "Machine load",     unit: "%",    step: "1",   min: "0", max: "100" },
];

const ADVANCED_FIELDS = [
  { key: "motor_current",           label: "Motor current",         unit: "A", step: "0.1", min: "0" },
  { key: "hours_since_maintenance", label: "Hours since maint.",    unit: "h", step: "1",   min: "0" },
  { key: "maintenance_count",       label: "Maintenance count",     unit: "",  step: "1",   min: "0" },
  { key: "previous_failures",       label: "Previous failures",     unit: "",  step: "1",   min: "0" },
];

/* Sensors shown on the deviation chart, mapped to the API's z-score keys. */
const DEVIATION_SENSORS = [
  { key: "temperature",      label: "Temp" },
  { key: "vibration",        label: "Vibration" },
  { key: "pressure",         label: "Pressure" },
  { key: "rotational_speed", label: "Speed" },
  { key: "motor_current",    label: "Current" },
];

const RISK_COLORS = {
  Low:      getVar("--risk-low",      "#2ea043"),
  Medium:   getVar("--risk-medium",   "#d4a017"),
  High:     getVar("--risk-high",     "#e8801a"),
  Critical: getVar("--risk-critical", "#e5484d"),
};

function getVar(name, fallback) {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (_) { return fallback; }
}

/* Health-score -> colour, matching config.HEALTH_BANDS ordering. */
function healthColor(score) {
  if (score >= 80) return RISK_COLORS.Low;
  if (score >= 60) return RISK_COLORS.Medium;
  if (score >= 40) return RISK_COLORS.High;
  return RISK_COLORS.Critical;
}

/* ---- DOM references (resolved on DOMContentLoaded) --------------------- */
const el = {};
let charts = { prob: null, health: null, sensor: null };
const hasChart = () => typeof window.Chart !== "undefined";

/* ---- Init ------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  el.form        = document.getElementById("predict-form");
  el.machineType = document.getElementById("machine_type");
  el.overview    = document.getElementById("machine-overview");
  el.sensorGrid  = document.querySelector(".sensor-grid");
  el.advGrid     = document.getElementById("advanced-grid");
  el.predictBtn  = document.getElementById("predict-btn");
  el.resetBtn    = document.getElementById("reset-btn");
  el.formError   = document.getElementById("form-error");
  el.results     = document.getElementById("results");
  el.emptyState  = document.getElementById("empty-state");
  el.resultBox   = document.getElementById("result-content");
  el.resultError = document.getElementById("result-error");
  el.riskBanner  = document.getElementById("risk-banner");
  el.riskValue   = document.getElementById("risk-value");
  el.probCenter  = document.getElementById("prob-center");
  el.probNote    = document.getElementById("prob-note");
  el.healthCenter= document.getElementById("health-center");
  el.healthNote  = document.getElementById("health-note");
  el.recBox      = document.getElementById("recommendation");
  el.recText     = document.getElementById("rec-text");
  el.chip        = document.getElementById("model-chip");
  el.chipText    = document.getElementById("model-chip-text");

  populateMachineTypes();
  buildFields();
  applyDefaults(el.machineType.value);

  el.machineType.addEventListener("change", () => applyDefaults(el.machineType.value));
  el.form.addEventListener("submit", onSubmit);
  el.resetBtn.addEventListener("click", () => {
    applyDefaults(el.machineType.value);
    clearError();
  });
});

function populateMachineTypes() {
  const types = Object.keys(MACHINE_BASELINES);
  el.machineType.innerHTML = types
    .map((t) => `<option value="${t}"${t === "Grinder" ? " selected" : ""}>${t}</option>`)
    .join("");
}

/* Build a single labelled numeric input with an optional unit suffix. */
function fieldMarkup(f) {
  const suffix = f.unit
    ? `<span class="input-unit__suffix">${f.unit}</span>` : "";
  const maxAttr = f.max ? ` max="${f.max}"` : "";
  return `
    <div class="field sensor-field">
      <label for="f_${f.key}">${f.label}</label>
      <div class="input-unit">
        <input type="number" id="f_${f.key}" name="${f.key}"
               inputmode="decimal" step="${f.step}" min="${f.min}"${maxAttr}
               required aria-describedby="f_${f.key}_u" />
        ${suffix}
      </div>
    </div>`;
}

function buildFields() {
  el.sensorGrid.insertAdjacentHTML(
    "beforeend", REQUIRED_FIELDS.map(fieldMarkup).join(""));
  el.advGrid.innerHTML = ADVANCED_FIELDS.map(fieldMarkup).join("");
}

/* Fill every input with a healthy reading for the chosen machine type. */
function applyDefaults(type) {
  const base = MACHINE_BASELINES[type] || MACHINE_BASELINES.Grinder;
  const values = {
    temperature: base.temperature,
    vibration: base.vibration,
    pressure: base.pressure,
    rotational_speed: base.rpm,
    motor_current: base.motor_current,
    ...CONTEXT_DEFAULTS,
  };
  for (const [key, val] of Object.entries(values)) {
    const input = document.getElementById(`f_${key}`);
    if (input) input.value = val;
  }
  el.overview.textContent = MACHINE_NOTES[type] || "";
}

/* Gather the form into the JSON payload the API expects. Returns
   { payload } on success or { error } with a user-facing message. */
function readForm() {
  const payload = { machine_type: el.machineType.value };
  const all = [...REQUIRED_FIELDS, ...ADVANCED_FIELDS];
  for (const f of all) {
    const input = document.getElementById(`f_${f.key}`);
    const raw = input ? input.value.trim() : "";
    const required = REQUIRED_FIELDS.some((r) => r.key === f.key);
    if (raw === "") {
      if (required) return { error: `Please enter a value for “${f.label}”.` };
      continue; // let the API apply its documented default
    }
    const num = Number(raw);
    if (!Number.isFinite(num)) {
      return { error: `“${f.label}” must be a number.` };
    }
    if (f.max !== undefined && num > Number(f.max)) {
      return { error: `“${f.label}” must be ${f.max} or less.` };
    }
    if (f.min !== undefined && num < Number(f.min)) {
      return { error: `“${f.label}” must be ${f.min} or more.` };
    }
    payload[f.key] = num;
  }
  return { payload };
}

/* ---- Submit / fetch --------------------------------------------------- */
async function onSubmit(event) {
  event.preventDefault();
  clearError();

  const { payload, error } = readForm();
  if (error) { showFormError(error); return; }

  setBusy(true);
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }

    if (!res.ok) {
      const msg = (data && data.error)
        ? data.error
        : `The prediction service returned an error (HTTP ${res.status}).`;
      showResultError(msg);
      return;
    }
    renderResult(data);
  } catch (_) {
    showResultError(
      "Could not reach the prediction service. Check that the server is " +
      "running and try again.");
  } finally {
    setBusy(false);
  }
}

function setBusy(on) {
  el.predictBtn.disabled = on;
  el.predictBtn.textContent = on ? "Scoring…" : "Predict failure risk";
  el.results.classList.toggle("is-loading", on);
}

/* ---- Render a successful prediction ----------------------------------- */
function renderResult(data) {
  const details = data.details || {};
  const risk = data.risk_level || "Low";
  const prob = Number(data.failure_probability) || 0;   // 0..1
  const health = Number(data.health_score) || 0;        // 0..100
  const probPct = Math.round(prob * 1000) / 10;         // one decimal

  el.emptyState.hidden = true;
  el.resultError.hidden = true;
  el.resultBox.hidden = false;

  // Risk banner
  el.riskValue.textContent = risk;
  el.riskBanner.className =
    "risk-banner risk-banner--" + risk.toLowerCase();

  // Gauge centre labels
  el.probCenter.textContent = probPct + "%";
  el.healthCenter.textContent = String(health);

  // Notes under gauges (grounded in the model's own numbers)
  const thr = details.alert_threshold;
  el.probNote.textContent = (typeof thr === "number")
    ? (details.alert
        ? `At/above the model's ${Math.round(thr * 100)}% alert threshold.`
        : `Below the model's ${Math.round(thr * 100)}% alert threshold.`)
    : "";
  el.healthNote.textContent = details.health_band
    ? `Condition: ${details.health_band}` : "";

  // Recommendation
  el.recText.textContent = data.recommendation || "—";
  el.recBox.classList.toggle("recommendation--urgent", !!details.urgent);

  // Model chip (populated from the real bundle metrics)
  const model = details.model;
  if (model && model.name) {
    el.chipText.textContent =
      `${model.name} · recall ${Number(model.recall).toFixed(2)}`;
    el.chip.hidden = false;
  }

  // Visuals
  drawProbGauge(probPct, RISK_COLORS[risk] || RISK_COLORS.Low);
  drawHealthGauge(health, healthColor(health));
  drawSensorChart(details.sensor_zscores || {});
}

/* ---- Gauges (semicircular doughnuts, with numeric fallback) ----------- */
function gaugeConfig(value, max, color) {
  return {
    type: "doughnut",
    data: {
      datasets: [{
        data: [value, Math.max(0, max - value)],
        backgroundColor: [color, "rgba(255,255,255,.07)"],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "72%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 500 },
    },
  };
}

function drawGauge(which, canvasId, value, max, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  if (!hasChart()) {
    // Numeric fallback: hide the canvas, let the centre label stand alone.
    canvas.closest(".gauge-wrap")?.classList.add("is-fallback");
    canvas.style.display = "none";
    return;
  }
  if (charts[which]) { charts[which].destroy(); }
  charts[which] = new window.Chart(canvas, gaugeConfig(value, max, color));
}

function drawProbGauge(pct, color) { drawGauge("prob", "prob-gauge", pct, 100, color); }
function drawHealthGauge(score, color) { drawGauge("health", "health-gauge", score, 100, color); }

/* ---- Sensor deviation bar chart (with table fallback) ----------------- */
function zColor(z) {
  const a = Math.abs(z);
  if (a >= 2.5) return RISK_COLORS.Critical;
  if (a >= 1.5) return RISK_COLORS.High;
  if (a >= 1.0) return RISK_COLORS.Medium;
  return RISK_COLORS.Low;
}

function drawSensorChart(zscores) {
  const labels = DEVIATION_SENSORS.map((s) => s.label);
  const values = DEVIATION_SENSORS.map((s) => Number(zscores[s.key]) || 0);
  const canvas = document.getElementById("sensor-chart");
  if (!canvas) return;

  if (!hasChart()) { renderSensorTable(zscores); return; }
  if (charts.sensor) { charts.sensor.destroy(); }

  charts.sensor = new window.Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(zColor),
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          title: { display: true, text: "σ from normal", color: "#8b98a9" },
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: { color: "#8b98a9" },
        },
        x: { grid: { display: false }, ticks: { color: "#c9d3de" } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => `${c.parsed.y.toFixed(2)} σ from healthy`,
          },
        },
      },
    },
  });
}

/* Accessible fallback if Chart.js failed to load from the CDN. */
function renderSensorTable(zscores) {
  const rows = DEVIATION_SENSORS.map((s) => {
    const z = Number(zscores[s.key]) || 0;
    return `<tr><th scope="row">${s.label}</th>` +
           `<td>${z >= 0 ? "+" : ""}${z.toFixed(2)} σ</td></tr>`;
  }).join("");
  const canvas = document.getElementById("sensor-chart");
  const table =
    `<table class="sensor-fallback"><thead><tr><th>Sensor</th>` +
    `<th>Deviation</th></tr></thead><tbody>${rows}</tbody></table>`;
  if (canvas) canvas.insertAdjacentHTML("afterend", table);
  if (canvas) canvas.style.display = "none";
}

/* ---- Error / state helpers -------------------------------------------- */
function showFormError(msg) {
  el.formError.textContent = msg;
  el.formError.hidden = false;
}

function showResultError(msg) {
  el.emptyState.hidden = true;
  el.resultBox.hidden = true;
  el.resultError.textContent = msg;
  el.resultError.hidden = false;
}

function clearError() {
  el.formError.hidden = true;
  el.formError.textContent = "";
  el.resultError.hidden = true;
  el.resultError.textContent = "";
}
