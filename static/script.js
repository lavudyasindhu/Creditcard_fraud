// ---------- Element shortcuts ----------
const $ = (id) => document.getElementById(id);
const slider = $("threshold");

let currentSampleId = null;   // transaction shown in the prediction card
let previous = null;          // metrics at the previous threshold (for arrows)
let chart = null;
let requestCounter = 0;       // ignore out-of-date responses while dragging

const pct = (v) => (v * 100).toFixed(1) + "%";
const fmtT = (v) => Number(v).toFixed(2);

// ---------- Talk to Flask ----------
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Server error: " + res.status);
  return res.json();
}

// Recalculate everything for the current slider value
async function update() {
  const t = slider.value;
  const myRequest = ++requestCounter;

  $("threshold-value").textContent = fmtT(t);
  $("rule-t").textContent = fmtT(t);
  highlightPreset(t);

  let url = `/api/evaluate?threshold=${t}`;
  if (currentSampleId !== null) url += `&sample_id=${currentSampleId}`;

  try {
    const data = await getJSON(url);
    if (myRequest !== requestCounter) return;   // a newer request exists
    showMetrics(data);
    if (data.transaction) showTransaction(data.transaction, data.threshold);
    updateChartLine(data.threshold);
  } catch (err) {
    $("verdict-label").textContent = "Could not reach the server";
    $("verdict-sub").textContent = "Make sure app.py is running.";
  }
}

// ---------- Metrics + confusion matrix ----------
function showMetrics(data) {
  const m = data.metrics;
  $("total").textContent = data.total.toLocaleString();
  $("total-fraud").textContent = data.total_fraud.toLocaleString();

  ["accuracy", "recall", "precision", "f1"].forEach((k) => {
    $("m-" + k).textContent = pct(m[k]);
    showDelta(k, m[k]);
  });

  ["tn", "fp", "fn", "tp"].forEach((k) => {
    const el = $(k);
    const newText = m[k].toLocaleString();
    if (el.textContent !== newText) {
      el.textContent = newText;
      el.classList.remove("flash");
      void el.offsetWidth;        // restart the small animation
      el.classList.add("flash");
    }
  });

  previous = m;
}

function showDelta(key, value) {
  const el = $("dl-" + key);
  el.className = "m-delta";
  if (!previous) { el.textContent = ""; return; }
  const diff = (value - previous[key]) * 100;
  if (Math.abs(diff) < 0.05) { el.textContent = "no change"; return; }
  el.textContent = (diff > 0 ? "\u25B2 +" : "\u25BC ") + diff.toFixed(1) + " pts";
  el.classList.add(diff > 0 ? "up" : "down");
}

// ---------- One transaction's prediction ----------
function showTransaction(tx, threshold) {
  currentSampleId = tx.id;
  const isFraud = tx.prediction === "Fraudulent";

  $("verdict").className = "verdict " + (isFraud ? "fraud" : "legit");
  $("verdict-label").textContent = tx.prediction;
  $("verdict-sub").textContent =
    `Fraud probability ${tx.probability.toFixed(3)} is ` +
    (isFraud ? "at or above" : "below") + ` the threshold ${fmtT(threshold)}`;

  $("prob-fill").style.width = (tx.probability * 100) + "%";
  $("prob-mark").style.left = (threshold * 100) + "%";
  $("prob-value").textContent = tx.probability.toFixed(3);

  $("d-amount").textContent = "$" + tx.amount.toFixed(2);
  $("d-actual").textContent = tx.actual;
  const correct = tx.actual === tx.prediction;
  $("d-correct").textContent = correct ? "Correct" : "Wrong";
  $("d-correct").className = correct ? "good" : "bad";
}

async function loadSample(kind) {
  const data = await getJSON(`/api/sample?kind=${kind}&threshold=${slider.value}`);
  showTransaction(data, slider.value);
}

// ---------- Graph ----------
const thresholdLine = {
  id: "thresholdLine",
  afterDraw(c) {
    if (c.$threshold === undefined) return;
    const x = c.scales.x.getPixelForValue(Math.round(c.$threshold * 100));
    const { top, bottom } = c.chartArea;
    const ctx = c.ctx;
    ctx.save();
    ctx.strokeStyle = "#14213d";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.restore();
  },
};

async function buildChart() {
  const points = await getJSON("/api/curve");
  const labels = points.map((p) => fmtT(p.threshold));
  const line = (label, key, color, dash) => ({
    label, data: points.map((p) => +(p[key] * 100).toFixed(2)),
    borderColor: color, backgroundColor: color, borderWidth: 3,
    borderDash: dash || [], pointRadius: 0, tension: 0.2,
  });

  chart = new Chart($("chart"), {
    type: "line",
    data: { labels, datasets: [
      line("Recall", "recall", "#c9202f"),
      line("Accuracy", "accuracy", "#14213d"),
      line("Precision", "precision", "#0f8b6d", [2, 5]),
    ]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Threshold" }, ticks: { maxTicksLimit: 11 } },
        y: { min: 0, max: 100, title: { display: true, text: "Score (%)" } },
      },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y}%` } } },
    },
    plugins: [thresholdLine],
  });
  updateChartLine(slider.value);
}

function updateChartLine(t) {
  if (!chart) return;
  chart.$threshold = parseFloat(t);
  chart.draw();
}

// ---------- Presets and events ----------
function highlightPreset(t) {
  document.querySelectorAll(".presets button").forEach((b) =>
    b.classList.toggle("active", Math.abs(parseFloat(b.dataset.t) - t) < 0.001));
}

slider.addEventListener("input", update);   // updates while dragging

document.querySelectorAll(".presets button").forEach((b) =>
  b.addEventListener("click", () => { slider.value = b.dataset.t; update(); }));

$("btn-any").addEventListener("click", () => loadSample("any"));
$("btn-fraud").addEventListener("click", () => loadSample("fraud"));
$("btn-legit").addEventListener("click", () => loadSample("legit"));

// ---------- Start ----------
(async function init() {
  await loadSample("fraud");   // start with a fraud case so the threshold effect is easy to see
  await update();
  await buildChart();
})();
