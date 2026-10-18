/* GeoCebada presentation. All model comparisons and predictions are loaded from
 * first-party CSV copies of the canonical repository artifacts.
 * No hidden reference yields, model fitting, or remote tracking are performed. */
"use strict";

const DATA = "./data/";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const fixed = (value, places = 6) => Number(value).toFixed(places);
const fmt = (value, places = 3) =>
  Number(value).toLocaleString("es-MX", { minimumFractionDigits: places, maximumFractionDigits: places });

// CSV parser with RFC 4180-style quoted fields, CRLF handling, and no third-party dependencies.
function parseCSV(source) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (char === '"') {
      if (quoted && source[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell); cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && source[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.some(value => value.length > 0)) rows.push(row);
      row = [];
    } else {
      cell += char;
    }
  }
  row.push(cell);
  if (row.some(value => value.length > 0)) rows.push(row);
  if (rows.length < 2) throw new Error("CSV sin registros.");
  const headers = rows.shift().map((s) => s.replace(/^\uFEFF/, "").trim());
  return rows.map((cells) => Object.fromEntries(headers.map((key, i) => [key, cells[i] ?? ""])));
}

async function loadCSV(name) {
  const response = await fetch(DATA + name, { cache: "no-store" });
  if (!response.ok) throw new Error("No se pudo descargar " + name + ": HTTP " + response.status);
  return parseCSV(await response.text());
}

function el(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function name07(model) {
  const replacements = {
    Local04D: "GeoRidge · seleccionado",
    Local07: "GeoRidge+",
    Prithvi07: "Prithvi-EO",
    catboost: "CatBoost"
  };
  if (replacements[model]) return replacements[model];
  const match = /^Local04D_plus_(Local07|Prithvi07|catboost)_w(0\.\d+)$/.exec(model);
  if (match) {
    const names = { Local07: "GeoRidge+", Prithvi07: "Prithvi-EO", catboost: "CatBoost" };
    return "GeoRidge + " + Math.round(Number(match[2]) * 100) + " % " + names[match[1]];
  }
  return model;
}

function name08(model) {
  return ({
    Local04D_frozen: "GeoRidge fijo · seleccionado",
    Local04D_nested: "GeoRidge reajustado",
    Local07_nested: "GeoRidge+ reajustado"
  })[model] || model;
}

function drawChart(containerId, data, labelFn, bestName, wantedNames) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.replaceChildren();
  const selected = wantedNames.map(name => data.find(x => x.model === name)).filter(Boolean);
  if (selected.length !== wantedNames.length) {
    container.append(el("p", "", "No se pudieron verificar todos los modelos de este banco."));
    return;
  }
  const max = Math.max(...selected.map(d => Number(d.rmse_pooled)));
  selected.forEach(row => {
    const best = row.model === bestName;
    const item = el("div", "bar-row" + (best ? " best" : ""));
    const label = el("span", "bar-label", labelFn(row.model).split(" · ")[0]);
    const track = el("div", "bar-track");
    const fill = el("span", "bar-fill");
    fill.style.width = (100 * Number(row.rmse_pooled) / max).toFixed(2) + "%";
    track.append(fill);
    item.append(label, track, el("span", "bar-value", fixed(row.rmse_pooled)));
    item.title = labelFn(row.model) + ": " + fixed(row.rmse_pooled) + " t/ha";
    container.append(item);
  });
}

function drawTable(id, rows, labelFn, bestName, filter = "all") {
  const body = document.getElementById(id);
  if (!body) return;
  body.replaceChildren();
  rows
    .filter(row => filter === "all" || (filter === "blend" ? row.model.includes("_plus_") : !row.model.includes("_plus_")))
    .sort((a, b) => Number(a.rmse_pooled) - Number(b.rmse_pooled))
    .forEach(row => {
      const tr = el("tr", row.model === bestName ? "selected-model" : "");
      [labelFn(row.model), fixed(row.rmse_pooled), fixed(row.rmse_mean_split), fixed(row.rmse_worst_split)].forEach(value => {
        tr.append(el("td", "", value));
      });
      body.append(tr);
    });
}

function drawBootstrap(rows) {
  const box = $("#bootstrap-results");
  if (!box) return;
  box.replaceChildren();
  rows.forEach(row => {
    const wrapper = el("div", "bootstrap-row");
    wrapper.append(el("span", "", name08(row.model)));
    const right = el("div");
    const track = el("div", "ci-track");
    const range = el("span", "ci-range");
    // Fixed 0.05 (t/ha)^2 plotting scale; the labels retain full source precision.
    range.style.left = Math.max(0, Number(row.ci95_low) / .05 * 100) + "%";
    range.style.width = Math.max(0, (Number(row.ci95_high) - Number(row.ci95_low)) / .05 * 100) + "%";
    track.append(range);
    right.append(
      el("small", "", "Δ MSE = +" + fixed(row.delta_mse) + " (t/ha)²"),
      track,
      el("small", "", "IC 95 % [" + fixed(row.ci95_low) + ", " + fixed(row.ci95_high) + "] (t/ha)²")
    );
    wrapper.append(right);
    box.append(wrapper);
  });
}

async function initResults() {
  try {
    const [a, b, ci] = await Promise.all([
      loadCSV("checkpoint07_summary.csv"),
      loadCSV("checkpoint08_summary.csv"),
      loadCSV("checkpoint08_bootstrap.csv")
    ]);
    // Guard against silently publishing a modified or incomplete ranking.
    if (a.length !== 13 || b.length !== 3 || ci.length !== 2 ||
        a.find(x => x.model === "Local04D")?.rmse_pooled?.slice(0, 8) !== "0.495741" ||
        b.find(x => x.model === "Local04D_frozen")?.rmse_pooled?.slice(0, 8) !== "0.481935") {
      throw new Error("Las tablas públicas no coinciden con el esquema verificado de 07/08.");
    }
    drawChart("chart07", a, name07, "Local04D", ["Local04D", "Local07", "Prithvi07", "catboost"]);
    drawChart("chart08", b, name08, "Local04D_frozen", ["Local04D_frozen", "Local04D_nested", "Local07_nested"]);
    drawTable("table07", a, name07, "Local04D");
    drawTable("table08", b, name08, "Local04D_frozen");
    drawBootstrap(ci);
    $$("[data-model-filter]").forEach(button => {
      button.addEventListener("click", () => {
        $$("[data-model-filter]").forEach(other => {
          other.classList.remove("active"); other.setAttribute("aria-pressed", "false");
        });
        button.classList.add("active"); button.setAttribute("aria-pressed", "true");
        drawTable("table07", a, name07, "Local04D", button.dataset.modelFilter);
      });
    });
  } catch (error) {
    console.error("Error al cargar métricas publicadas:", error);
    ["chart07", "chart08", "table07", "table08", "bootstrap-results"].forEach(id => {
      const node = document.getElementById(id);
      if (!node) return;
      const warning = el(id.startsWith("table") ? "tr" : "p", "load-error");
      if (id.startsWith("table")) {
        const cell = el("td", "", "No se pudieron cargar las métricas. Consulta el CSV original.");
        cell.colSpan = 4; warning.append(cell);
      } else warning.textContent = "No se pudieron cargar las métricas. Consulta el informe en GitHub.";
      node.replaceChildren(warning);
    });
  }
}

const BASE_LABELS = { GlobalMean: "Promedio global", StateMean: "Promedio por estado", PLS_C0: "Modelo global PLS", CatBoost_C1: "Modelo global CatBoost", GeoKNN_10: "Vecinos geográficos (k=10)", Graph04D: "Grafo espacial", Local04D: "GeoRidge · seleccionado" };
const TIERS = { high: ["Alto", 3], mid: ["Medio", 2], low: ["Bajo", 1] };

const FAMILY_STEPS = [["DummyMean", "Sin variables (promedio)"], ["A0_geometry_admin", "Ubicación y datos administrativos"], ["A1_static_environment", "+ suelo, relieve y elevación"], ["A2_static_plus_siap_history", "+ historial municipal SIAP"], ["A3_clean_plus_climate", "+ clima histórico"], ["A4_clean_plus_basic_history", "+ satélite histórico"], ["A6_competition_remote_weather", "+ satélite, clima y agua 2025"], ["A7_competition_full", "+ SIAP 2025"]];

async function initFamilies() {
  const body = $("#table-families");
  if (!body) return;
  try {
    const rows = await loadCSV("checkpoint02_baseline_summary.csv");
    const get = (ablation, protocol, model) => {
      const row = rows.find(r => r.protocol === protocol && r.model === model && (model === "DummyMean" ? r.ablation === "A0_geometry_admin" : r.ablation === ablation));
      if (!row) throw new Error("Falta " + ablation + " " + protocol + " " + model);
      return { rmse: Number(row.rmse_mean), n: Number(row.n_features) };
    };
    if (fixed(get("A0_geometry_admin", "fold_state_stratified", "ExtraTrees").rmse, 4) !== "0.5095") throw new Error("Tabla distinta del esquema verificado.");
    const cols = [["fold_state_stratified", "ExtraTrees"], ["fold_state_stratified", "Ridge"], ["fold_municipality_grouped", "ExtraTrees"], ["fold_municipality_grouped", "Ridge"]];
    const data = FAMILY_STEPS.map(([ablation, label]) => ({ label, n: ablation === "DummyMean" ? 0 : get(ablation, cols[0][0], "ExtraTrees").n, vals: cols.map(([p, m]) => get(ablation, p, ablation === "DummyMean" ? "DummyMean" : m).rmse) }));
    const best = cols.map((_, c) => Math.min(...data.slice(1).map(d => d.vals[c])));
    body.replaceChildren();
    data.forEach(d => {
      const tr = el("tr");
      tr.append(el("td", "", d.label), el("td", "", d.n ? String(d.n) : "—"));
      d.vals.forEach((v, c) => tr.append(el("td", d.label.startsWith("Sin") ? "" : (v === best[c] ? "best" : ""), fixed(v, 3))));
      body.append(tr);
    });
  } catch (error) {
    console.error("Error al cargar ablaciones:", error);
    const cell = el("td", "load-error", "No se pudieron cargar las ablaciones."); cell.colSpan = 6;
    body.replaceChildren(el("tr", "", "")); body.firstChild.append(cell);
  }
}

const VAR_NAMES = { siap_hist__yield_trend_recent: "Tendencia reciente del rendimiento municipal (SIAP)", siap_hist__yield_mean_recent: "Rendimiento municipal reciente medio (SIAP)", agro_thermal__2025__gdd_proxy_base5p0c: "Grados-día 2025 (base 5 °C)", clim_official_2025__prec__season_sum: "Precipitación de la temporada 2025", soilgrids__soc__depth_weighted_0_30cm: "Carbono orgánico del suelo (0–30 cm)" };
const FAMILY_NAMES = { water_productivity: "Productividad del agua", thermal: "Térmica", water_timing: "Momento de la lluvia", cross_domain: "Interacciones cultivo–agua–clima", soil_interaction: "Suelo × clima", soil_profile: "Perfil del suelo", nonlinear_basis: "Transformaciones no lineales", sensor_agreement: "Concordancia entre sensores", phenology: "Fenología (curva del cultivo)", phenology_anomaly: "Anomalías fenológicas vs. histórico", all_agronomic: "Todas juntas" };

async function initAnnexE() {
  const tailBody = $("#table-tail"), comboBody = $("#table-combos");
  if (!tailBody || !comboBody) return;
  try {
    const [tail, expr, risk] = await Promise.all([loadCSV("checkpoint06_feature_family_tail_importance.csv"), loadCSV("checkpoint03b_expression_stability.csv"), loadCSV("checkpoint06_feature_family_error_risk.csv")]);
    const rowOf = (family, task) => tail.find(r => r.family === family && r.task === task);
    const auc = (family, task) => Number(rowOf(family, task)?.mean_split_roc_auc);
    if (fixed(auc("water_productivity", "low_vs_rest"), 3) !== "0.889") throw new Error("Tabla de extremos distinta del esquema verificado.");
    const families = [...new Set(tail.map(r => r.family))].filter(f => f !== "all_agronomic").sort((a, b) => auc(b, "low_vs_rest") - auc(a, "low_vs_rest"));
    const bestLow = Math.max(...families.map(f => auc(f, "low_vs_rest"))), bestHigh = Math.max(...families.map(f => auc(f, "high_vs_rest")));
    tailBody.replaceChildren();
    [...families, "all_agronomic"].forEach(f => {
      const low = auc(f, "low_vs_rest"), high = auc(f, "high_vs_rest"), tr = el("tr");
      tr.append(el("td", "", FAMILY_NAMES[f] || f), el("td", "", rowOf(f, "low_vs_rest").n_features),
        el("td", low === bestLow ? "best" : "", fixed(low, 3)), el("td", high === bestHigh ? "best" : "", fixed(high, 3)));
      tailBody.append(tr);
    });
    const bestR2 = Math.max(...risk.filter(r => r.method === "Local04D").map(r => Number(r.risk_r2_log)));
    $("#risk-note").textContent = "Con estas mismas familias no fue posible anticipar en qué parcelas se equivocará más el modelo: el mejor R² al predecir el riesgo de error fue " + fmt(bestR2, 2) + " (un valor ≤ 0 no supera predecir el promedio).";

    const pairs = new Map();
    expr.forEach(r => {
      const key = [r.left, r.right].sort().join("|");
      const p = pairs.get(key) || { left: r.left, right: r.right, state: 0, grouped: 0, rho: 0 };
      const folds = Number(r.selected_folds);
      if (r.protocol === "fold_state_stratified") p.state = Math.max(p.state, folds); else p.grouped = Math.max(p.grouped, folds);
      p.rho = Math.max(p.rho, Number(r.mean_abs_spearman_train));
      pairs.set(key, p);
    });
    const motifs = Array.from(pairs.values()).filter(p => p.state >= 3 && p.grouped >= 3).sort((a, b) => (b.state + b.grouped) - (a.state + a.grouped) || b.rho - a.rho);
    if (motifs.length !== 4) throw new Error("Las combinaciones recurrentes no coinciden con el esquema verificado.");
    comboBody.replaceChildren();
    motifs.forEach(p => {
      const tr = el("tr");
      tr.append(el("td", "", (VAR_NAMES[p.left] || p.left) + " ↔ " + (VAR_NAMES[p.right] || p.right)), el("td", "", p.state + " / 5"), el("td", "", p.grouped + " / 5"), el("td", "", fixed(p.rho, 2)));
      comboBody.append(tr);
    });
  } catch (error) {
    console.error("Error al cargar variables del anexo E:", error);
    [[tailBody, 4], [comboBody, 4]].forEach(([body, span]) => {
      const cell = el("td", "load-error", "No se pudieron cargar estas tablas. Consulta los CSV del repositorio."); cell.colSpan = span;
      const tr = el("tr"); tr.append(cell); body.replaceChildren(tr);
    });
  }
}

async function initBaselines() {
  const box = $("#chart-baselines");
  if (!box) return;
  try {
    const [b04, b05] = await Promise.all([loadCSV("checkpoint04b_protocol_summary.csv"), loadCSV("checkpoint05_protocol_summary.csv")]);
    const source = { GlobalMean: b04, StateMean: b04, GeoKNN_10: b04, PLS_C0: b05, CatBoost_C1: b05, Graph04D: b05, Local04D: b05 };
    const rows = Object.keys(BASE_LABELS).map(model => {
      const row = source[model].find(r => r.family === "target_matched" && r.method === model);
      if (!row || !Number.isFinite(Number(row.rmse_mean))) throw new Error("Falta " + model);
      return { model, rmse_pooled: row.rmse_mean };
    });
    const value = model => Number(rows.find(r => r.model === model).rmse_pooled);
    if (fixed(value("GlobalMean")) !== "0.849973" || fixed(value("Local04D")) !== "0.487845") throw new Error("Tablas de referencia distintas del esquema verificado.");
    drawChart("chart-baselines", rows, model => BASE_LABELS[model], "Local04D", Object.keys(BASE_LABELS));
    const best = value("Local04D");
    const gains = [["StateMean", "frente al promedio por estado"], ["CatBoost_C1", "frente a un modelo global (CatBoost)"], ["GeoKNN_10", "frente a vecinos geográficos simples"]];
    $("#baseline-gains").replaceChildren(...gains.map(([model, text]) => {
      const item = el("div", "gain");
      item.append(el("strong", "", "−" + fmt(100 * (1 - best / value(model)), 1) + " %"), el("span", "", text));
      return item;
    }));
  } catch (error) {
    console.error("Error al cargar referencias:", error);
    box.replaceChildren(el("p", "load-error", "No se pudieron cargar las referencias de comparación."));
  }
}

function buildSupport(rows, records) {
  if (!rows) return null;
  const map = new Map(rows.map(r => [String(r.ID_POLIGONO).trim(), { tier: r.x_support_tier, diff: Number(r.local_graph_abs_disagreement), y: Number(r.predicted_final) }]));
  const bad = map.size !== 59 || records.some(r => { const s = map.get(r.id); return !s || !TIERS[s.tier] || !Number.isFinite(s.diff) || Math.abs(s.y - r.y) > 1e-6; });
  return bad ? null : map;
}

function initPredictions(rows, support) {
  const selector = $("#parcel-select"), input = $("#parcel-search");
  const value = $("#prediction-value"), id = $("#prediction-id"), status = $("#prediction-status");
  const records = rows.map(x => ({ id: String(x.ID_POLIGONO).trim(), y: Number(x.RENDIMIENTO_T_HA) }))
    .sort((a, b) => a.id.localeCompare(b.id, "es"));
  const unique = new Set(records.map(x => x.id));
  if (records.length !== 59 || unique.size !== 59 || records.some(x => !/^AGC_\d+$/.test(x.id) || !Number.isFinite(x.y) || x.y < 0)) {
    throw new Error("Archivo de predicciones distinto del contrato de 59 consultas.");
  }
  const sup = buildSupport(support, records);
  const showSupport = id => {
    if (!sup) return;
    const s = sup.get(id), [label, level] = TIERS[s.tier];
    $("#support-label").textContent = label;
    $$("#support-meter i").forEach((bar, i) => bar.classList.toggle("on", i < level));
    $("#support-diff").textContent = fmt(s.diff, 2) + " t/ha";
  };
  if (sup) {
    const n = tier => Array.from(sup.values()).filter(s => s.tier === tier).length;
    $("#support-box").hidden = false;
    $("#support-summary").hidden = false;
    $("#support-summary").textContent = "Soporte de datos de las 59 consultas: Alto " + n("high") + " · Medio " + n("mid") + " · Bajo " + n("low");
  }
  selector.replaceChildren();
  records.forEach(row => {
    const option = el("option", "", row.id);
    option.value = row.id; selector.append(option);
  });

  const count = 12;
  const min = Math.min(...records.map(r => r.y)), max = Math.max(...records.map(r => r.y));
  const bins = Array(count).fill(0), width = (max - min) / count || 1;
  records.forEach(record => { bins[Math.min(count - 1, Math.floor((record.y - min) / width))]++; });
  const histogram = $("#prediction-histogram");
  histogram.replaceChildren();
  const highest = Math.max(...bins);
  bins.forEach((countValue, index) => {
    const bar = el("span", "histogram-bar");
    bar.style.height = (100 * countValue / highest).toFixed(2) + "%";
    bar.title = "Intervalo " + (index + 1) + ": " + countValue + " parcelas";
    histogram.append(bar);
  });
  $("#hist-min").textContent = fmt(min, 2);
  $("#hist-max").textContent = fmt(max, 2);

  const update = (record, syncText = true) => {
    if (!record) return;
    selector.value = record.id;
    if (syncText) input.value = record.id;
    value.textContent = fmt(record.y, 3);
    id.textContent = "ID_POLIGONO: " + record.id;
    status.textContent = "Estimación calculada y congelada en la etapa 5. El rendimiento real de esta parcela no está disponible.";
    showSupport(record.id);
    const active = Math.min(count - 1, Math.floor((record.y - min) / width));
    Array.from(histogram.children).forEach((bar, index) => {
      bar.style.opacity = index === active ? "1" : ".44";
      bar.style.filter = index === active ? "brightness(1.25)" : "";
    });
  };

  selector.addEventListener("change", () => update(records.find(r => r.id === selector.value)));
  input.addEventListener("input", () => {
    const query = input.value.trim().toUpperCase();
    const record = records.find(r => r.id === query) || records.find(r => r.id.startsWith(query));
    if (record) update(record, false);
    else status.textContent = "Identificador no encontrado entre las 59 consultas de este archivo.";
  });
  update(records[0], false);
  input.value = "";
  input.placeholder = "Ej. " + records[0].id;
}

async function loadPredictions() {
  try {
    const [rows, support] = await Promise.all([loadCSV("predictions.csv"), loadCSV("final_prediction_diagnostics.csv").catch(() => null)]);
    initPredictions(rows, support);
  } catch (error) {
    console.error("Error al cargar estimaciones:", error);
    $("#prediction-status").textContent = "No se pueden verificar las 59 estimaciones. Descarga el CSV desde el repositorio.";
    $("#parcel-select").replaceChildren(el("option", "", "No disponible"));
    $("#parcel-select").disabled = true;
    $("#parcel-search").disabled = true;
  }
}

function initChrome() {
  const progress = $("#scroll-progress");
  const scroll = () => {
    const total = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (total > 0 ? Math.max(0, Math.min(100, window.scrollY / total * 100)) : 0) + "%";
  };
  window.addEventListener("scroll", scroll, { passive: true });
  scroll();

  const menu = $("#menu-toggle"), links = $("#nav-links");
  menu.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    menu.setAttribute("aria-expanded", String(open));
    menu.setAttribute("aria-label", open ? "Cerrar navegación" : "Mostrar navegación");
  });
  $$(".nav-links a").forEach(link => link.addEventListener("click", () => {
    links.classList.remove("open");
    menu.setAttribute("aria-expanded", "false");
  }));
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add("visible"); observer.unobserve(entry.target); }
    }), { threshold: .10, rootMargin: "0px 0px -35px 0px" });
    $$("[data-reveal]").forEach(node => observer.observe(node));
  } else {
    $$("[data-reveal]").forEach(node => node.classList.add("visible"));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initChrome();
  initResults();
  initBaselines();
  initFamilies();
  initAnnexE();
  loadPredictions();
});
