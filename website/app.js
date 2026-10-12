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
    Local04D: "Local04D · seleccionado",
    Local07: "Local07",
    Prithvi07: "Prithvi07",
    catboost: "CatBoost07"
  };
  if (replacements[model]) return replacements[model];
  const match = /^Local04D_plus_(Local07|Prithvi07|catboost)_w(0\.\d+)$/.exec(model);
  if (match) {
    const names = { Local07: "Local07", Prithvi07: "Prithvi07", catboost: "CatBoost07" };
    return "Local04D + " + Math.round(Number(match[2]) * 100) + " % " + names[match[1]];
  }
  return model;
}

function name08(model) {
  return ({
    Local04D_frozen: "Local04D fijo · seleccionado",
    Local04D_nested: "Local04D con ajuste interno",
    Local07_nested: "Local07 con ajuste interno"
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
    wrapper.append(el("span", "", name08(row.model).split(" con ")[0] + " (" + (row.model.includes("07") ? "nuevo" : "reajustado") + ")"));
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

function initPredictions(rows) {
  const selector = $("#parcel-select"), input = $("#parcel-search");
  const value = $("#prediction-value"), id = $("#prediction-id"), status = $("#prediction-status");
  const records = rows.map(x => ({ id: String(x.ID_POLIGONO).trim(), y: Number(x.RENDIMIENTO_T_HA) }))
    .sort((a, b) => a.id.localeCompare(b.id, "es"));
  const unique = new Set(records.map(x => x.id));
  if (records.length !== 59 || unique.size !== 59 || records.some(x => !/^AGC_\d+$/.test(x.id) || !Number.isFinite(x.y) || x.y < 0)) {
    throw new Error("Archivo de predicciones distinto del contrato de 59 consultas.");
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
    status.textContent = "Estimación calculada y congelada en Checkpoint 05. El rendimiento real de esta parcela no está disponible.";
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
    const rows = await loadCSV("predictions.csv");
    initPredictions(rows);
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
  loadPredictions();
});
