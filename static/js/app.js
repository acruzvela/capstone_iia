/* =========================
   CARGA GRID DE TICKERS
========================= */
async function cargarGridDesdeAPI() {
  const tbody = document.querySelector("#tablaTickers tbody");
  tbody.innerHTML = "";

  const resp = await fetch("/api/companies");
  const data = await resp.json();

  data.companies.forEach(item => {
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td><input type="checkbox" value="${item.ticker}"></td>
      <td>${item.ticker}</td>
      <td>${item.name}</td>
      <td>${item.sector_icb || ""}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* =========================
   DATOS / AGENTE
========================= */
async function consultarSeleccion() {
  const seleccionados = Array.from(
    document.querySelectorAll("#tablaTickers input:checked")
  ).map(ch => ch.value);

  if (!seleccionados.length) {
    alert("Selecciona al menos un ticker");
    return;
  }

  const resp = await fetch("/api/multi", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ tickers: seleccionados, limit: 200 })
  });

  const data = await resp.json();

  // resumen corto
  let lines = [];
  for (const t of seleccionados) {
    const T = t.toUpperCase();
    const n = (data[T] && data[T].prices) ? data[T].prices.length : 0;
    lines.push(`${T}: ${n} filas`);
  }
  addTextCard("Consulta – selección", lines.join("\n"));
}


async function actualizarSeleccion() {
  const seleccionados = Array.from(
    document.querySelectorAll("#tablaTickers input:checked")
  ).map(ch => ch.value);

  if (!seleccionados.length) {
    alert("Selecciona al menos un ticker");
    return;
  }

  const resp = await fetch("/api/update", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ tickers: seleccionados })
  });

  const data = await resp.json();
  addTextCard("Agente – actualización", data.message || "(sin mensaje)");
}

/* =========================
   pipeline NLP
========================= */

async function showRAG() {
  try {
    const resp = await fetch("/api/pipeline/rag");
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const data = await resp.json();
    renderRAGCard(data);

  } catch (err) {
    addTextCard(
      "Pipeline NLP – RAG documental",
      "❌ Error cargando documentos RAG:\n" + err.message
    );
  }
}


async function showSentiment() {
  try {
    const resp = await fetch("/api/pipeline/sentiment");
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const data = await resp.json();
    renderPipelineSentimentCard(data);

  } catch (err) {
    addTextCard("2) Sentimiento agregado", "❌ Error:\n" + err.message);
  }
}


async function showFeatures() {
  try {
    const resp = await fetch("/api/pipeline/features?market=index");
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const data = await resp.json();
    renderPipelineFeaturesCard(data);

  } catch (err) {
    addTextCard(
      "Pipeline NLP – Features mercado + macro",
      escapeHtml("❌ Error obteniendo features:\n" + err.message)
    );
  }
}

async function showSentimentFiles() {
  try {
    const resp = await fetch("/api/pipeline/sentiment-files");
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const data = await resp.json();

    const html = renderSentimentFilesCard(data);
    addHtmlCard("Pipeline NLP – Sentiment outputs", html);

  } catch (err) {
    addTextCard("Pipeline NLP – Sentiment outputs", `❌ Error: ${escapeHtml(String(err))}`);
  }
}

/* =========================
   TRANSFORMER MARKET
========================= */
async function showTransformer() {
  try {
    const stResp = await fetch("/api/transformer/status");
    if (!stResp.ok) throw new Error("HTTP " + stResp.status);
    const st = await stResp.json();

    if (!st.ready) {
      addTextCard(
        "Deep Learning – Transformer (W60 → P5)",
        "⚠️ Transformer no está listo todavía.\n\n" +
        "Detalle:\n" + JSON.stringify(st.exists, null, 2)
      );
      return;
    }

    const mResp = await fetch("/api/transformer/metrics");
    if (!mResp.ok) throw new Error("HTTP " + mResp.status);
    const metrics = await mResp.json();

    const pResp = await fetch("/api/transformer/preds?limit=25");
    if (!pResp.ok) throw new Error("HTTP " + pResp.status);
    const preds = await pResp.json();

    // ---------- construir tabla (FUERA del html) ----------
    const rows = preds.rows || [];
    const cols = ["date", "y_true", "y_pred", "error", "abs_error"];

    function fmt(x) {
      if (x === null || x === undefined) return "";
      if (typeof x === "number") return x.toFixed(6);
      return String(x);
    }

    const tableHtml = `
      <div style="margin-top:12px;">
        <details>
          <summary>Ver predicciones (últimas ${rows.length})</summary>
          <div style="overflow:auto; margin-top:8px;">
            <table>
              <thead>
                <tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>
              </thead>
              <tbody>
                ${rows.map(r => `
                  <tr>
                    ${cols.map(c => `<td>${escapeHtml(fmt(r[c]))}</td>`).join("")}
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    `;

    // ---------- HTML final ----------
    const t = Date.now();
    const html = `
      <button onclick="togglePlot(this)">Ver / ocultar plots</button>

      <div style="display:none; margin-top:10px; text-align:center;">
        <div style="margin:10px 0;">
          <div style="font-weight:600; margin-bottom:6px;">Loss train vs val</div>
          <img src="/api/transformer/plot/loss_train_vs_val.png?t=${t}"
               style="width:100%; max-width:900px; border:1px solid #eee; border-radius:10px;">
        </div>

        <div style="margin:10px 0;">
          <div style="font-weight:600; margin-bottom:6px;">Pred vs True (scatter)</div>
          <img src="/api/transformer/plot/scatter_pred_vs_true.png?t=${t}"
               style="width:100%; max-width:900px; border:1px solid #eee; border-radius:10px;">
        </div>

        <div style="margin:10px 0;">
          <div style="font-weight:600; margin-bottom:6px;">Distribución</div>
          <img src="/api/transformer/plot/hist_pred_true.png?t=${t}"
               style="width:100%; max-width:900px; border:1px solid #eee; border-radius:10px;">
        </div>
      </div>

      ${renderTransformerMetricsCard(metrics)}

      ${tableHtml}
    `;

    addHtmlCard("Deep Learning – Transformer (W60 → P5)", html);

  } catch (err) {
    addTextCard("Deep Learning – Transformer (W60 → P5)", "❌ Error:\n" + String(err));
  }
}




/* =========================
   CAPSTONE MODELOS
========================= */
async function runCapstone(mode) {
  const resp = await fetch(
    `/api/run/model?mode=${mode}&market=index&horizon=3`,
    { method: "POST" }
  );

  const data = await resp.json();
  const out = document.getElementById("capstoneOut");

  const card = document.createElement("div");
  card.className = "capstone-card";

  card.innerHTML = `
    <strong>${mode}</strong>
    <div style="opacity:.7; font-size:.85rem;">
      market=index · horizon=3 · ${new Date().toLocaleString()}
    </div>

    <button onclick="togglePlot(this)">Ver / ocultar plot</button>

    <div style="display:none; text-align:center; margin-top:10px;">
      <img src="/api/plot/${mode}?t=${Date.now()}"
           style="width:100%; max-width:900px; border:1px solid #eee; border-radius:10px;">
    </div>

    <details style="margin-top:10px;">
      <summary>Ver results (JSON)</summary>
      <pre>${JSON.stringify(data.results, null, 2)}</pre>
    </details>
  `;

  out.prepend(card);
}

function togglePlot(btn) {
  const div = btn.nextElementSibling;
  div.style.display = div.style.display === "none" ? "block" : "none";
}

function clearCapstone() {
  document.getElementById("capstoneOut").innerHTML = "";
}

function addTextCard(title, text) {
  const out = document.getElementById("capstoneOut");
  const card = document.createElement("div");
  card.className = "capstone-card";
  card.innerHTML = `
    <strong>${title}</strong>
    <div style="opacity:.7; font-size:.85rem;">${new Date().toLocaleString()}</div>
    <pre style="margin-top:10px;">${text}</pre>
  `;
  out.prepend(card);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function addHtmlCard(title, html) {
  const out = document.getElementById("capstoneOut");
  const card = document.createElement("div");
  card.className = "capstone-card";
  card.innerHTML = `
    <strong>${escapeHtml(title)}</strong>
    <div style="opacity:.7; font-size:.85rem;">${new Date().toLocaleString()}</div>
    <div style="margin-top:10px;">${html}</div>
  `;
  out.prepend(card);
}

function renderPipelineFeaturesCard(d) {
  const txt = `
DATASET FINAL – FEATURES MERCADO + MACRO
---------------------------------------
Market usado: ${d.market || "N/A"}

Dimensiones:
- Filas: ${String(d.shape?.[0] || 0)}
- Columnas: ${String(d.shape?.[1] || 0)}

Rango temporal:
- Desde: ${d.date_range?.min || "N/A"}
- Hasta: ${d.date_range?.max || "N/A"}

Eventos macro:
- has_macro=True: ${String(d.counts?.has_macro_true || 0)}
- macro_sent no nulo: ${String(d.counts?.macro_sent_notna || 0)}

Columnas:
${(d.columns || []).join(", ")}

Muestra (primeras filas):
${JSON.stringify(d.head || [], null, 2)}
`.trim();

  addTextCard(
    "Pipeline NLP – Features mercado + macro",
    txt
  );
}

function renderPipelineSentimentCard(data) {
  let txt = `
SENTIMIENTO MACRO AGREGADO
--------------------------
Fuente: ${data.source || ""}

Rango temporal:
- Desde: ${data.date_range?.min || "N/A"}
- Hasta: ${data.date_range?.max || "N/A"}

Eventos:
- has_macro=True: ${String(data.counts?.has_macro_true || 0)}
- macro_sent válido: ${String(data.counts?.macro_sent_notna || 0)}

Distribución de señales:
${JSON.stringify(data.signal_bins || {}, null, 2)}

Muestra:
${JSON.stringify(data.head || [], null, 2)}
`;

  addTextCard("Pipeline NLP – Sentimiento macro", txt);
}

function renderRAGCard(data) {
  // soporta: {sources:{...}}  o  {buckets:{...}} o incluso vacío
  const sources = data?.sources || data?.buckets || {};
  const total =
    data?.total_docs ??
    data?.totals?.n_files ??
    Object.values(sources).reduce((acc, v) => acc + (v?.n_files ?? v ?? 0), 0);

  const exts = data?.extensions || data?.allowed_exts || [];

  let text = `
RAG DOCUMENTAL – CORPUS MACRO
----------------------------

Root: ${escapeHtml(data?.root || data?.dir || data?.base_dir || "?")}
Total documentos: ${escapeHtml(String(total))}

`;

  // render de fuentes:
  if (Object.keys(sources).length === 0) {
    text += "Fuentes: (no recibidas en payload)\n";
  } else {
    text += "Fuentes:\n";
    for (const [k, v] of Object.entries(sources)) {
      // si v es número -> conteo simple
      // si v es objeto -> buscamos n_files
      const n = (typeof v === "number") ? v : (v?.n_files ?? 0);
      text += `- ${escapeHtml(k)}: ${escapeHtml(String(n))} docs\n`;
    }
  }

  if (exts.length) {
    text += `\nExtensiones: ${escapeHtml(exts.join(", "))}\n`;
  }

  // samples: puede venir como samples o como buckets[x].samples
  const samples = data?.samples || [];
  if (samples.length) {
    text += "\nMuestra:\n";
    samples.forEach(s => {
      text += `- [${escapeHtml(s.source ?? "?")}] ${escapeHtml(s.file ?? s.relpath ?? "?")}\n`;
    });
  }

  addTextCard("Pipeline NLP – RAG documental", escapeHtml(text));
}

function renderSentimentFilesCard(data) {
  const files = data.files || [];

  const header = `
  <div style="margin-bottom:10px;">
    <div><strong>SENTIMENT – OUTPUTS (CSV)</strong></div>
    <div style="opacity:.8; overflow-wrap:anywhere; word-break:break-word;">
      Dir: ${escapeHtml(data.dir || "")}
    </div>
    <div style="opacity:.8;">Timestamp: ${escapeHtml(data.ts || "")}</div>
    <div style="opacity:.8;">N ficheros: ${escapeHtml(String(files.length))}</div>
  </div>
`;


  const blocks = files.map(f => {
    const cols = escapeHtml((f.columns || []).join(", "));
    const shape = f.shape ? `${escapeHtml(String(f.shape[0]))} x ${escapeHtml(String(f.shape[1]))}` : "—";
    const dr = f.date_range ? `${escapeHtml(f.date_range.min || "—")} → ${escapeHtml(f.date_range.max || "—")}` : "—";

    // head como JSON (seguro y simple)
    const headJson = escapeHtml(JSON.stringify(f.head || [], null, 2));

    return `
      <details style="margin:10px 0; padding:8px; border:1px solid #eee; border-radius:8px;">
        <summary style="cursor:pointer;">
          <strong>${escapeHtml(f.file || "")}</strong>
          <span style="opacity:.7;"> · shape ${shape} · date ${dr}</span>
        </summary>

        <div style="margin-top:8px; font-size:.9rem;">
          <div><strong>Columnas:</strong> <span style="opacity:.85;">${cols}</span></div>
          <div style="margin-top:8px;"><strong>Head:</strong></div>
          <pre style="margin-top:6px; white-space:pre; overflow:auto;">${headJson}</pre>
        </div>
      </details>
    `;
  }).join("");

  return header + blocks;
}

function fmtNum(x, d = 4) {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number" || !isFinite(x)) return "—";
  return x.toFixed(d);
}

function renderTransformerMetricsCard(data) {
  const model = data.model || {};
  const base = data.baseline_zero || {};
  const nTest = escapeHtml(String(data.n_test_samples ?? "—"));

  return `
    <details style="margin-top:10px;">
      <summary style="cursor:pointer;">Ver métricas</summary>

      <div style="margin-top:10px; font-size:.9rem;">

        <div style="margin-bottom:8px;">
          <strong>Modelo Transformer</strong>
        </div>

        <table>
          <tbody>
            <tr><td>MAE</td><td>${escapeHtml(fmtNum(model.mae))}</td></tr>
            <tr><td>RMSE</td><td>${escapeHtml(fmtNum(model.rmse))}</td></tr>
            <tr><td>R²</td><td>${escapeHtml(fmtNum(model.r2))}</td></tr>
            <tr><td>Directional accuracy</td><td>${escapeHtml(fmtNum(model.directional_accuracy, 3))}</td></tr>
          </tbody>
        </table>

        <div style="margin:12px 0 6px;">
          <strong>Baseline (retorno = 0)</strong>
        </div>

        <table>
          <tbody>
            <tr><td>MAE</td><td>${escapeHtml(fmtNum(base.mae))}</td></tr>
            <tr><td>RMSE</td><td>${escapeHtml(fmtNum(base.rmse))}</td></tr>
            <tr><td>Directional accuracy</td><td>${escapeHtml(fmtNum(base.directional_accuracy, 3))}</td></tr>
          </tbody>
        </table>

        <div style="margin-top:10px; opacity:.8;">
          Nº muestras test: <strong>${nTest}</strong>
        </div>

        <details style="margin-top:10px;">
          <summary>Ver JSON completo</summary>
          <pre style="margin-top:6px;">${escapeHtml(JSON.stringify(data, null, 2))}</pre>
        </details>

      </div>
    </details>
  `;
}

/* ========================= */
window.addEventListener("DOMContentLoaded", cargarGridDesdeAPI);
