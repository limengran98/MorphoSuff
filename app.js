/* MorphoSuff project page. All scientific values come from frozen source data. */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[character]));
const labels = {quantitative_proxy:"Quantitative proxy",ranking_proxy:"Ranking proxy",measurement_required:"Measurement required",not_identifiable:"Not identifiable",unresolved:"Unresolved"};
let evidence, selected, activeTier = "all";

const menuButton = $(".menu-toggle");
menuButton.addEventListener("click", () => {
  const open = menuButton.getAttribute("aria-expanded") !== "true";
  menuButton.setAttribute("aria-expanded", String(open));
  menuButton.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  $("#nav").classList.toggle("is-open", open);
});
$$("#nav a").forEach(link => link.addEventListener("click", () => {
  menuButton.setAttribute("aria-expanded", "false");
  menuButton.setAttribute("aria-label", "Open navigation");
  $("#nav").classList.remove("is-open");
}));
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
    menuButton.click(); menuButton.focus();
  }
});

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) $$("#nav a[href^='#']").forEach(link => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
  });
}, {rootMargin:"-15% 0px -65% 0px", threshold:0});
["approach", "evidence", "external", "software"].forEach(id => observer.observe(document.getElementById(id)));

function chart() {
  const width=470, height=305, left=48, right=17, top=16, bottom=46;
  const x = value => left + (value - 0.45) / 0.55 * (width - left - right);
  const y = value => height - bottom - value * (height - top - bottom);
  let grid = "";
  [0, .2, .4, .6, .8, 1].forEach(value => {
    grid += `<line x1="${left}" y1="${y(value)}" x2="${width-right}" y2="${y(value)}" stroke="#e6eded" stroke-width="1"/><text x="${left-10}" y="${y(value)+3.5}" text-anchor="end" class="plot-label">${value.toFixed(1)}</text>`;
  });
  [.5,.6,.7,.8,.9,1].forEach(value => {
    grid += `<text x="${x(value)}" y="${height-bottom+20}" text-anchor="middle" class="plot-label">${value.toFixed(1)}</text>`;
  });
  const points = evidence.reporters.filter(row => Number.isFinite(row.recovery) && Number.isFinite(row.hitRecall)).map(row =>
    `<circle class="reporter-dot" data-reporter="${escapeHTML(row.id)}" cx="${x(row.recovery)}" cy="${y(row.hitRecall)}" r="4.7" fill="${evidence.metadata.palette[row.tier]}" fill-opacity=".85" tabindex="-1" role="button" aria-label="${escapeHTML(row.name)}: recovery ${row.recovery.toFixed(2)}, hit recall ${row.hitRecall.toFixed(2)}, ${labels[row.tier]}"><title>${escapeHTML(row.name)} · ${labels[row.tier]}</title></circle>`
  ).join("");
  $("#landscape").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="group" aria-label="Reporter recovery versus top-5% hit recall. X axis 0.45 to 1; Y axis 0 to 1. Use arrow keys to explore points."><rect x="${x(.7)}" y="${top}" width="${width-right-x(.7)}" height="${y(.6)-top}" fill="#f1f6f9"/>${grid}<line x1="${x(.7)}" y1="${top}" x2="${x(.7)}" y2="${height-bottom}" stroke="#829eaf" stroke-dasharray="4 4"/><line x1="${left}" y1="${y(.6)}" x2="${width-right}" y2="${y(.6)}" stroke="#829eaf" stroke-dasharray="4 4"/>${points}<text class="plot-label plot-axis-title" x="${(left+width-right)/2}" y="${height-5}" text-anchor="middle">Held-out gene recovery (Pearson r)</text><text class="plot-label plot-axis-title" transform="translate(12 ${(top+height-bottom)/2}) rotate(-90)" text-anchor="middle">Top-5% hit recall</text></svg>`;
  const dots = $$(".reporter-dot");
  dots.forEach((dot, index) => {
    dot.addEventListener("click", () => selectReporter(dot.dataset.reporter, true));
    dot.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault(); selectReporter(dot.dataset.reporter, true);
      } else if (["ArrowRight","ArrowDown","ArrowLeft","ArrowUp","Home","End"].includes(event.key)) {
        event.preventDefault();
        let next = index + (["ArrowRight","ArrowDown"].includes(event.key) ? 1 : -1);
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = dots.length-1;
        const target = dots[(next+dots.length)%dots.length];
        selectReporter(target.dataset.reporter, true); target.focus();
      }
    });
  });
}

function renderFilters() {
  const counts = evidence.metadata.tierCounts, palette = evidence.metadata.palette;
  $("#tier-distribution").innerHTML = evidence.metadata.tierOrder.map(tier => `<span class="tier-segment" style="width:${100*counts[tier]/evidence.reporters.length}%;background:${palette[tier]}" title="${labels[tier]}: ${counts[tier]}"></span>`).join("");
  $("#tier-filters").innerHTML = `<button class="tier-filter" data-tier="all" aria-pressed="true">All <span class="filter-count">${evidence.reporters.length}</span></button>` + evidence.metadata.tierOrder.map(tier => `<button class="tier-filter" data-tier="${tier}" aria-pressed="false"><i class="legend-dot" style="background:${palette[tier]}"></i>${labels[tier]}<span class="filter-count">${counts[tier]}</span></button>`).join("");
  $$(".tier-filter").forEach(button => button.addEventListener("click", () => {
    activeTier = button.dataset.tier;
    updateFilterState(); renderList(true);
  }));
}
function updateFilterState() {
  $$(".tier-filter").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.tier === activeTier)));
}
function filteredReporters() {
  const query = $("#reporter-search").value.trim().toLocaleLowerCase();
  return evidence.reporters.filter(row => (activeTier === "all" || row.tier === activeTier) && `${row.name} ${row.category} ${row.id}`.toLocaleLowerCase().includes(query));
}
function renderList(reselect=false) {
  const rows = filteredReporters();
  $("#reporter-count").textContent = `${rows.length} / ${evidence.reporters.length}`;
  if (reselect && rows.length && !rows.some(row => row.id === selected?.id)) {
    selected = rows[0]; renderDetail(); updateChartSelection();
  }
  $("#reporter-list").innerHTML = rows.length ? rows.map(row => `<button class="reporter-row" data-reporter="${escapeHTML(row.id)}" aria-pressed="${row.id===selected?.id}" aria-label="${escapeHTML(row.name)}, ${escapeHTML(row.category)}, ${labels[row.tier]}"><i class="legend-dot" style="background:${evidence.metadata.palette[row.tier]}"></i><strong title="${escapeHTML(row.name)}">${escapeHTML(row.name)}</strong><span class="reporter-category" title="${escapeHTML(row.category)}">${escapeHTML(row.category)}</span></button>`).join("") : '<p class="empty-message">No matching reporters. Try another name or decision filter.</p>';
  $$(".reporter-row").forEach(button => button.addEventListener("click", () => selectReporter(button.dataset.reporter)));
}
function selectReporter(id, resetFilter=false) {
  const row = evidence.reporters.find(item => item.id === id);
  if (!row) return;
  selected = row;
  if (resetFilter) {activeTier="all"; $("#reporter-search").value=""; updateFilterState();}
  renderList(); renderDetail(); updateChartSelection();
}
function updateChartSelection() {
  if (!selected) return;
  $$(".reporter-dot").forEach(dot => {
    const isSelected = dot.dataset.reporter === selected.id;
    dot.classList.toggle("is-selected", isSelected);
    dot.setAttribute("tabindex", isSelected ? "0" : "-1");
    dot.setAttribute("aria-pressed", String(isSelected));
  });
  $("#chart-selection").innerHTML = `<i class="legend-dot" style="background:${evidence.metadata.palette[selected.tier]}"></i><strong>${escapeHTML(selected.name)}</strong><span>${labels[selected.tier]}</span><a href="#reporter-detail">Inspect all criteria ↗</a>`;
}
function metric(key, label, gate, min, max, bandMin, bandMax) {
  const value = selected[key];
  const fraction = item => Math.max(0,Math.min(100,100*(item-min)/(max-min)));
  const missing = !Number.isFinite(value);
  const outside = !missing && (value < bandMin || value > bandMax);
  return `<div class="metric-row"><div class="metric-name">${label}<span class="metric-gate">${gate}</span></div><div class="metric-rail" aria-hidden="true"><span class="metric-band" style="left:${fraction(bandMin)}%;width:${fraction(bandMax)-fraction(bandMin)}%"></span>${missing ? "" : `<span class="metric-pin${outside?" outside":""}" style="left:${fraction(value)}%"></span>`}</div><span class="metric-value" aria-label="${label}: ${missing?"not available":value.toFixed(2)}">${missing?"—":value.toFixed(2)}</span></div>`;
}
function renderDetail() {
  if (!selected) return;
  const gates=evidence.metadata.thresholds;
  let note;
  if (selected.id === "endosome_vps35") note = "<strong>Beyond the tier.</strong> VPS35 meets the quantitative-response criteria, but its leading GO biological-process terms are not retained after response replacement (top-10 Jaccard = 0). Functional conclusions need their own check.";
  else if (selected.tier === "not_identifiable") note = "<strong>The reference matters.</strong> Measured-response reliability is below the reference operating point. This limits what can be established about substitution, even if prediction scores are high.";
  else if (selected.tier === "ranking_proxy") note = "<strong>Supported for prioritization.</strong> The ranking-use criteria are met, but one or more quantitative-response criteria are not. Ranking uses a top-5% recall gate of 0.50; the quantitative reference shown here is 0.60.";
  else if (selected.tier === "measurement_required") note = "<strong>Keep the targeted assay for the tested uses.</strong> The measured response is reproducible, but the prediction does not meet the study’s proxy criteria.";
  else if (selected.tier === "unresolved") note = "<strong>Mixed evidence.</strong> The reference rules do not establish either proxy use or a measurement-required assignment. Inspect the individual criteria in relation to the intended use.";
  else note = `<strong>Check the downstream use.</strong> The quantitative-response criteria are met. Top-10 GO biological-process Jaccard after replacement: <strong>${Number.isFinite(selected.goBP)?selected.goBP.toFixed(2):"not available"}</strong>. This functional readout is evaluated separately from the tier.`;
  $("#reporter-detail").innerHTML = `<div class="detail-top"><div><h3>${escapeHTML(selected.name)}</h3><p class="detail-category">${escapeHTML(selected.category)} · held-out genes</p></div><span class="tier-badge" style="border-left-color:${evidence.metadata.palette[selected.tier]}">${labels[selected.tier]}</span></div>` +
    metric("recovery","Recovery",`Pearson r ≥ ${gates.recovery.toFixed(2)}`,0,1,gates.recovery,1) +
    metric("rank","Response rank",`Spearman ρ ≥ ${gates.rank.toFixed(2)}`,0,1,gates.rank,1) +
    metric("variance","Amplitude",`Variance ratio ${gates.variance[0].toFixed(1)}–${gates.variance[1].toFixed(1)}`,0,2,...gates.variance) +
    metric("hitRecall","Top-5% hit recall",`Recall ≥ ${gates.hitRecall.toFixed(2)}`,0,1,gates.hitRecall,1) +
    metric("reliability","Reference reliability",`Split-half r ≥ ${gates.reliability.toFixed(2)}`,0,1,gates.reliability,1) +
    `<p class="metric-key"><i></i> Quantitative reference bands · scales 0–1; variance 0–2</p><p class="detail-note">${note}</p>`;
}
$("#reporter-search").addEventListener("input", () => renderList(true));

const contexts = {
  ops:[["Paired measurements","Phase-image features → 52 fluorescent reporters."],["Evaluation unit","Held-out genes, with additional field and screen generalization analyses."],["What it asks","Which response and downstream-use properties survive computational replacement?"]],
  oasis:[["Paired measurements","Brightfield images → well-level metabolic activity and LDH-release assays."],["Evaluation unit","Independent held-out compounds; doses and production sources kept within each compound group."],["What it asks","Can images prioritize strong metabolic loss? How does this compare with preserving broader response profiles?"]],
  periscope:[["Paired measurements","Four fluorescent Cell Painting channels → anti-TOMM20. This setting uses fluorescent inputs."],["Evaluation unit","Per-cell and per-guide predictions; guide-partition response reliability."],["What it asks","Can good predictive recovery establish sufficiency when the measured perturbation response has low reliability?"]]
};
function renderContext(key) {
  $$("[data-context]").forEach(button => {
    const active=button.dataset.context===key;
    button.setAttribute("aria-selected",String(active)); button.tabIndex=active?0:-1;
  });
  $("#context-detail").setAttribute("aria-labelledby",`tab-${key}`);
  $("#context-detail").innerHTML=contexts[key].map(([title,body])=>`<dl class="context-item"><dt>${title}</dt><dd>${body}</dd></dl>`).join("");
}
function keyboardTabs(buttons, select) {
  buttons.forEach((button,index)=>button.addEventListener("keydown",event=>{
    if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
    event.preventDefault();
    let next=(index+(event.key==="ArrowRight"?1:-1)+buttons.length)%buttons.length;
    if(event.key==="Home")next=0;if(event.key==="End")next=buttons.length-1;
    select(buttons[next]);buttons[next].focus();
  }));
}
$$("[data-context]").forEach(button=>button.addEventListener("click",()=>renderContext(button.dataset.context)));
keyboardTabs($$("[data-context]"),button=>renderContext(button.dataset.context));
renderContext("ops");

const snippets={
  install:{text:'git clone https://github.com/limengran98/MorphoSuff.git\ncd MorphoSuff\n\npython -m pip install -e ".[sklearn,parquet]"\n\n# Open the new-dataset guide to connect your own data.\n# docs/adapt_new_dataset.md',note:"Install using a GitHub account with repository access. The package includes measurement-assessment tools and optional adapters for predictive models."},
  evidence:{text:'import pandas as pd\nfrom measurement_sufficiency.analysis.tiers import tier_table\n\npath = "paper/figure_sources/final/figure6/source_data/"\nevidence = pd.read_csv(path + "coherent_ensemble_evidence.csv")\nevidence = evidence.rename(columns={\n    "reporter_slug": "reporter_id",\n    "ensemble_recoverability_r": "recoverability_r",\n    "magnitude_variance_ratio": "variance_ratio",\n})\n\ndecisions = tier_table(\n    evidence, group_columns=["reporter_id"], aggregate="none"\n)',note:"Run from the cloned repository root. This example classifies the included evidence table; fitting a predictor to new paired data is a separate step."}
};
let activeCode="install";
function showCode(key){
  activeCode=key;
  $$("[data-code]").forEach(button=>{const active=button.dataset.code===key;button.setAttribute("aria-selected",String(active));button.tabIndex=active?0:-1;});
  $("#code-text").textContent=snippets[key].text;
  $("#code-note").textContent=snippets[key].note;
  $("#code-content").setAttribute("aria-labelledby",`code-${key}`);
}
$$("[data-code]").forEach(button=>button.addEventListener("click",()=>showCode(button.dataset.code)));
keyboardTabs($$("[data-code]"),button=>showCode(button.dataset.code));
showCode("install");
let toastTimer;
function toast(message){$("#toast").textContent=message;$("#toast").classList.add("visible");clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("#toast").classList.remove("visible"),2200);}
$("#copy-code").addEventListener("click",async()=>{
  try{await navigator.clipboard.writeText(snippets[activeCode].text);toast("Code copied");}
  catch{const range=document.createRange();range.selectNodeContents($("#code-text"));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);toast("Code selected — press Ctrl+C or ⌘C");}
});

const dialog=$("#figure-dialog");
let figureTrigger=null;
$$(".figure-zoom").forEach(button=>button.addEventListener("click",()=>{
  figureTrigger=button;
  $("#dialog-title").textContent=button.dataset.caption;
  $("#dialog-image").src=button.dataset.image;
  $("#dialog-image").alt=$("img",button).alt;
  $("#dialog-pdf").href=button.dataset.pdf;
  dialog.showModal();document.body.classList.add("modal-open");$("#close-dialog").focus();
}));
$("#close-dialog").addEventListener("click",()=>dialog.close());
dialog.addEventListener("click",event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}});
dialog.addEventListener("close",()=>{document.body.classList.remove("modal-open");figureTrigger?.focus({preventScroll:true});});

async function loadEvidence(){
  try{
    const response=await fetch("data/evidence.json");
    if(!response.ok)throw new Error(`Evidence request failed (${response.status})`);
    evidence=await response.json();
    if(evidence.reporters.length!==52)throw new Error("Unexpected evidence-table size");
    selected=evidence.reporters.find(row=>row.id==="endosome_vps35")||evidence.reporters[0];
    renderFilters();chart();renderList();renderDetail();updateChartSelection();
  }catch(error){
    $("#landscape").innerHTML='<p class="loading-message">The evidence table could not be loaded. Please reload the page.</p>';
    $("#reporter-detail").textContent="Interactive evidence is unavailable. The original figure PDFs remain available above and below.";
    console.error(error);
  }
}
loadEvidence();
