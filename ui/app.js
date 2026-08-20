/* ================================================================
   SpamGuard AI — app.js
   Full interactive logic for the SMS Spam Detection UI
   ================================================================ */

'use strict';

// ── STATE ──────────────────────────────────────────────────────
const state = {
  history: JSON.parse(localStorage.getItem('spamguard_history') || '[]'),
  currentTab: 'detector',
  analyzing: false,
};

// ── API CONFIG ──────────────────────────────────────────────────
const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:5000' : '';
const FASTAPI_BASE = 'http://127.0.0.1:8000';

// Charts storage
let charts = {};

// ── DOM REFS ───────────────────────────────────────────────────
const msgInput      = document.getElementById('msgInput');
const charCount     = document.getElementById('charCount');
const analyzeBtn    = document.getElementById('analyzeBtn');
const resultPanel   = document.getElementById('resultPanel');
const historySection= document.getElementById('historySection');
const historyList   = document.getElementById('historyList');
const sidebar       = document.getElementById('sidebar');
const hamburger     = document.getElementById('hamburger');

// ── INIT ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initCharCounter();
  renderHistory();
  if (state.history.length > 0) historySection.style.display = 'block';

  // Check backend server status and load metrics
  checkServerStatus();
  loadMetrics();
  
  // Set up periodic check
  setInterval(checkServerStatus, 10000);
});

// ── SERVER & METRICS LOAD ───────────────────────────────────────
async function checkServerStatus() {
  const serverStatus = document.getElementById('serverStatus');
  const serverPulse = document.getElementById('serverPulse');
  const serverLabel = document.getElementById('serverLabel');
  const statusBadgeMobile = document.getElementById('statusBadgeMobile');
  const statusList = document.getElementById('statusList');

  try {
    const res = await fetch(`${FASTAPI_BASE}/health`);
    if (!res.ok) throw new Error('Offline');
    const data = await res.json();
    
    serverPulse.className = 'pulse';
    serverLabel.textContent = 'FastAPI Connected';
    serverStatus.style.background = 'rgba(16,185,129,0.15)';
    serverStatus.style.borderColor = 'rgba(16,185,129,0.3)';
    serverStatus.style.color = '#10b981';

    if (statusBadgeMobile) {
      statusBadgeMobile.className = 'status-badge-mobile active';
      statusBadgeMobile.innerHTML = '<span class="pulse"></span> Ready';
    }

    if (statusList) {
      statusList.innerHTML = `
        <div class="status-row">
          <span class="status-dot online"></span>
          <span>XGBoost V2</span>
          <small style="margin-left:auto; opacity:0.6;">Ready</small>
        </div>
      `;
    }
  } catch (err) {
    try {
      const resFlask = await fetch(`${API_BASE}/health`);
      if (resFlask.ok) {
        const flaskData = await resFlask.json();
        serverPulse.className = 'pulse';
        serverLabel.textContent = 'Flask Connected';
        serverStatus.style.background = 'rgba(16,185,129,0.15)';
        serverStatus.style.borderColor = 'rgba(16,185,129,0.3)';
        serverStatus.style.color = '#10b981';
        return;
      }
    } catch (e) {}

    serverPulse.className = 'pulse offline';
    serverLabel.textContent = 'Server Offline';
    serverStatus.style.background = 'rgba(239,68,68,0.15)';
    serverStatus.style.borderColor = 'rgba(239,68,68,0.3)';
    serverStatus.style.color = '#ef4444';

    if (statusBadgeMobile) {
      statusBadgeMobile.className = 'status-badge-mobile offline';
      statusBadgeMobile.innerHTML = '<span class="pulse"></span> Offline';
    }

    if (statusList) {
      statusList.innerHTML = `
        <div class="status-row">
          <span class="status-dot offline"></span>
          <span style="color:#f87171;">Server Offline</span>
        </div>
      `;
    }
  }
}

async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) throw new Error('Failed to load metrics');
    const metrics = await res.json();
    
    // Update KPI cards
    let bestAcc = 0;
    let bestF1 = 0;
    let bestModelName = '';
    
    Object.values(metrics).forEach(m => {
      if (m.accuracy > bestAcc) bestAcc = m.accuracy;
      if (m.f1 > bestF1) {
        bestF1 = m.f1;
        bestModelName = m.name;
      }
    });

    document.getElementById('kpiAccVal').textContent = `${bestAcc.toFixed(2)}%`;
    document.getElementById('kpiF1Val').textContent = `${bestF1.toFixed(2)}%`;
    document.getElementById('kpiModelVal').textContent = bestModelName;

    // Update Top Performing Model Card in Analytics tab
    const bmName = document.getElementById('bmName');
    const bmType = document.getElementById('bmType');
    const bmMetrics = document.getElementById('bmMetrics');
    if (bmName && bmType && bmMetrics) {
      const bestModelKey = Object.keys(metrics).find(k => metrics[k].f1 === bestF1) || 'transformer';
      const bm = metrics[bestModelKey];
      bmName.textContent = bm.name;
      bmType.textContent = `${bm.type} Model · Trained by Member ${bm.member}`;
      bmMetrics.innerHTML = `
        <div class="bm-metric">
          <div class="bm-metric-val">${bm.accuracy.toFixed(2)}%</div>
          <div class="bm-metric-lbl">Accuracy</div>
        </div>
        <div class="bm-metric">
          <div class="bm-metric-val">${bm.precision.toFixed(2)}%</div>
          <div class="bm-metric-lbl">Precision</div>
        </div>
        <div class="bm-metric">
          <div class="bm-metric-val">${bm.recall.toFixed(2)}%</div>
          <div class="bm-metric-lbl">Recall</div>
        </div>
        <div class="bm-metric">
          <div class="bm-metric-val" style="color: ${bm.color}">${bm.f1.toFixed(2)}%</div>
          <div class="bm-metric-lbl">F1-Score</div>
        </div>
      `;
    }

    // Update Leaderboard Card in Analytics tab
    const leaderboard = document.getElementById('leaderboard');
    if (leaderboard) {
      const sortedModels = Object.entries(metrics).sort((a, b) => b[1].f1 - a[1].f1);
      leaderboard.innerHTML = sortedModels.map(([key, m], index) => {
        const rankClass = index === 0 ? 'rank-1' : index === 1 ? 'rank-2' : index === 2 ? 'rank-3' : 'rank-other';
        const rankNum = index === 0 ? '<svg class="lb-crown-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" width="14" height="14"><path d="M2 4l3 12h14l3-12-6 7-4-7-4 7-6-7z"/><path d="M5 20h14"/></svg>' : index + 1;
        return `
          <div class="lb-item">
            <span class="lb-rank ${rankClass}">${rankNum}</span>
            <div class="lb-info">
              <div class="lb-name">${m.name}</div>
              <div class="lb-meta">${m.type} · Member ${m.member}</div>
            </div>
            <span class="lb-f1">${m.f1.toFixed(2)}% <small style="font-size:0.6rem;opacity:0.7;">F1</small></span>
            <div class="lb-bar" style="width: ${m.f1}%"></div>
          </div>
        `;
      }).join('');
    }

    // Update Full Ranking Table in Analytics tab
    const rankingTableBody = document.getElementById('rankingTableBody');
    if (rankingTableBody) {
      const sortedModels = Object.entries(metrics).sort((a, b) => b[1].f1 - a[1].f1);
      rankingTableBody.innerHTML = sortedModels.map(([key, m], index) => {
        const rankClass = index === 0 ? 'rank-1' : index === 1 ? 'rank-2' : index === 2 ? 'rank-3' : 'rank-other';
        return `
          <tr>
            <td><span class="rank-num ${rankClass}">${index + 1}</span></td>
            <td><strong>${m.name}</strong></td>
            <td><span class="type-chip ${m.type.toLowerCase()}">${m.type}</span></td>
            <td>Member ${m.member}</td>
            <td class="metric-cell">${m.accuracy.toFixed(2)}%</td>
            <td class="metric-cell">${m.precision.toFixed(2)}%</td>
            <td class="metric-cell">${m.recall.toFixed(2)}%</td>
            <td class="metric-cell best"><strong style="color:${m.color}">${m.f1.toFixed(2)}%</strong></td>
          </tr>
        `;
      }).join('');
    }

    // Render charts
    renderCharts(metrics);

  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

function renderCharts(metrics) {
  const allKeys = ['lr', 'rf', 'xgb', 'cnn', 'lstm', 'transformer'];
  const labels = allKeys.map(k => metrics[k]?.short || k.toUpperCase());
  const colors = allKeys.map(k => metrics[k]?.color || '#a855f7');

  const configs = [
    { id: 'chartAccuracy', label: 'Accuracy', field: 'accuracy' },
    { id: 'chartF1', label: 'F1-Score', field: 'f1' },
    { id: 'chartPrecision', label: 'Precision', field: 'precision' },
    { id: 'chartRecall', label: 'Recall', field: 'recall' }
  ];

  configs.forEach(cfg => {
    const ctx = document.getElementById(cfg.id);
    if (!ctx) return;

    if (charts[cfg.id]) {
      charts[cfg.id].destroy();
    }

    const dataValues = allKeys.map(k => metrics[k]?.[cfg.field] || 0);

    charts[cfg.id] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: cfg.label,
          data: dataValues,
          backgroundColor: colors,
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1e293b',
            titleFont: { family: 'Inter', weight: 'bold' },
            bodyFont: { family: 'Inter' },
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 10 } }
          },
          y: {
            min: 80,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 10 } }
          }
        }
      }
    });
  });
}

// ── TAB SWITCHING ──────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(n => n.classList.remove('active'));

  document.getElementById('tab' + capitalize(tab)).classList.add('active');
  document.getElementById('nav' + capitalize(tab)).classList.add('active');

  state.currentTab = tab.toLowerCase();

  if (window.innerWidth <= 768) closeSidebar();
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ── MOBILE SIDEBAR ─────────────────────────────────────────────
function toggleSidebar() {
  const isOpen = sidebar.classList.contains('open');
  if (isOpen) closeSidebar();
  else openSidebar();
}

function openSidebar() {
  sidebar.classList.add('open');
  hamburger.classList.add('active');
  if (!document.getElementById('sidebarBackdrop')) {
    const bd = document.createElement('div');
    bd.id = 'sidebarBackdrop';
    bd.style.cssText = `
      position:fixed; inset:0; z-index:99;
      background:rgba(0,0,0,0.6);
      backdrop-filter: blur(4px);
      animation: fadeIn 250ms ease;
    `;
    bd.onclick = closeSidebar;
    document.body.appendChild(bd);
  }
}

// Close mobile sidebar
function closeSidebar() {
  sidebar.classList.remove('open');
  hamburger.classList.remove('active');
  const bd = document.getElementById('sidebarBackdrop');
  if (bd) bd.remove();
}

// ── CHAR COUNTER ───────────────────────────────────────────────
function initCharCounter() {
  msgInput.addEventListener('input', () => {
    const len = msgInput.value.length;
    charCount.textContent = len;
    charCount.style.color = len > 450
      ? '#f87171'
      : len > 350
        ? '#fbbf24'
        : '#94a3b8';
  });
}

// ── CLEAR INPUT ────────────────────────────────────────────────
function clearInput() {
  msgInput.value = '';
  charCount.textContent = '0';
  charCount.style.color = '';
  msgInput.focus();
}

// ── LOAD SAMPLE ────────────────────────────────────────────────
function loadSample(text) {
  msgInput.value = text;
  charCount.textContent = text.length;
  msgInput.focus();
  msgInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
  msgInput.style.boxShadow = '0 0 0 3px rgba(168,85,247,0.3), inset 0 0 24px rgba(168,85,247,0.06)';
  msgInput.style.borderColor = 'rgba(168,85,247,0.5)';
  setTimeout(() => {
    msgInput.style.boxShadow = '';
    msgInput.style.borderColor = '';
  }, 1200);
}

// ── GET SELECTED MODEL ─────────────────────────────────────────
function getSelectedModel() {
  const checked = document.querySelector('input[name="model"]:checked');
  return checked ? checked.value : 'lr';
}

// ── INTERACTIVE INFERENCE routes ───────────────────────────────
async function handleAnalyze() {
  if (state.analyzing) return;

  const message = msgInput.value.trim();

  if (!message) {
    showError('Please enter an SMS message to analyze.');
    msgInput.focus();
    shakeElement(msgInput);
    return;
  }

  if (message.length < 1) {
    showError('Please enter a valid SMS message.');
    return;
  }

  state.analyzing = true;
  setLoadingState(true);

  // Hide comparisons/explainability beforehand
  const comparePanel = document.getElementById('comparePanel');
  if (comparePanel) comparePanel.style.display = 'none';
  const explainPanel = document.getElementById('explainPanel');
  if (explainPanel) explainPanel.style.display = 'none';

  try {
    const res = await fetch(`${FASTAPI_BASE}/api/v1/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message })
    });

    if (!res.ok) {
      if (res.status === 422) {
        throw new Error('Please enter a valid SMS message.');
      } else if (res.status === 503) {
        throw new Error('Prediction service is currently unavailable. Please try again.');
      } else if (res.status === 500) {
        throw new Error('Prediction failed due to an internal server error. Please try again.');
      } else {
        throw new Error(`Service request failed (${res.status}).`);
      }
    }

    const data = await res.json();
    const formattedResult = {
      isSpam: data.prediction === 1 || data.label === 'SPAM',
      confidence: typeof data.confidence === 'number' ? data.confidence : 0,
      modelName: data.model || 'XGBoost V2',
      modelKey: 'xgb_v2',
      modelType: 'ML',
      processingTimeMs: data.processing_time_ms,
      cleanedText: data.cleaned_text
    };

    showResult(formattedResult, message);
    addToHistory(formattedResult, message);
    updateExplainability(message, formattedResult.isSpam);

  } catch (err) {
    if (err.name === 'TypeError' || (err.message && err.message.includes('fetch'))) {
      showError('Cannot connect to the prediction service. Please make sure the FastAPI server is running.');
    } else {
      showError(err.message || 'Cannot connect to the prediction service. Please make sure the FastAPI server is running.');
    }
  } finally {
    state.analyzing = false;
    setLoadingState(false);
  }
}

async function handleCompareAll() {
  if (state.analyzing) return;

  const message = msgInput.value.trim();

  if (!message) {
    showError('Please enter an SMS message to compare.');
    msgInput.focus();
    shakeElement(msgInput);
    return;
  }

  if (message.length < 3) {
    showError('Message is too short. Please enter at least 3 characters.');
    return;
  }

  state.analyzing = true;
  setLoadingState(true);

  // Hide prediction results / explainability
  resultPanel.style.display = 'none';
  const explainPanel = document.getElementById('explainPanel');
  if (explainPanel) explainPanel.style.display = 'none';

  try {
    const res = await fetch(`${API_BASE}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message })
    });

    if (!res.ok) {
      let errMsg = 'Comparison failed';
      try {
        const errData = await res.json();
        errMsg = errData.error || errMsg;
      } catch (e) {
        errMsg = `Server error (${res.status})`;
      }
      throw new Error(errMsg);
    }


    const data = await res.json();
    
    const comparePanel = document.getElementById('comparePanel');
    const compareBody = document.getElementById('compareBody');
    
    if (comparePanel && compareBody) {
      compareBody.innerHTML = data.results.map((r, index) => {
        const isSpam = r.prediction === 1;
        const predLabel = isSpam 
          ? '<span class="pred-tag pred-tag-spam"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> SPAM</span>' 
          : '<span class="pred-tag pred-tag-ham"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> HAM</span>';
        const predClass = isSpam ? 'pred-spam' : 'pred-ham';
        const typeClass = r.model_type.toLowerCase() === 'ml' ? 'tb-ml' : 'tb-dl';
        const bestBadge = r.is_best ? ' <span class="best-star">★</span>' : '';
        
        return `
          <tr class="${r.is_best ? 'best-row' : ''}">
            <td><strong>${index + 1}</strong></td>
            <td><strong>${r.model_name}</strong></td>
            <td><span class="type-badge ${typeClass}">${r.model_type}</span></td>
            <td><span class="pred-badge ${predClass}">${predLabel}</span></td>
            <td><span class="conf-pill">${r.confidence.toFixed(2)}%${bestBadge}</span></td>
            <td>${r.is_best ? '<strong>Best Match</strong>' : '—'}</td>
          </tr>
        `;
      }).join('');
      
      comparePanel.style.display = 'block';
      comparePanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

  } catch (err) {
    showError(err.message || 'Comparison failed. Make sure the server is running and models are trained.');
  } finally {
    state.analyzing = false;
    setLoadingState(false);
  }
}

// ── LOADING STATE ──────────────────────────────────────────────
function setLoadingState(loading) {
  analyzeBtn.disabled = loading;
  const compareBtn = document.getElementById('compareBtn');
  if (compareBtn) compareBtn.disabled = loading;

  const btnLabel = document.getElementById('analyzeBtnLabel');
  const btnIconWrap = document.getElementById('analyzeBtnIcon');

  if (loading) {
    if (btnLabel) btnLabel.innerHTML = 'Analyzing…';
    if (btnIconWrap) btnIconWrap.innerHTML = `<div class="spinner"></div>`;
  } else {
    if (btnLabel) btnLabel.innerHTML = 'Analyze Message';
    if (btnIconWrap) {
      btnIconWrap.innerHTML = `
        <svg viewBox="0 0 20 20" fill="none" width="18" height="18">
          <path d="M10 2L3 6V10C3 14.4 6.1 18.7 10 20C13.9 18.7 17 14.4 17 10V6L10 2Z" fill="rgba(255,255,255,.2)" stroke="white" stroke-width="1.5"/>
          <path d="M7 10l2.5 2.5 4-5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>`;
    }
  }
}

// ── SHOW RESULT ────────────────────────────────────────────────
function showResult(result, message) {
  const { isSpam, confidence, modelName, modelType, processingTimeMs } = result;

  const typeLabel  = isSpam ? 'SPAM' : 'HAM';
  const typeClass  = isSpam ? 'spam-result' : 'ham-result';
  const bigIcon    = isSpam 
    ? '<svg class="res-svg spam-res-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="36" height="36"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="17"/></svg>' 
    : '<svg class="res-svg ham-res-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="36" height="36"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>';
  const badgeText  = isSpam 
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:4px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Spam Detected' 
    : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:4px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg> Safe Message';
  const mlDlLabel  = modelType === 'ML' ? 'ML Model' : 'DL Model';
  const truncated  = message.length > 80 ? message.slice(0, 80) + '…' : message;
  const latencyStr = processingTimeMs != null ? `&nbsp;·&nbsp; Latency: <code>${processingTimeMs.toFixed(1)}ms</code>` : '';

  resultPanel.style.display = 'block';
  resultPanel.innerHTML = `
    <div class="result-card ${typeClass}">
      <div class="result-emoji">${bigIcon}</div>
      <div class="result-body">
        <div class="result-verdict">${typeLabel}</div>
        <div class="result-meta">
          Model: <code>${modelName}</code> (${mlDlLabel}) &nbsp;·&nbsp;
          Confidence: <code>${confidence.toFixed(2)}%</code>${latencyStr}
          <br/>
          <span style="margin-top:4px;display:block;font-size:0.78rem;opacity:0.7;">
            "${truncated}"
          </span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-label">Confidence: ${confidence.toFixed(2)}%</div>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%"></div>
          </div>
        </div>
      </div>
      <div class="result-badge">${badgeText}</div>
    </div>
  `;

  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── EXPLAINABILITY HIGHLIGHTING ────────────────────────────────
function updateExplainability(message, isSpam) {
  const explainPanel = document.getElementById('explainPanel');
  const explainDesc = document.getElementById('explainDesc');
  const indicatorChips = document.getElementById('indicatorChips');
  const highlightedMsg = document.getElementById('highlightedMsg');

  if (!explainPanel || !explainDesc || !indicatorChips || !highlightedMsg) return;

  if (!isSpam) {
    explainPanel.style.display = 'none';
    return;
  }

  // Common spam indicators
  const indicators = [
    { regex: /\b(win|winner|won)\b/gi, label: 'Winning Claim' },
    { regex: /\b(free|prize|reward|gift|cash|bonus|promo|coupon)\b/gi, label: 'Financial Bait' },
    { regex: /\b(urgent|now|limited|alert|expire|action required)\b/gi, label: 'Urgency / Pressure' },
    { regex: /\b(claim|call|text|reply|contact|apply)\b/gi, label: 'Call to Action' },
    { regex: /£|\$|\d+\s*pound|\d+\s*dollar/gi, label: 'Currency / Money' },
    { regex: /\b\d{4,}\b/g, label: 'Shortcode / Phone Number' },
    { regex: /\b[A-Z]{3,}\b/g, label: 'ALL CAPS' }
  ];

  let matchedChips = new Set();
  let highlighted = escapeHtml(message);

  indicators.forEach(ind => {
    const matches = message.match(ind.regex);
    if (matches && matches.length > 0) {
      matchedChips.add(ind.label);
    }
  });

  const keywords = [
    'win', 'winner', 'won', 'free', 'prize', 'reward', 'gift', 'cash', 'bonus', 'urgent', 'now', 'limited', 'offer', 'text now',
    'reply', 'claim', 'exclusive', 'alert', 'apply', 'contact', 'call', 'expire', 'expires', 'selected', 'congratulations'
  ];

  let tempMsg = highlighted;
  
  tempMsg = tempMsg.replace(/([£$])/g, '<span class="highlight-word">$1</span>');
  tempMsg = tempMsg.replace(/(\b\d{4,}\b)/g, '<span class="highlight-word">$1</span>');
  
  keywords.forEach(kw => {
    const regex = new RegExp(`\\b(${kw})\\b`, 'gi');
    tempMsg = tempMsg.replace(regex, '<span class="highlight-word">$1</span>');
  });

  highlightedMsg.innerHTML = tempMsg;

  if (matchedChips.size > 0) {
    indicatorChips.innerHTML = Array.from(matchedChips).map(chip => `
      <span class="ind-chip found">${chip}</span>
    `).join('');
    explainDesc.textContent = 'Our ensemble AI models identified several high-risk spam indicators within the message text. Below are the key signals and highlighted triggers:';
  } else {
    indicatorChips.innerHTML = '<span class="ind-chip safe">Length / Cap Ratio</span>';
    explainDesc.textContent = 'This message was classified as spam due to structural features (such as length or character casing distribution):';
  }

  explainPanel.style.display = 'block';
  explainPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── SHOW ERROR ─────────────────────────────────────────────────
function showError(msg) {
  resultPanel.style.display = 'block';
  resultPanel.innerHTML = `
    <div class="error-toast">
      <svg viewBox="0 0 20 20" fill="none" width="20" height="20" style="flex-shrink:0">
        <circle cx="10" cy="10" r="9" stroke="#f87171" stroke-width="1.6"/>
        <path d="M10 6v4M10 13h.01" stroke="#f87171" stroke-width="1.6" stroke-linecap="round"/>
      </svg>
      ${msg}
    </div>
  `;
}

// ── HISTORY ────────────────────────────────────────────────────
function addToHistory(result, message) {
  const entry = {
    id: Date.now(),
    isSpam: result.isSpam,
    confidence: result.confidence,
    modelName: result.modelName,
    message: message.slice(0, 120),
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };

  state.history.unshift(entry);
  if (state.history.length > 20) state.history.pop();
  localStorage.setItem('spamguard_history', JSON.stringify(state.history));
  renderHistory();
  historySection.style.display = 'block';
}

function renderHistory() {
  if (!historyList) return;

  if (state.history.length === 0) {
    historySection.style.display = 'none';
    return;
  }

  historyList.innerHTML = state.history.map(entry => {
    const truncated = entry.message.length > 70
      ? entry.message.slice(0, 70) + '…'
      : entry.message;
    return `
      <div class="history-item">
        <span class="hi-badge ${entry.isSpam ? 'hi-spam' : 'hi-ham'}">
          ${entry.isSpam ? 'SPAM' : 'HAM'}
        </span>
        <span class="hi-msg" title="${escapeHtml(entry.message)}">${escapeHtml(truncated)}</span>
        <span class="hi-meta">${entry.modelName} · ${entry.time}</span>
      </div>
    `;
  }).join('');
}

function clearHistory() {
  state.history = [];
  localStorage.removeItem('spamguard_history');
  renderHistory();
}

// ── UTILITIES ──────────────────────────────────────────────────
function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function shakeElement(el) {
  el.style.animation = 'none';
  el.offsetHeight;
  el.style.animation = 'shake 0.4s ease';
  el.addEventListener('animationend', () => { el.style.animation = ''; }, { once: true });
}

const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
  @keyframes shake {
    0%,100% { transform: translateX(0); }
    15%      { transform: translateX(-6px); }
    30%      { transform: translateX(6px); }
    45%      { transform: translateX(-4px); }
    60%      { transform: translateX(4px); }
    75%      { transform: translateX(-2px); }
    90%      { transform: translateX(2px); }
  }
`;
document.head.appendChild(shakeStyle);

// ── PARTICLE CANVAS ────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W = canvas.width  = window.innerWidth;
  let H = canvas.height = window.innerHeight;

  const PARTICLE_COUNT = Math.min(60, Math.floor(W * H / 20000));

  const particles = Array.from({ length: PARTICLE_COUNT }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    r: 1 + Math.random() * 2.5,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    alpha: 0.1 + Math.random() * 0.4,
    color: Math.random() > 0.5 ? '168,85,247' : '99,102,241',
  }));

  function draw() {
    ctx.clearRect(0, 0, W, H);

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(99,102,241,${0.08 * (1 - dist / 130)})`;
          ctx.lineWidth = 1;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`;
      ctx.fill();
    });

    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;
    });
  }

  function loop() {
    draw();
    requestAnimationFrame(loop);
  }
  loop();

  window.addEventListener('resize', () => {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  });
}

// ── KEYBOARD SHORTCUTS ──────────────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    if (state.currentTab === 'detector') handleAnalyze();
  }
  if (e.key === 'Escape') closeSidebar();
});

msgInput.addEventListener('focus', () => {
  analyzeBtn.style.boxShadow = '0 8px 36px rgba(124,58,237,0.55), 0 4px 16px rgba(0,0,0,0.5)';
});
msgInput.addEventListener('blur', () => {
  analyzeBtn.style.boxShadow = '';
});

