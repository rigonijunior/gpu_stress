#!/usr/bin/env python3
"""
Generates a self-contained HTML report from GPU stress test JSON data.
Uses Google Charts + Tailwind CSS via CDN.
"""

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Stress Report — {{TITLE}}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://www.gstatic.com/charts/loader.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
        body { font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .glass { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(16px); border: 1px solid rgba(148, 163, 184, 0.1); }
        .glass-card { background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(148, 163, 184, 0.08); }
        .glow-green { box-shadow: 0 0 20px rgba(34, 197, 94, 0.15); }
        .glow-red { box-shadow: 0 0 20px rgba(239, 68, 68, 0.15); }
        .glow-yellow { box-shadow: 0 0 20px rgba(234, 179, 8, 0.15); }
        .glow-blue { box-shadow: 0 0 20px rgba(59, 130, 246, 0.15); }
        .gradient-text { background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-value { font-size: 2rem; font-weight: 900; line-height: 1; }
        .chart-container { width: 100%; height: 300px; }
        .heatmap-cell { width: 100%; height: 32px; display: flex; gap: 1px; }
        .heatmap-cell div { flex: 1; border-radius: 2px; min-width: 2px; }
        /* Animate cards on load */
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .animate-card { animation: fadeUp 0.5s ease-out forwards; opacity: 0; }
        .animate-card:nth-child(1) { animation-delay: 0.05s; }
        .animate-card:nth-child(2) { animation-delay: 0.1s; }
        .animate-card:nth-child(3) { animation-delay: 0.15s; }
        .animate-card:nth-child(4) { animation-delay: 0.2s; }
        .animate-card:nth-child(5) { animation-delay: 0.25s; }
        .animate-card:nth-child(6) { animation-delay: 0.3s; }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">

<script>
const REPORT = {{JSON_DATA}};
</script>

<!-- Header -->
<header class="max-w-7xl mx-auto mb-8">
    <div class="glass rounded-2xl p-6 md:p-8">
        <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
                <h1 class="text-3xl md:text-4xl font-black gradient-text mb-2">🔥 GPU Stress Report</h1>
                <p class="text-slate-400 text-lg" id="subtitle"></p>
            </div>
            <div id="verdict-badge" class="px-6 py-3 rounded-xl text-xl font-bold text-center"></div>
        </div>
    </div>
</header>

<!-- Summary Cards -->
<section class="max-w-7xl mx-auto mb-8">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="summary-cards"></div>
</section>

<!-- GPU Sections -->
<div id="gpu-sections" class="max-w-7xl mx-auto space-y-8"></div>

<!-- Footer -->
<footer class="max-w-7xl mx-auto mt-12 mb-8">
    <div class="text-center text-slate-600 text-sm">
        <p>Gerado por <span class="font-semibold text-slate-500">GPU Stress Tester</span> — <span id="footer-date"></span></p>
    </div>
</footer>

<script>
// ─── Helpers ───

const MODE_LABELS = {
    compute: "Compute (CUDA Cores)", vram: "VRAM (Memória)", mix: "Misto (Compute+VRAM)",
    pcie: "PCIe / NVLink", transient: "Picos de Energia", nvenc: "NVENC / Vídeo",
    training: "Treinamento IA", precision: "Precisão FP64/INT8", all_sequential: "Todos em Sequência",
};

function fmtDuration(s) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
    return `${h > 0 ? h + 'h ' : ''}${m}m ${sec}s`;
}

function fmtDate(iso) {
    try { const d = new Date(iso); return d.toLocaleString('pt-BR'); } catch { return iso; }
}

function tempColor(t) {
    if (t >= 90) return { text: 'text-red-400', bg: 'bg-red-500' };
    if (t >= 80) return { text: 'text-yellow-400', bg: 'bg-yellow-500' };
    if (t >= 70) return { text: 'text-orange-400', bg: 'bg-orange-500' };
    if (t >= 60) return { text: 'text-green-400', bg: 'bg-green-500' };
    return { text: 'text-cyan-400', bg: 'bg-cyan-500' };
}

function getGpuSnapshots(gpuIdx) {
    const snaps = [];
    for (const s of REPORT.snapshots || []) {
        for (const g of s.gpus || []) {
            if (g.idx === gpuIdx) snaps.push({ ...g, elapsed_s: s.elapsed_s, ts: s.ts });
        }
    }
    return snaps;
}

function calcStats(values) {
    if (!values.length) return { min: 0, max: 0, avg: 0 };
    const min = Math.min(...values), max = Math.max(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    return { min, max, avg: Math.round(avg * 10) / 10 };
}

// ─── Render Header ───

function renderHeader() {
    const cfg = REPORT.config || {};
    const mode = MODE_LABELS[cfg.mode] || cfg.mode;
    const gpus = (cfg.gpus || []).map(g => `GPU ${g[0]}: ${g[1]}`).join(', ');
    document.getElementById('subtitle').textContent =
        `${mode} — ${gpus} — ${fmtDuration(REPORT.total_elapsed_s || 0)}`;

    // Verdict
    const maxTemps = [];
    for (const key of Object.keys(REPORT)) {
        if (key.startsWith('gpu_') && key.endsWith('_peak')) {
            maxTemps.push(REPORT[key].max_temp_c || 0);
        }
    }
    const peakTemp = maxTemps.length ? Math.max(...maxTemps) : 0;
    const badge = document.getElementById('verdict-badge');
    const result = REPORT.result || '';

    if (result.includes('ABORTADO') || peakTemp >= 95) {
        badge.textContent = '🔴 REPROVADO';
        badge.className = 'px-6 py-3 rounded-xl text-xl font-bold text-center bg-red-500/20 text-red-400 border border-red-500/30 glow-red';
    } else if (peakTemp >= 85 || result.includes('Interrompido')) {
        badge.textContent = '🟡 ATENÇÃO';
        badge.className = 'px-6 py-3 rounded-xl text-xl font-bold text-center bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 glow-yellow';
    } else {
        badge.textContent = '🟢 APROVADO';
        badge.className = 'px-6 py-3 rounded-xl text-xl font-bold text-center bg-green-500/20 text-green-400 border border-green-500/30 glow-green';
    }

    document.getElementById('footer-date').textContent = fmtDate(REPORT.test_started);
}

// ─── Summary Cards ───

function renderSummaryCards() {
    const cfg = REPORT.config || {};
    const cards = [
        { icon: '🔧', label: 'Modo', value: MODE_LABELS[cfg.mode] || cfg.mode, color: 'blue' },
        { icon: '⏱️', label: 'Duração', value: fmtDuration(REPORT.total_elapsed_s || 0), color: 'purple' },
        { icon: '📅', label: 'Início', value: fmtDate(REPORT.test_started), color: 'slate' },
        { icon: '📊', label: 'Amostras', value: (REPORT.snapshots || []).length.toString(), color: 'cyan' },
    ];

    const container = document.getElementById('summary-cards');
    cards.forEach((c, i) => {
        const colorMap = {
            blue: 'border-blue-500/20', purple: 'border-purple-500/20',
            slate: 'border-slate-500/20', cyan: 'border-cyan-500/20',
        };
        container.innerHTML += `
            <div class="glass-card rounded-xl p-5 border ${colorMap[c.color]} animate-card">
                <div class="text-2xl mb-2">${c.icon}</div>
                <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">${c.label}</div>
                <div class="text-lg font-bold text-slate-200">${c.value}</div>
            </div>
        `;
    });
}

// ─── GPU Section ───

function renderGpuSection(gpuIdx, gpuName) {
    const snaps = getGpuSnapshots(gpuIdx);
    if (!snaps.length) return;

    const peakKey = `gpu_${gpuIdx}_peak`;
    const peak = REPORT[peakKey] || {};

    const temps = snaps.map(s => s.temp_c);
    const powers = snaps.map(s => s.power_w);
    const utils = snaps.map(s => s.util_gpu);
    const vrams = snaps.map(s => s.mem_used_gb);
    const vramPcts = snaps.map(s => s.mem_pct);
    const fans = snaps.map(s => s.fan_pct);
    const coreClks = snaps.map(s => s.clock_core_mhz);
    const memClks = snaps.map(s => s.clock_mem_mhz);
    const elapsed = snaps.map(s => s.elapsed_s);

    const tStats = calcStats(temps);
    const pStats = calcStats(powers);
    const uStats = calcStats(utils);
    const vStats = calcStats(vrams);
    const fStats = calcStats(fans);

    const tc = tempColor(tStats.max);
    const totalVram = snaps[0]?.mem_total_gb || 0;

    const section = document.createElement('div');
    section.className = 'space-y-6';

    // ── GPU Title ──
    section.innerHTML = `
        <div class="glass rounded-2xl p-6">
            <h2 class="text-2xl font-bold text-white mb-6">🖥️ GPU ${gpuIdx}: ${gpuName}</h2>

            <!-- Stat cards -->
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                <div class="glass-card rounded-xl p-4 animate-card">
                    <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">🌡️ Temp Máx</div>
                    <div class="stat-value ${tc.text}">${tStats.max}°C</div>
                    <div class="text-xs text-slate-500 mt-1">avg ${tStats.avg}°C · min ${tStats.min}°C</div>
                </div>
                <div class="glass-card rounded-xl p-4 animate-card">
                    <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">⚡ Power Máx</div>
                    <div class="stat-value text-yellow-400">${pStats.max} W</div>
                    <div class="text-xs text-slate-500 mt-1">avg ${pStats.avg} W · min ${pStats.min} W</div>
                </div>
                <div class="glass-card rounded-xl p-4 animate-card">
                    <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">📊 GPU Load</div>
                    <div class="stat-value text-green-400">${uStats.avg}%</div>
                    <div class="text-xs text-slate-500 mt-1">max ${uStats.max}% · min ${uStats.min}%</div>
                </div>
                <div class="glass-card rounded-xl p-4 animate-card">
                    <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">💾 VRAM Máx</div>
                    <div class="stat-value text-purple-400">${vStats.max} GB</div>
                    <div class="text-xs text-slate-500 mt-1">de ${totalVram} GB total</div>
                </div>
                <div class="glass-card rounded-xl p-4 animate-card">
                    <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">🌀 Fan Máx</div>
                    <div class="stat-value text-cyan-400">${fStats.max >= 0 ? fStats.max + '%' : 'N/A'}</div>
                    <div class="text-xs text-slate-500 mt-1">${fStats.max >= 0 ? 'avg ' + fStats.avg + '%' : 'water cooled?'}</div>
                </div>
            </div>

            <!-- Charts -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div class="glass-card rounded-xl p-4">
                    <h3 class="text-sm font-semibold text-slate-400 mb-2">🌡️ Temperatura ao Longo do Tempo</h3>
                    <div id="chart-temp-${gpuIdx}" class="chart-container"></div>
                </div>
                <div class="glass-card rounded-xl p-4">
                    <h3 class="text-sm font-semibold text-slate-400 mb-2">⚡ Potência ao Longo do Tempo</h3>
                    <div id="chart-power-${gpuIdx}" class="chart-container"></div>
                </div>
                <div class="glass-card rounded-xl p-4">
                    <h3 class="text-sm font-semibold text-slate-400 mb-2">📊 GPU Load ao Longo do Tempo</h3>
                    <div id="chart-load-${gpuIdx}" class="chart-container"></div>
                </div>
                <div class="glass-card rounded-xl p-4">
                    <h3 class="text-sm font-semibold text-slate-400 mb-2">💾 VRAM ao Longo do Tempo</h3>
                    <div id="chart-vram-${gpuIdx}" class="chart-container"></div>
                </div>
            </div>

            <!-- Heatmap -->
            <div class="glass-card rounded-xl p-4 mb-4">
                <h3 class="text-sm font-semibold text-slate-400 mb-4">🗺️ Heatmap Visual</h3>
                <div class="space-y-3">
                    <div>
                        <div class="flex items-center gap-3 mb-1">
                            <span class="text-xs text-slate-500 w-16">🌡️ Temp</span>
                            <div class="heatmap-cell flex-1" id="heatmap-temp-${gpuIdx}"></div>
                        </div>
                        <div class="flex items-center gap-3">
                            <span class="w-16"></span>
                            <div class="flex gap-2 text-[10px] text-slate-600">
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-cyan-500 inline-block"></span>&lt;60°C</span>
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-green-500 inline-block"></span>60-70</span>
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-yellow-500 inline-block"></span>70-80</span>
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-red-500 inline-block"></span>80+</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="flex items-center gap-3 mb-1">
                            <span class="text-xs text-slate-500 w-16">📊 Load</span>
                            <div class="heatmap-cell flex-1" id="heatmap-load-${gpuIdx}"></div>
                        </div>
                        <div class="flex items-center gap-3">
                            <span class="w-16"></span>
                            <div class="flex gap-2 text-[10px] text-slate-600">
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-red-500 inline-block"></span>&lt;40%</span>
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-yellow-500 inline-block"></span>40-70%</span>
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-green-500 inline-block"></span>70-95%</span>
                                <span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-emerald-400 inline-block"></span>95%+</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="flex items-center gap-3 mb-1">
                            <span class="text-xs text-slate-500 w-16">⚡ Power</span>
                            <div class="heatmap-cell flex-1" id="heatmap-power-${gpuIdx}"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detailed Table -->
            <div class="glass-card rounded-xl overflow-hidden">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-slate-700/50">
                            <th class="text-left p-3 text-slate-500 font-medium">Métrica</th>
                            <th class="text-right p-3 text-green-400 font-medium">Mín</th>
                            <th class="text-right p-3 text-yellow-400 font-medium">Média</th>
                            <th class="text-right p-3 text-red-400 font-medium">Máx</th>
                        </tr>
                    </thead>
                    <tbody id="table-${gpuIdx}" class="mono"></tbody>
                </table>
            </div>
        </div>
    `;

    document.getElementById('gpu-sections').appendChild(section);

    // ── Fill Table ──
    const metrics = [
        { label: '🌡️ Temperatura', values: temps, unit: '°C' },
        { label: '⚡ Potência', values: powers, unit: ' W' },
        { label: '📊 GPU Load', values: utils, unit: '%' },
        { label: '💾 VRAM Usada', values: vrams, unit: ' GB' },
        { label: '💾 VRAM %', values: vramPcts, unit: '%' },
        { label: '🌀 Fan', values: fans.filter(f => f >= 0), unit: '%' },
        { label: '🕐 Core Clk', values: coreClks, unit: ' MHz' },
        { label: '🕐 Mem Clk', values: memClks, unit: ' MHz' },
    ];

    const tbody = document.getElementById(`table-${gpuIdx}`);
    metrics.forEach(m => {
        if (!m.values.length) return;
        const s = calcStats(m.values);
        tbody.innerHTML += `
            <tr class="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                <td class="p-3 text-slate-300">${m.label}</td>
                <td class="p-3 text-right text-green-400">${s.min}${m.unit}</td>
                <td class="p-3 text-right text-yellow-400">${s.avg}${m.unit}</td>
                <td class="p-3 text-right text-red-400">${s.max}${m.unit}</td>
            </tr>
        `;
    });

    // ── Heatmaps ──
    buildHeatmap(`heatmap-temp-${gpuIdx}`, temps, [
        [90, '#ef4444'], [80, '#f59e0b'], [70, '#eab308'], [60, '#22c55e'], [0, '#06b6d4']
    ]);
    buildHeatmap(`heatmap-load-${gpuIdx}`, utils, [
        [95, '#34d399'], [70, '#22c55e'], [40, '#eab308'], [0, '#ef4444']
    ]);
    const maxPwr = pStats.max || 1;
    buildHeatmapRelative(`heatmap-power-${gpuIdx}`, powers, maxPwr, [
        [0.9, '#ef4444'], [0.7, '#eab308'], [0.4, '#22c55e'], [0, '#475569']
    ]);

    // ── Google Charts ──
    drawLineChart(`chart-temp-${gpuIdx}`, elapsed, temps, 'Temp (°C)', '#f97316', [tStats.min - 5, tStats.max + 5]);
    drawLineChart(`chart-power-${gpuIdx}`, elapsed, powers, 'Power (W)', '#eab308', [0, pStats.max * 1.1]);
    drawLineChart(`chart-load-${gpuIdx}`, elapsed, utils, 'GPU Load (%)', '#22c55e', [0, 105]);
    drawAreaChart(`chart-vram-${gpuIdx}`, elapsed, vrams, `VRAM (GB)`, '#a855f7', [0, totalVram]);
}

// ─── Heatmap / Charts ───

function buildHeatmap(containerId, values, thresholds) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const maxBlocks = 200;
    const step = Math.max(1, Math.floor(values.length / maxBlocks));
    for (let i = 0; i < values.length; i += step) {
        const v = values[i];
        let color = thresholds[thresholds.length - 1][1];
        for (const [limit, c] of thresholds) {
            if (v >= limit) { color = c; break; }
        }
        const el = document.createElement('div');
        el.style.backgroundColor = color;
        el.title = `${v}`;
        container.appendChild(el);
    }
}

function buildHeatmapRelative(containerId, values, maxVal, thresholds) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const maxBlocks = 200;
    const step = Math.max(1, Math.floor(values.length / maxBlocks));
    for (let i = 0; i < values.length; i += step) {
        const ratio = maxVal > 0 ? values[i] / maxVal : 0;
        let color = thresholds[thresholds.length - 1][1];
        for (const [limit, c] of thresholds) {
            if (ratio >= limit) { color = c; break; }
        }
        const el = document.createElement('div');
        el.style.backgroundColor = color;
        el.title = `${values[i]} W`;
        container.appendChild(el);
    }
}

function drawLineChart(containerId, xVals, yVals, label, color, vAxisRange) {
    const data = new google.visualization.DataTable();
    data.addColumn('number', 'Tempo (s)');
    data.addColumn('number', label);
    for (let i = 0; i < xVals.length; i++) data.addRow([xVals[i], yVals[i]]);

    const options = {
        backgroundColor: 'transparent', chartArea: { left: 60, top: 20, right: 20, bottom: 40, width: '100%', height: '80%' },
        colors: [color], legend: 'none', lineWidth: 2, curveType: 'function',
        hAxis: { title: 'Tempo (s)', textStyle: { color: '#64748b', fontSize: 10 }, titleTextStyle: { color: '#64748b', fontSize: 11 }, gridlines: { color: '#1e293b' }, baselineColor: '#1e293b' },
        vAxis: { viewWindow: { min: vAxisRange[0], max: vAxisRange[1] }, textStyle: { color: '#64748b', fontSize: 10 }, gridlines: { color: '#1e293b' }, baselineColor: '#1e293b' },
        tooltip: { textStyle: { fontSize: 12 } },
    };

    const chart = new google.visualization.LineChart(document.getElementById(containerId));
    chart.draw(data, options);
}

function drawAreaChart(containerId, xVals, yVals, label, color, vAxisRange) {
    const data = new google.visualization.DataTable();
    data.addColumn('number', 'Tempo (s)');
    data.addColumn('number', label);
    for (let i = 0; i < xVals.length; i++) data.addRow([xVals[i], yVals[i]]);

    const options = {
        backgroundColor: 'transparent', chartArea: { left: 60, top: 20, right: 20, bottom: 40, width: '100%', height: '80%' },
        colors: [color], legend: 'none', lineWidth: 2, curveType: 'function', areaOpacity: 0.15,
        hAxis: { title: 'Tempo (s)', textStyle: { color: '#64748b', fontSize: 10 }, titleTextStyle: { color: '#64748b', fontSize: 11 }, gridlines: { color: '#1e293b' }, baselineColor: '#1e293b' },
        vAxis: { viewWindow: { min: vAxisRange[0], max: vAxisRange[1] }, textStyle: { color: '#64748b', fontSize: 10 }, gridlines: { color: '#1e293b' }, baselineColor: '#1e293b' },
        tooltip: { textStyle: { fontSize: 12 } },
    };

    const chart = new google.visualization.AreaChart(document.getElementById(containerId));
    chart.draw(data, options);
}

// ─── Init ───

google.charts.load('current', { packages: ['corechart'] });
google.charts.setOnLoadCallback(() => {
    renderHeader();
    renderSummaryCards();
    for (const [idx, name] of (REPORT.config?.gpus || [])) {
        renderGpuSection(idx, name);
    }
});
</script>
</body>
</html>"""


def generate_html_report(report_data, output_path):
    """Generate a self-contained HTML report from a report dict."""
    import json

    cfg = report_data.get("config", {})
    mode = cfg.get("mode", "?")
    ts = report_data.get("test_started", "")[:19].replace("T", " ")
    title = f"{mode} — {ts}"

    json_str = json.dumps(report_data, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("{{JSON_DATA}}", json_str).replace("{{TITLE}}", title)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
