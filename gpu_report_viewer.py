#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  GPU Stress Report Viewer                                    ║
║  Beautiful TUI interpreter for gpu_stress JSON reports        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python3 gpu_report_viewer.py                       # interactive file picker
    python3 gpu_report_viewer.py gpu_report_XXXX.json  # direct file
"""

import os
import sys
import json
import glob
import math
import datetime
import statistics

import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich import box


console = Console()

# ─────────────────────── HELPERS ──────────────────────────────

MODE_LABELS = {
    "compute": "Compute (CUDA Cores)",
    "vram": "VRAM (Memória)",
    "mix": "Misto (Compute+VRAM)",
    "pcie": "PCIe / NVLink",
    "transient": "Picos de Energia",
    "nvenc": "NVENC / Vídeo",
    "training": "Treinamento IA",
    "precision": "Precisão FP64/INT8",
    "all_sequential": "Todos em Sequência",
}


def _fmt_duration(seconds):
    """Pretty format seconds into HH:MM:SS."""
    return str(datetime.timedelta(seconds=int(seconds)))


def _sparkline(values, width=50):
    """Generate an ASCII sparkline chart from a list of numbers."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1
    blocks = " ▁▂▃▄▅▆▇█"
    line = ""
    # Resample values to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    for v in sampled:
        idx = int((v - mn) / span * (len(blocks) - 1))
        line += blocks[idx]
    return line


def _temp_color(temp_c):
    if temp_c >= 90:
        return "bold red"
    if temp_c >= 80:
        return "yellow"
    if temp_c >= 70:
        return "dark_orange"
    if temp_c >= 60:
        return "green"
    return "cyan"


def _verdict_color(verdict):
    if verdict == "APROVADO":
        return "bold green"
    if verdict == "ATENÇÃO":
        return "bold yellow"
    return "bold red"


def _safe_stdev(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)

# ─────────────────────── STATISTICS ───────────────────────────

def compute_gpu_stats(snapshots, gpu_idx):
    """Extract per-GPU statistics from all snapshots."""
    metrics = {
        "temp_c": [], "power_w": [], "util_gpu": [], "util_mem": [],
        "mem_used_gb": [], "mem_pct": [], "fan_pct": [],
        "clock_core_mhz": [], "clock_mem_mhz": [],
    }
    timestamps = []

    for snap in snapshots:
        for g in snap.get("gpus", []):
            if g["idx"] != gpu_idx:
                continue
            for key in metrics:
                if key in g:
                    metrics[key].append(g[key])
            timestamps.append(snap.get("elapsed_s", 0))

    stats = {}
    for key, vals in metrics.items():
        if not vals:
            stats[key] = {"min": 0, "max": 0, "avg": 0, "stdev": 0, "values": []}
            continue
        stats[key] = {
            "min": min(vals),
            "max": max(vals),
            "avg": round(statistics.mean(vals), 1),
            "stdev": round(_safe_stdev(vals), 1),
            "values": vals,
        }
    stats["_timestamps"] = timestamps
    return stats


# ─────────────────────── RENDER ───────────────────────────────

def render_header(report):
    """Render the top summary panel."""
    config = report.get("config", {})
    mode = config.get("mode", "?")
    mode_label = MODE_LABELS.get(mode, mode)
    dur_req = config.get("duration_requested_s", 0)
    total = report.get("total_elapsed_s", 0)
    result = report.get("result", "?")

    # GPU list
    gpus_list = config.get("gpus", [])
    gpu_names = ", ".join(f"GPU {g[0]}: {g[1]}" for g in gpus_list)

    # Parse dates
    started = report.get("test_started", "?")
    ended = report.get("test_ended", "?")
    try:
        dt_start = datetime.datetime.fromisoformat(started)
        started = dt_start.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        pass
    try:
        dt_end = datetime.datetime.fromisoformat(ended)
        ended = dt_end.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        pass

    # Result style
    if "Concluído" in result or "✅" in result:
        result_style = "bold green"
    elif "Interrompido" in result:
        result_style = "bold yellow"
    else:
        result_style = "bold red"

    t = Table(show_header=False, box=None, expand=True, padding=(0, 2))
    t.add_column("label", style="bold cyan", min_width=22)
    t.add_column("value", style="white", ratio=1)

    t.add_row("📋 Modo de Teste:", f"[bold]{mode_label}[/bold]")
    t.add_row("🖥️  GPU(s):", gpu_names)
    t.add_row("⏱️  Duração Solicitada:", _fmt_duration(dur_req) if dur_req > 0 else "Indefinida")
    t.add_row("⏱️  Duração Real:", _fmt_duration(total))
    t.add_row("📅 Início:", started)
    t.add_row("📅 Término:", ended)
    t.add_row("📊 Snapshots:", str(len(report.get("snapshots", []))))
    t.add_row("🏁 Resultado:", Text(result, style=result_style))

    console.print(Panel(
        t,
        title="[bold white]═══ RESUMO DO TESTE ═══[/bold white]",
        border_style="bright_blue",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    ))


def render_gpu_stats(stats, gpu_idx, gpu_name, peak_data):
    """Render detailed stats for a single GPU."""
    # ── Stats Table ──
    t = Table(
        title=f"[bold]📊 Estatísticas Detalhadas[/bold]",
        box=box.SIMPLE_HEAVY,
        expand=True,
        show_lines=True,
    )
    t.add_column("Métrica", style="bold cyan", min_width=18)
    t.add_column("Mínimo", style="green", justify="right", min_width=10)
    t.add_column("Média", style="yellow", justify="right", min_width=10)
    t.add_column("Máximo", style="red", justify="right", min_width=10)
    t.add_column("σ (Desvio)", style="dim", justify="right", min_width=10)
    t.add_column("Sparkline", min_width=30)

    rows = [
        ("🌡  Temperatura", "temp_c", "°C", _temp_color),
        ("⚡ Potência", "power_w", " W", None),
        ("📊 GPU Load", "util_gpu", "%", None),
        ("📊 Mem Bus Load", "util_mem", "%", None),
        ("💾 VRAM Usada", "mem_used_gb", " GB", None),
        ("💾 VRAM %", "mem_pct", "%", None),
        ("🌀 Fan", "fan_pct", "%", None),
        ("🕐 Core Clock", "clock_core_mhz", " MHz", None),
        ("🕐 Mem Clock", "clock_mem_mhz", " MHz", None),
    ]

    for label, key, unit, color_fn in rows:
        s = stats.get(key, {})
        if not s.get("values"):
            continue
        # Skip fan if all negative (water cooled)
        if key == "fan_pct" and s["max"] < 0:
            continue

        spark = _sparkline(s["values"])

        # Color max temp
        max_style = ""
        if color_fn:
            max_style = color_fn(s["max"])
            max_val = f"[{max_style}]{s['max']}{unit}[/{max_style}]"
        else:
            max_val = f"{s['max']}{unit}"

        t.add_row(
            label,
            f"{s['min']}{unit}",
            f"{s['avg']}{unit}",
            max_val,
            f"±{s['stdev']}{unit}",
            spark,
        )

    # ── Peak Data (from report) ──
    peak_table = None
    if peak_data:
        peak_table = Table(
            title="[bold]🏆 Picos Registrados[/bold]",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        peak_table.add_column("Métrica", style="bold cyan")
        peak_table.add_column("Valor", style="bold white", justify="right")

        tc = _temp_color(peak_data.get("max_temp_c", 0))
        peak_table.add_row("🌡  Temp. Máxima", f"[{tc}]{peak_data.get('max_temp_c', '?')} °C[/{tc}]")
        peak_table.add_row("⚡ Potência Máxima", f"{peak_data.get('max_power_w', '?')} W")
        peak_table.add_row("💾 VRAM Máxima", f"{peak_data.get('max_mem_used_gb', '?')} GB")
        peak_table.add_row("📊 GPU Load Médio", f"{peak_data.get('avg_util_gpu', '?')}%")

    # ── Health Verdict ──
    max_temp = stats.get("temp_c", {}).get("max", 0)
    avg_util = stats.get("util_gpu", {}).get("avg", 0)

    if max_temp >= 95:
        verdict = "REPROVADO"
        verdict_detail = f"Temperatura atingiu {max_temp}°C — acima do limite seguro!"
    elif max_temp >= 85:
        verdict = "ATENÇÃO"
        verdict_detail = f"Temperatura alta ({max_temp}°C). Verifique refrigeração."
    elif max_temp >= 75:
        verdict = "APROVADO"
        verdict_detail = f"Temperaturas normais (pico {max_temp}°C). GPU saudável."
    else:
        verdict = "APROVADO"
        verdict_detail = f"Temperaturas excelentes (pico {max_temp}°C). ❄️ GPU fria."

    if avg_util < 50 and stats.get("util_gpu", {}).get("max", 0) > 80:
        verdict_detail += " ⚠️ Load instável (oscilações grandes)."

    vc = _verdict_color(verdict)
    verdict_panel = Panel(
        Align.center(Text(f"\n{verdict}\n\n{verdict_detail}\n", justify="center")),
        title="[bold]🩺 Diagnóstico[/bold]",
        border_style=vc.replace("bold ", ""),
        box=box.DOUBLE_EDGE,
    )

    # ── Compose GPU Section ──
    gpu_title = f"GPU {gpu_idx}: {gpu_name}"
    console.print()
    console.print(Panel(
        t,
        title=f"[bold white]═══ {gpu_title} ═══[/bold white]",
        border_style="magenta",
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
    ))

    if peak_table:
        console.print(peak_table)

    console.print(verdict_panel)


def render_timeline_heatmap(stats, gpu_name):
    """Render a timeline heatmap for temperature using colored blocks."""
    temps = stats.get("temp_c", {}).get("values", [])
    powers = stats.get("power_w", {}).get("values", [])
    timestamps = stats.get("_timestamps", [])

    if not temps:
        return

    t = Table(
        title=f"[bold]🗺️  Timeline Heatmap — {gpu_name}[/bold]",
        box=box.SIMPLE,
        expand=True,
    )
    t.add_column("Métrica", style="bold cyan", min_width=12)
    t.add_column("Timeline", ratio=1)
    t.add_column("Legenda", style="dim", min_width=20)

    # Temperature heatmap
    temp_line = Text()
    max_width = min(len(temps), 80)
    step = max(1, len(temps) // max_width)
    for i in range(0, len(temps), step):
        temp = temps[i]
        if temp >= 90:
            temp_line.append("█", style="bold red")
        elif temp >= 80:
            temp_line.append("█", style="red")
        elif temp >= 70:
            temp_line.append("█", style="yellow")
        elif temp >= 60:
            temp_line.append("█", style="green")
        else:
            temp_line.append("█", style="cyan")

    t.add_row("🌡 Temp", temp_line, "[cyan]<60[/] [green]60-70[/] [yellow]70-80[/] [red]80-90[/] [bold red]90+[/]")

    # Power heatmap
    if powers:
        pwr_line = Text()
        max_pwr = max(powers) if powers else 1
        for i in range(0, len(powers), step):
            pwr = powers[i]
            ratio = pwr / max_pwr if max_pwr > 0 else 0
            if ratio >= 0.9:
                pwr_line.append("█", style="bold red")
            elif ratio >= 0.7:
                pwr_line.append("█", style="yellow")
            elif ratio >= 0.4:
                pwr_line.append("█", style="green")
            else:
                pwr_line.append("█", style="dim")
        t.add_row("⚡ Power", pwr_line, f"[dim]<40%[/] [green]40-70%[/] [yellow]70-90%[/] [bold red]90%+[/] (of {max_pwr:.0f}W)")

    # GPU utilization heatmap
    utils = stats.get("util_gpu", {}).get("values", [])
    if utils:
        util_line = Text()
        for i in range(0, len(utils), step):
            u = utils[i]
            if u >= 95:
                util_line.append("█", style="bold green")
            elif u >= 70:
                util_line.append("█", style="green")
            elif u >= 40:
                util_line.append("█", style="yellow")
            else:
                util_line.append("█", style="red")
        t.add_row("📊 Load", util_line, "[red]<40%[/] [yellow]40-70%[/] [green]70-95%[/] [bold green]95%+[/]")

    # Time axis
    if timestamps:
        dur = timestamps[-1]
        marks = ["0s"]
        q_points = [0.25, 0.5, 0.75, 1.0]
        for q in q_points:
            marks.append(_fmt_duration(dur * q))
        t.add_row("⏱️ Tempo", " │ ".join(marks), "")

    console.print()
    console.print(t)


def render_comparison(all_stats, config):
    """If multiple GPUs, render a side-by-side comparison."""
    if len(all_stats) < 2:
        return

    t = Table(
        title="[bold]⚔️  Comparação entre GPUs[/bold]",
        box=box.DOUBLE_EDGE,
        expand=True,
        show_lines=True,
    )
    t.add_column("Métrica", style="bold cyan")

    gpus = config.get("gpus", [])
    for idx, name in gpus:
        t.add_column(f"GPU {idx}", style="white", justify="right")

    compare_rows = [
        ("🌡 Temp Máx", "temp_c", "max", "°C"),
        ("🌡 Temp Média", "temp_c", "avg", "°C"),
        ("⚡ Power Máx", "power_w", "max", " W"),
        ("⚡ Power Média", "power_w", "avg", " W"),
        ("📊 GPU Load Médio", "util_gpu", "avg", "%"),
        ("💾 VRAM Máx", "mem_used_gb", "max", " GB"),
        ("🕐 Core Clock Máx", "clock_core_mhz", "max", " MHz"),
    ]

    for label, key, agg, unit in compare_rows:
        row = [label]
        for idx, name in gpus:
            s = all_stats.get(idx, {}).get(key, {})
            val = s.get(agg, "?")
            row.append(f"{val}{unit}")
        t.add_row(*row)

    console.print()
    console.print(t)


# ─────────────────────── FILE PICKER ──────────────────────────

def pick_report_file():
    """Let the user pick a JSON report interactively."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(script_dir, "gpu_report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)  # newest first

    if not files:
        console.print("[red]❌ Nenhum relatório encontrado no diretório.[/red]")
        sys.exit(1)

    if len(files) == 1:
        return files[0]

    choices = []
    for f in files:
        basename = os.path.basename(f)
        size_kb = round(os.path.getsize(f) / 1024, 1)
        # Try to extract date from filename
        try:
            parts = basename.replace("gpu_report_", "").replace(".json", "")
            dt = datetime.datetime.strptime(parts, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            date_str = "?"

        # Quick peek at mode
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
                mode = data.get("config", {}).get("mode", "?")
                mode_label = MODE_LABELS.get(mode, mode)
                result = data.get("result", "?")
                n_snap = len(data.get("snapshots", []))
                desc = f"{date_str}  │  {mode_label}  │  {n_snap} snaps  │  {size_kb} KB  │  {result[:30]}"
        except Exception:
            desc = f"{basename} ({size_kb} KB)"

        choices.append(questionary.Choice(desc, value=f))

    selected = questionary.select(
        "📂 Selecione um relatório para visualizar:",
        choices=choices,
    ).ask()

    if not selected:
        sys.exit(0)
    return selected


# ─────────────────────── MAIN ─────────────────────────────────

def main():
    # ── Load report ──
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = pick_report_file()

    if not os.path.exists(filepath):
        console.print(f"[red]❌ Arquivo não encontrado: {filepath}[/red]")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        report = json.load(f)

    console.clear()

    # ── Banner ──
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════════════════╗\n", style="bright_blue")
    banner.append("║          ", style="bright_blue")
    banner.append("🔍 GPU STRESS TEST — RELATÓRIO DETALHADO", style="bold white")
    banner.append("          ║\n", style="bright_blue")
    banner.append("╚══════════════════════════════════════════════════════════════╝", style="bright_blue")
    console.print(Align.center(banner))
    console.print()

    # ── Header Summary ──
    render_header(report)

    # ── Per-GPU Analysis ──
    config = report.get("config", {})
    gpus = config.get("gpus", [])
    snapshots = report.get("snapshots", [])

    if not snapshots:
        console.print("\n[yellow]⚠️  Nenhum snapshot gravado neste relatório.[/yellow]")
        return

    all_stats = {}
    for gpu_idx, gpu_name in gpus:
        stats = compute_gpu_stats(snapshots, gpu_idx)
        all_stats[gpu_idx] = stats

        # Peak data from report
        peak_key = f"gpu_{gpu_idx}_peak"
        peak_data = report.get(peak_key, None)

        render_gpu_stats(stats, gpu_idx, gpu_name, peak_data)
        render_timeline_heatmap(stats, gpu_name)

    # ── Multi-GPU comparison ──
    render_comparison(all_stats, config)

    # ── Footer ──
    console.print()
    console.print(Panel(
        f"  📄 Arquivo: [link=file://{filepath}]{os.path.basename(filepath)}[/link]\n"
        f"  📏 Tamanho: {round(os.path.getsize(filepath) / 1024, 1)} KB\n"
        f"  📊 Total de amostras: {len(snapshots)}",
        title="[dim]Info do Relatório[/dim]",
        border_style="dim",
        box=box.SIMPLE,
    ))


if __name__ == "__main__":
    main()
