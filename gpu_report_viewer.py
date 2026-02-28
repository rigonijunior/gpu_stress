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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich import box
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
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
    return str(datetime.timedelta(seconds=int(seconds)))


def _sparkline_rich(values, width=None):
    """Generate a full-width Rich Text sparkline with color gradient."""
    if not values:
        return Text("")
    if width is None:
        width = max(console.size.width - 10, 40)
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1
    blocks = " ▁▂▃▄▅▆▇█"

    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    line = Text()
    for v in sampled:
        ratio = (v - mn) / span
        idx = int(ratio * (len(blocks) - 1))
        # Color gradient: cyan → green → yellow → red
        if ratio >= 0.85:
            style = "bold red"
        elif ratio >= 0.65:
            style = "yellow"
        elif ratio >= 0.35:
            style = "green"
        else:
            style = "cyan"
        line.append(blocks[idx], style=style)
    return line


def _big_bar(value, maximum, width=40, label=""):
    """Create a large visual bar with percentage."""
    if maximum <= 0:
        return ""
    pct = min(value / maximum * 100, 100)
    filled = int(round(pct / 100 * width))
    empty = width - filled

    if pct >= 90:
        color = "red"
    elif pct >= 70:
        color = "yellow"
    elif pct >= 40:
        color = "green"
    else:
        color = "cyan"

    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim] {pct:.0f}% {label}"


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


def _safe_stdev(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


# ─────────────────────── STATISTICS ───────────────────────────

def compute_gpu_stats(snapshots, gpu_idx):
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
    config = report.get("config", {})
    mode = config.get("mode", "?")
    mode_label = MODE_LABELS.get(mode, mode)
    dur_req = config.get("duration_requested_s", 0)
    total = report.get("total_elapsed_s", 0)
    result = report.get("result", "?")

    gpus_list = config.get("gpus", [])
    gpu_names = ", ".join(f"GPU {g[0]}: {g[1]}" for g in gpus_list)

    started = report.get("test_started", "?")
    ended = report.get("test_ended", "?")
    try:
        started = datetime.datetime.fromisoformat(started).strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        pass
    try:
        ended = datetime.datetime.fromisoformat(ended).strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        pass

    if "Concluído" in result or "✅" in result:
        result_style = "bold green"
    elif "Interrompido" in result:
        result_style = "bold yellow"
    else:
        result_style = "bold red"

    lines = Text()
    lines.append("  📋 Modo:       ", style="bold cyan")
    lines.append(f"{mode_label}\n", style="bold white")
    lines.append("  �️  GPU(s):     ", style="bold cyan")
    lines.append(f"{gpu_names}\n", style="white")
    lines.append("  ⏱️  Solicitado:  ", style="bold cyan")
    lines.append(f"{_fmt_duration(dur_req) if dur_req > 0 else 'Indefinida'}\n", style="white")
    lines.append("  ⏱️  Real:        ", style="bold cyan")
    lines.append(f"{_fmt_duration(total)}\n", style="white")
    lines.append("  📅 Início:      ", style="bold cyan")
    lines.append(f"{started}\n", style="white")
    lines.append("  📅 Término:     ", style="bold cyan")
    lines.append(f"{ended}\n", style="white")
    lines.append("  📊 Snapshots:   ", style="bold cyan")
    lines.append(f"{len(report.get('snapshots', []))}\n", style="white")
    lines.append("  🏁 Resultado:   ", style="bold cyan")
    lines.append(f"{result}", style=result_style)

    console.print(Panel(
        lines,
        title="[bold white]══ RESUMO DO TESTE ══[/bold white]",
        border_style="bright_blue",
        box=box.DOUBLE_EDGE,
        padding=(1, 1),
    ))


def render_gpu_section(stats, gpu_idx, gpu_name, peak_data):
    """Render a complete GPU analysis section — clean and large."""

    console.print()
    console.print(f"  [bold magenta]{'═' * 60}[/bold magenta]")
    console.print(f"  [bold magenta]  GPU {gpu_idx}: {gpu_name}[/bold magenta]")
    console.print(f"  [bold magenta]{'═' * 60}[/bold magenta]")
    console.print()

    # ── Simple 4-column stats table (no sparklines) ──
    t = Table(box=box.ROUNDED, expand=True, show_lines=True, padding=(0, 1))
    t.add_column("Métrica", style="bold cyan", min_width=16)
    t.add_column("Mín", style="green", justify="right", min_width=12)
    t.add_column("Média", style="yellow", justify="right", min_width=12)
    t.add_column("Máx", style="red", justify="right", min_width=12)

    rows = [
        ("🌡  Temperatura", "temp_c", "°C"),
        ("⚡ Potência", "power_w", " W"),
        ("📊 GPU Load", "util_gpu", "%"),
        ("📊 Mem Bus", "util_mem", "%"),
        ("💾 VRAM", "mem_used_gb", " GB"),
        ("💾 VRAM %", "mem_pct", "%"),
        ("🌀 Fan", "fan_pct", "%"),
        ("🕐 Core Clk", "clock_core_mhz", " MHz"),
        ("🕐 Mem Clk", "clock_mem_mhz", " MHz"),
    ]

    for label, key, unit in rows:
        s = stats.get(key, {})
        if not s.get("values"):
            continue
        if key == "fan_pct" and s["max"] < 0:
            continue

        # Color the max temperature
        if key == "temp_c":
            tc = _temp_color(s["max"])
            max_val = f"[{tc}]{s['max']}{unit}[/{tc}]"
        else:
            max_val = f"{s['max']}{unit}"

        t.add_row(label, f"{s['min']}{unit}", f"{s['avg']}{unit}", max_val)

    console.print(t)

    # ── Peak summary (horizontal, compact) ──
    if peak_data:
        console.print()
        tc = _temp_color(peak_data.get("max_temp_c", 0))
        peak_text = Text()
        peak_text.append("  🏆 Picos:  ", style="bold white")
        peak_text.append(f"Temp ", style="dim")
        peak_text.append(f"{peak_data.get('max_temp_c', '?')}°C", style=tc)
        peak_text.append(f"  │  ", style="dim")
        peak_text.append(f"Power ", style="dim")
        peak_text.append(f"{peak_data.get('max_power_w', '?')} W", style="bold white")
        peak_text.append(f"  │  ", style="dim")
        peak_text.append(f"VRAM ", style="dim")
        peak_text.append(f"{peak_data.get('max_mem_used_gb', '?')} GB", style="bold white")
        peak_text.append(f"  │  ", style="dim")
        peak_text.append(f"Load Médio ", style="dim")
        peak_text.append(f"{peak_data.get('avg_util_gpu', '?')}%", style="bold white")
        console.print(peak_text)

    # ── Full-width sparkline graphs (one per line, easy to read) ──
    console.print()
    console.print("  [bold white]📈 Gráficos Temporais[/bold white]")
    console.print()

    spark_width = max(console.size.width - 20, 30)

    spark_items = [
        ("  🌡  Temp    ", "temp_c", "°C"),
        ("  ⚡ Power   ", "power_w", " W"),
        ("  📊 GPU %   ", "util_gpu", "%"),
        ("  💾 VRAM %  ", "mem_pct", "%"),
    ]

    for label, key, unit in spark_items:
        s = stats.get(key, {})
        vals = s.get("values", [])
        if not vals:
            continue

        # Label with range
        header = Text()
        header.append(label, style="bold cyan")
        header.append(f"[{s['min']}{unit} → {s['max']}{unit}]", style="dim")
        console.print(header)

        # Full-width sparkline
        spark = _sparkline_rich(vals, width=spark_width)
        console.print(f"  ", end="")
        console.print(spark)
        console.print()

    # ── Timeline heatmap (3 wide rows) ──
    render_heatmap(stats)

    # ── Health Verdict ──
    render_verdict(stats)


def render_heatmap(stats):
    """Wide colorful heatmap blocks for temp, power, load."""
    temps = stats.get("temp_c", {}).get("values", [])
    if not temps:
        return

    bar_width = max(console.size.width - 20, 30)

    console.print("  [bold white]🗺️  Heatmap[/bold white]")
    console.print()

    def _build_heatmap_line(values, thresholds):
        """thresholds: list of (limit, style) from highest to lowest."""
        step = max(1, len(values) // bar_width)
        line = Text()
        # Use wider blocks ██ for better visibility
        for i in range(0, len(values), step):
            v = values[i]
            style = thresholds[-1][1]  # default
            for limit, s in thresholds:
                if v >= limit:
                    style = s
                    break
            line.append("██", style=style)
        return line

    # Temperature
    console.print("  [bold cyan]🌡  Temp[/bold cyan]   ", end="")
    line = _build_heatmap_line(temps, [
        (90, "bold red"), (80, "red"), (70, "yellow"), (60, "green"), (0, "cyan")
    ])
    console.print(line)
    console.print("             [cyan]<60[/] [green]60-70[/] [yellow]70-80[/] [red]80-90[/] [bold red]90+[/]")
    console.print()

    # Power (relative)
    powers = stats.get("power_w", {}).get("values", [])
    if powers:
        max_pwr = max(powers)
        console.print("  [bold cyan]⚡ Power[/bold cyan]  ", end="")
        step = max(1, len(powers) // bar_width)
        line = Text()
        for i in range(0, len(powers), step):
            ratio = powers[i] / max_pwr if max_pwr > 0 else 0
            if ratio >= 0.9:
                line.append("██", style="bold red")
            elif ratio >= 0.7:
                line.append("██", style="yellow")
            elif ratio >= 0.4:
                line.append("██", style="green")
            else:
                line.append("██", style="dim")
        console.print(line)
        console.print(f"             [dim]<40%[/] [green]40-70%[/] [yellow]70-90%[/] [bold red]90%+[/] (max {max_pwr:.0f}W)")
        console.print()

    # GPU Load
    utils = stats.get("util_gpu", {}).get("values", [])
    if utils:
        console.print("  [bold cyan]📊 Load[/bold cyan]   ", end="")
        line = _build_heatmap_line(utils, [
            (95, "bold green"), (70, "green"), (40, "yellow"), (0, "red")
        ])
        console.print(line)
        console.print("             [red]<40%[/] [yellow]40-70%[/] [green]70-95%[/] [bold green]95%+[/]")
        console.print()

    # Time axis
    timestamps = stats.get("_timestamps", [])
    if timestamps:
        dur = timestamps[-1]
        axis = f"             0s ─── {_fmt_duration(dur * 0.25)} ─── {_fmt_duration(dur * 0.5)} ─── {_fmt_duration(dur * 0.75)} ─── {_fmt_duration(dur)}"
        console.print(f"[dim]{axis}[/dim]")
        console.print()


def render_verdict(stats):
    max_temp = stats.get("temp_c", {}).get("max", 0)
    avg_util = stats.get("util_gpu", {}).get("avg", 0)

    if max_temp >= 95:
        verdict = "🔴 REPROVADO"
        detail = f"Temperatura atingiu {max_temp}°C — acima do limite seguro!"
        border = "red"
    elif max_temp >= 85:
        verdict = "🟡 ATENÇÃO"
        detail = f"Temperatura alta ({max_temp}°C). Verifique refrigeração."
        border = "yellow"
    elif max_temp >= 75:
        verdict = "🟢 APROVADO"
        detail = f"Temperaturas normais (pico {max_temp}°C). GPU saudável."
        border = "green"
    else:
        verdict = "🟢 APROVADO"
        detail = f"Temperaturas excelentes (pico {max_temp}°C). ❄️ GPU fria."
        border = "green"

    if avg_util < 50 and stats.get("util_gpu", {}).get("max", 0) > 80:
        detail += " ⚠️ Load instável (oscilações grandes)."

    content = Text(justify="center")
    content.append(f"\n{verdict}\n\n", style=f"bold {border}")
    content.append(f"{detail}\n", style="white")

    console.print(Panel(
        Align.center(content),
        title="[bold]🩺 Diagnóstico[/bold]",
        border_style=border,
        box=box.DOUBLE_EDGE,
        padding=(0, 2),
    ))


def render_comparison(all_stats, config):
    if len(all_stats) < 2:
        return

    t = Table(
        title="[bold]⚔️  Comparação entre GPUs[/bold]",
        box=box.ROUNDED,
        expand=True,
        show_lines=True,
    )
    t.add_column("Métrica", style="bold cyan", min_width=16)

    gpus = config.get("gpus", [])
    for idx, name in gpus:
        t.add_column(f"GPU {idx}", style="white", justify="right", min_width=14)

    compare_rows = [
        ("🌡 Temp Máx", "temp_c", "max", "°C"),
        ("🌡 Temp Média", "temp_c", "avg", "°C"),
        ("⚡ Power Máx", "power_w", "max", " W"),
        ("⚡ Power Média", "power_w", "avg", " W"),
        ("📊 Load Médio", "util_gpu", "avg", "%"),
        ("💾 VRAM Máx", "mem_used_gb", "max", " GB"),
        ("🕐 Core Clk Máx", "clock_core_mhz", "max", " MHz"),
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(script_dir, "gpu_report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)

    if not files:
        console.print("[red]❌ Nenhum relatório encontrado no diretório.[/red]")
        sys.exit(1)

    if len(files) == 1:
        return files[0]

    console.print("\n[bold cyan]📂 Relatórios disponíveis:[/bold cyan]\n")

    for i, f in enumerate(files, 1):
        basename = os.path.basename(f)
        size_kb = round(os.path.getsize(f) / 1024, 1)
        try:
            parts = basename.replace("gpu_report_", "").replace(".json", "")
            dt = datetime.datetime.strptime(parts, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            date_str = "?"

        try:
            with open(f, "r") as fh:
                data = json.load(fh)
                mode = data.get("config", {}).get("mode", "?")
                mode_label = MODE_LABELS.get(mode, mode)
                result = data.get("result", "?")
                n_snap = len(data.get("snapshots", []))
                desc = f"{date_str}  │  {mode_label}  │  {n_snap} snaps  │  {result[:30]}"
        except Exception:
            desc = f"{basename} ({size_kb} KB)"

        console.print(f"  [bold yellow]{i:>2}[/bold yellow]) {desc}")

    console.print()
    try:
        choice = input("  Escolha (número): ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
        else:
            console.print("[red]Número inválido.[/red]")
            sys.exit(1)
    except (ValueError, EOFError, KeyboardInterrupt):
        console.print()
        sys.exit(0)


# ─────────────────────── MAIN ─────────────────────────────────

def main():
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
    console.print()
    console.print(Align.center(Text(
        "🔍 GPU STRESS TEST — RELATÓRIO DETALHADO",
        style="bold white on rgb(20,20,80)",
    )))
    console.print()

    # ── Header ──
    render_header(report)

    # ── Per-GPU Analysis ──
    config = report.get("config", {})
    gpus = config.get("gpus", [])
    snapshots = report.get("snapshots", [])

    if not snapshots:
        console.print("\n[yellow]⚠️  Nenhum snapshot neste relatório.[/yellow]")
        return

    all_stats = {}
    for gpu_idx, gpu_name in gpus:
        stats = compute_gpu_stats(snapshots, gpu_idx)
        all_stats[gpu_idx] = stats

        peak_key = f"gpu_{gpu_idx}_peak"
        peak_data = report.get(peak_key, None)

        render_gpu_section(stats, gpu_idx, gpu_name, peak_data)

    # ── Multi-GPU comparison ──
    render_comparison(all_stats, config)

    # ── Footer ──
    console.print()
    console.print(
        f"  [dim]📄 {os.path.basename(filepath)}  │  "
        f"{round(os.path.getsize(filepath) / 1024, 1)} KB  │  "
        f"{len(snapshots)} amostras[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
