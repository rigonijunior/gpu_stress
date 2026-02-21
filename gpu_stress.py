#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  NVIDIA RTX GPU Stress Tester                                ║
║  Stress test for NVIDIA GPUs with real-time TUI dashboard    ║
║  Supports multiple GPUs in parallel                          ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    source venv/bin/activate
    python3 gpu_stress.py

Requirements:
    torch (with CUDA), pynvml, rich, questionary
"""

import os
import sys
import time
import json
import signal
import warnings
import datetime
import traceback
import subprocess
import multiprocessing as mp

# Suppress pynvml deprecation warning
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

import pynvml
import questionary
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

# ─────────────────────────── CONSTANTS ────────────────────────────
TEMP_LIMIT_C = 95       # Hard thermal limit in °C
SAMPLE_INTERVAL = 1.0   # Sensor polling interval in seconds
REPORT_INTERVAL = 5     # Seconds between history snapshots
UI_REFRESH_HZ = 2       # Dashboard refreshes per second

# ─────────────────────────── SENSOR READER ────────────────────────
def read_gpu_metrics(gpu_index):
    """Read all relevant metrics for a single GPU via NVML."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            power_w = 0.0

        try:
            power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except pynvml.NVMLError:
            power_limit_w = 0.0

        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan = -1  # water-cooled or unsupported

        try:
            clk_core = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except pynvml.NVMLError:
            clk_core = 0

        try:
            clk_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError:
            clk_mem = 0

        return {
            "idx": gpu_index,
            "name": name,
            "util_gpu": util.gpu,
            "util_mem": util.memory,
            "mem_used_gb": round(mem.used / (1024 ** 3), 2),
            "mem_total_gb": round(mem.total / (1024 ** 3), 2),
            "mem_pct": round(mem.used / mem.total * 100, 1),
            "temp_c": temp,
            "power_w": round(power_w, 1),
            "power_limit_w": round(power_limit_w, 0),
            "fan_pct": fan,
            "clock_core_mhz": clk_core,
            "clock_mem_mhz": clk_mem,
        }
    except pynvml.NVMLError:
        return None


# ─────────────────────────── STRESS WORKERS ───────────────────────

# Number of kernel launches between each abort-event check.
# Higher = less CPU overhead = more GPU saturation.
_BATCH_ITERS = 50


def _worker_compute(gpu_index, abort_event):
    """
    Saturate GPU CUDA/Tensor cores with continuous matrix multiplications.
    Uses multiple CUDA streams + FP32 & FP16 workloads for maximum load.
    """
    import torch
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)

        # Large matrices to keep all SMs busy
        size = 8192

        # FP32 workload
        a32 = torch.randn(size, size, device=dev)
        b32 = torch.randn(size, size, device=dev)

        # FP16 workload (stresses Tensor Cores on RTX cards)
        a16 = torch.randn(size, size, device=dev, dtype=torch.float16)
        b16 = torch.randn(size, size, device=dev, dtype=torch.float16)

        # Multiple CUDA streams to keep the GPU pipeline full
        streams = [torch.cuda.Stream(device=dev) for _ in range(4)]

        while not abort_event.is_set():
            for _ in range(_BATCH_ITERS):
                # Dispatch work across streams so kernels overlap
                with torch.cuda.stream(streams[0]):
                    torch.matmul(a32, b32)
                with torch.cuda.stream(streams[1]):
                    torch.matmul(a16, b16)
                with torch.cuda.stream(streams[2]):
                    torch.matmul(a32, b32)
                with torch.cuda.stream(streams[3]):
                    torch.matmul(a16, b16)
            # Sync only once per batch to check the abort flag
            torch.cuda.synchronize(dev)
    except Exception:
        traceback.print_exc()


def _worker_vram(gpu_index, abort_event):
    """Fill VRAM to the maximum and then perform continuous R/W on it."""
    import torch
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)

        chunks = []
        chunk_elems = 64 * 1024 * 1024  # ~256 MB per chunk (float32)

        # Phase 1: allocate until OOM
        while not abort_event.is_set():
            try:
                chunks.append(torch.randn(chunk_elems, device=dev))
            except torch.cuda.OutOfMemoryError:
                break

        if not chunks:
            return

        # Phase 2: heavy R/W to keep memory bus and compute busy
        streams = [torch.cuda.Stream(device=dev) for _ in range(2)]
        n = len(chunks)
        while not abort_event.is_set():
            for _ in range(_BATCH_ITERS):
                for i in range(n):
                    s = streams[i % 2]
                    with torch.cuda.stream(s):
                        # Continuous arithmetic to stress memory bandwidth + ALUs
                        chunks[i].mul_(1.00001).add_(0.00001)
            torch.cuda.synchronize(dev)
    except Exception:
        traceback.print_exc()


def _worker_mix(gpu_index, abort_event):
    """Combined heavy compute + VRAM fill — maximum possible GPU stress."""
    import torch
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)

        # ── VRAM fill (~70% of free memory) ──
        pynvml.nvmlInit()
        mem = pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        )
        target_bytes = int(mem.free * 0.70)
        chunk_elems = 64 * 1024 * 1024
        allocated = 0
        vram_chunks = []
        while allocated < target_bytes and not abort_event.is_set():
            try:
                vram_chunks.append(torch.randn(chunk_elems, device=dev))
                allocated += chunk_elems * 4
            except torch.cuda.OutOfMemoryError:
                break

        # ── Compute workload with remaining memory ──
        size = 4096  # Smaller to fit alongside VRAM allocation
        a32 = torch.randn(size, size, device=dev)
        b32 = torch.randn(size, size, device=dev)
        a16 = torch.randn(size, size, device=dev, dtype=torch.float16)
        b16 = torch.randn(size, size, device=dev, dtype=torch.float16)

        streams = [torch.cuda.Stream(device=dev) for _ in range(4)]
        n_chunks = len(vram_chunks)

        while not abort_event.is_set():
            for _ in range(_BATCH_ITERS):
                # Compute on two streams
                with torch.cuda.stream(streams[0]):
                    torch.matmul(a32, b32)
                with torch.cuda.stream(streams[1]):
                    torch.matmul(a16, b16)
                # VRAM R/W on the other two streams
                if n_chunks > 0:
                    with torch.cuda.stream(streams[2]):
                        vram_chunks[0].mul_(1.00001).add_(0.00001)
                    if n_chunks > 1:
                        with torch.cuda.stream(streams[3]):
                            vram_chunks[1].mul_(1.00001).add_(0.00001)
            torch.cuda.synchronize(dev)
    except Exception:
        traceback.print_exc()


def _worker_pcie(gpu_index, abort_event):
    """Heavy Host-to-Device and Device-to-Host transfers to stress PCIe/NVLink."""
    import torch
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)
        
        # ~256MB chunks for large DMA transfers
        size = 64 * 1024 * 1024
        
        # Pinned memory ensures max PCIe bandwidth
        host_tensor = torch.randn(size, pin_memory=True)
        device_tensor = torch.empty(size, device=dev)
        
        while not abort_event.is_set():
            for _ in range(_BATCH_ITERS):
                # Copy H2D
                device_tensor.copy_(host_tensor, non_blocking=True)
                # Copy D2H
                host_tensor.copy_(device_tensor, non_blocking=True)
            torch.cuda.synchronize(dev)
    except Exception:
        traceback.print_exc()


def _worker_transient(gpu_index, abort_event):
    """Spikes GPU load 0% to 100% rapidly to test PSU stability."""
    import time
    import torch
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)
        
        size = 8192
        a32 = torch.randn(size, size, device=dev)
        b32 = torch.randn(size, size, device=dev)
        
        while not abort_event.is_set():
            # 100% Load spike
            for _ in range(20):
                torch.matmul(a32, b32)
            torch.cuda.synchronize(dev)
            
            # 0% Load sleep (creates a transient power spike when waking up)
            time.sleep(0.1)
    except Exception:
        traceback.print_exc()


def _worker_nvenc(gpu_index, abort_event):
    """Uses ffmpeg to stress NVENC/NVDEC chips on the GPU."""
    import time
    import subprocess
    try:
        # Generate 4K dummy video and hard encode to null via NVENC
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda", "-hwaccel_device", str(gpu_index),
            "-f", "lavfi", "-i", "testsrc=duration=3600:size=3840x2160:rate=60",
            "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-b:v", "50M",
            "-f", "null", "-"
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        while not abort_event.is_set():
            # Restart if process fails or ends
            if proc.poll() is not None:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)
            
        proc.terminate()
        proc.wait()
    except Exception:
        traceback.print_exc()


def _worker_training(gpu_index, abort_event):
    """Simulates real-world AI training (Linear layers, Loss, Backprop)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)
        
        # Build an overly wide MLP to saturate CUDA cores + memory
        model = nn.Sequential(
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1000)
        ).to(dev)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        batch_size = 512
        inputs = torch.randn(batch_size, 8192, device=dev)
        targets = torch.randn(batch_size, 1000, device=dev)
        
        while not abort_event.is_set():
            # Train loop
            for _ in range(10): # Smaller inner loop for heavy graph ops
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(dev)
    except Exception:
        traceback.print_exc()


def _worker_precision(gpu_index, abort_event):
    """Stresses non-standard calculations (FP64 and FP16 combined logic)."""
    import torch
    try:
        dev = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(dev)
        
        # Consumer cards are notoriously slow at FP64 (usually 1:64 ratio),
        # so this is massive queue stress.
        size = 4096
        a64 = torch.randn(size, size, device=dev, dtype=torch.float64)
        b64 = torch.randn(size, size, device=dev, dtype=torch.float64)
        
        # We also stress torch matrix int-like ops (FP16/BrainFloat natively supported)
        a16 = torch.randn(size, size, device=dev, dtype=torch.float16)
        b16 = torch.randn(size, size, device=dev, dtype=torch.float16)

        while not abort_event.is_set():
            for _ in range(_BATCH_ITERS):
                # Heavy FP64 bottleneck
                torch.matmul(a64, b64)
                # Super fast FP16
                torch.matmul(a16, b16)
                # Intertwining them breaks branch prediction / pipeline
            torch.cuda.synchronize(dev)
    except Exception:
        traceback.print_exc()


STRESS_FUNCTIONS = {
    "compute": _worker_compute,
    "vram": _worker_vram,
    "mix": _worker_mix,
    "pcie": _worker_pcie,
    "transient": _worker_transient,
    "nvenc": _worker_nvenc,
    "training": _worker_training,
    "precision": _worker_precision,
}


# ─────────────────────────── TUI RENDERING ────────────────────────
def _temp_color(temp_c):
    if temp_c >= 90:
        return "bold red"
    if temp_c >= 80:
        return "yellow"
    if temp_c >= 70:
        return "dark_orange"
    return "green"


def _bar(value, maximum=100, width=20):
    """Return a plain-text bar like ████████░░░░░░░░ 65%."""
    filled = int(round(value / maximum * width)) if maximum else 0
    filled = max(0, min(width, filled))
    empty = width - filled
    pct = f"{value:.0f}%"
    return f"{'█' * filled}{'░' * empty} {pct}"


def build_dashboard(gpus_metrics, elapsed_s, duration_s, status, mode_label):
    """Build the full Rich Layout for the dashboard."""
    console = Console()
    width = console.size.width
    height = console.size.height

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # ── Header ──
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    if duration_s > 0:
        dur_str = str(datetime.timedelta(seconds=int(duration_s)))
        remaining = max(0, duration_s - elapsed_s)
        rem_str = str(datetime.timedelta(seconds=int(remaining)))
        time_info = f"⏱  {elapsed_str} / {dur_str}  (restante: {rem_str})"
    else:
        time_info = f"⏱  {elapsed_str}  (sem limite)"

    status_style = "bold green" if "Running" in status else "bold red"
    if "Concluído" in status:
        status_style = "bold cyan"

    hdr = Table.grid(expand=True)
    hdr.add_column(ratio=1)
    hdr.add_column(ratio=2)
    hdr.add_column(ratio=1)
    hdr.add_row(
        f" 🔥 [bold]GPU Stress[/]",
        Align.center(Text(time_info)),
        Align.right(Text(f"[{mode_label.upper()}] ", style="bold magenta") + Text(status, style=status_style)),
    )
    layout["header"].update(Panel(hdr, style="white on rgb(20,20,60)", box=box.HEAVY))

    # ── GPU panels ──
    n = len(gpus_metrics)
    if n == 0:
        layout["body"].update(Panel("Nenhuma GPU monitorada."))
    else:
        # Decide layout: side-by-side if width allows, else stacked
        if n <= 2 and width >= 80:
            layout["body"].split_row(
                *[Layout(name=f"g{i}") for i in range(n)]
            )
        elif n <= 4 and width >= 120:
            layout["body"].split_row(
                *[Layout(name=f"g{i}") for i in range(n)]
            )
        else:
            layout["body"].split_column(
                *[Layout(name=f"g{i}") for i in range(n)]
            )

        for i, m in enumerate(gpus_metrics):
            if m is None:
                layout[f"g{i}"].update(Panel("[red]Erro ao ler GPU[/red]"))
                continue

            t = Table(show_header=False, box=None, expand=True, padding=(0, 1))
            t.add_column("label", style="bold cyan", min_width=14)
            t.add_column("value", style="white", ratio=1)

            tc = _temp_color(m["temp_c"])
            t.add_row("🌡  Temp:", f"[{tc}]{m['temp_c']} °C[/{tc}]")

            fan_str = f"{m['fan_pct']}%" if m["fan_pct"] >= 0 else "N/A (water?)"
            t.add_row("🌀 Fan:", fan_str)

            t.add_row("⚡ Power:", f"{m['power_w']} W / {m['power_limit_w']:.0f} W")
            t.add_row("🕐 Core Clk:", f"{m['clock_core_mhz']} MHz")
            t.add_row("🕐 Mem Clk:", f"{m['clock_mem_mhz']} MHz")
            t.add_row("", "")
            t.add_row("📊 GPU Load:", _bar(m["util_gpu"]))
            t.add_row(
                f"💾 VRAM:",
                f"{_bar(m['mem_pct'])}  ({m['mem_used_gb']:.1f}/{m['mem_total_gb']:.1f} GB)",
            )

            border = "red" if m["temp_c"] >= 90 else ("yellow" if m["temp_c"] >= 80 else "cyan")
            panel = Panel(
                t,
                title=f"[bold]GPU {m['idx']}: {m['name']}[/bold]",
                border_style=border,
                box=box.ROUNDED,
            )
            layout[f"g{i}"].update(panel)

    # ── Footer ──
    footer_txt = Text(
        f"  CTRL+C = Abortar  │  Limite térmico: {TEMP_LIMIT_C}°C  │  Resultados salvos em JSON ao finalizar",
        style="dim",
    )
    layout["footer"].update(Panel(footer_txt, box=box.SIMPLE))

    return layout


# ─────────────────────────── MAIN ─────────────────────────────────
def main():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    if device_count == 0:
        print("❌ Nenhuma GPU NVIDIA encontrada. Verifique os drivers.")
        pynvml.nvmlShutdown()
        sys.exit(1)

    # ── Interactive menu ──
    gpu_choices = []
    for i in range(device_count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        mem_gb = round(mem.total / (1024 ** 3), 1)
        gpu_choices.append(
            questionary.Choice(f"GPU {i}: {name} ({mem_gb} GB)", value=(i, name))
        )

    selected = questionary.checkbox(
        "🎮 Selecione as GPUs para estressar (espaço para marcar):",
        choices=gpu_choices,
        validate=lambda a: True if len(a) > 0 else "Selecione ao menos uma GPU.",
    ).ask()
    if not selected:
        return

    mode = questionary.select(
        "🔧 Tipo de estresse:",
        choices=[
            questionary.Choice("Compute — Multiplicação de matrizes (estresse nos CUDA cores)", "compute"),
            questionary.Choice("VRAM — Alocação máxima de memória de vídeo", "vram"),
            questionary.Choice("Misto — Compute + VRAM simultaneamente", "mix"),
            questionary.Choice("PCIe/NVLink — Transferências massivas Host <-> Device", "pcie"),
            questionary.Choice("Picos de Energia (Transient) — Carga oscilante (0 a 100%)", "transient"),
            questionary.Choice("NVENC/Video — Teste isolado no chip de codificação de vídeo", "nvenc"),
            questionary.Choice("Treinamento de IA — Simulação de Finetuning de Rede Neural", "training"),
            questionary.Choice("Precisão — Alta carga matemática FP64 e INT8", "precision"),
        ],
    ).ask()
    if not mode:
        return

    dur_choice = questionary.select(
        "⏱  Duração:",
        choices=[
            questionary.Choice("5 minutos", 5 * 60),
            questionary.Choice("15 minutos", 15 * 60),
            questionary.Choice("30 minutos", 30 * 60),
            questionary.Choice("1 hora", 3600),
            questionary.Choice("Indefinido (até CTRL+C)", 0),
            questionary.Choice("Personalizado", -1),
        ],
    ).ask()
    if dur_choice is None:
        return

    if dur_choice == -1:
        raw = questionary.text(
            "  Insira a duração em segundos:",
            validate=lambda v: True if v.isdigit() and int(v) > 0 else "Número > 0",
        ).ask()
        if not raw:
            return
        dur_choice = int(raw)

    duration_s = dur_choice
    mode_labels = {
        "compute": "Compute", 
        "vram": "VRAM", 
        "mix": "Misto",
        "pcie": "PCIe/NVLink",
        "transient": "Transient",
        "nvenc": "Video/NVenc",
        "training": "IA Training",
        "precision": "Precisão",
    }

    # ── Launch workers ──
    abort_event = mp.Event()
    workers = []
    for idx, name in selected:
        p = mp.Process(
            target=STRESS_FUNCTIONS[mode],
            args=(idx, abort_event),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # ── Monitoring loop ──
    console = Console()
    start_ts = time.time()
    last_snap = start_ts
    status = "Running"

    report = {
        "test_started": datetime.datetime.now().isoformat(),
        "config": {
            "gpus": [(i, n) for i, n in selected],
            "mode": mode,
            "duration_requested_s": duration_s,
        },
        "snapshots": [],
        "result": "unknown",
    }

    try:
        with Live(console=console, screen=True, refresh_per_second=UI_REFRESH_HZ) as live:
            while True:
                now = time.time()
                elapsed = now - start_ts

                # ── Duration check ──
                if duration_s > 0 and elapsed >= duration_s:
                    status = "Concluído ✅"
                    abort_event.set()
                    # final render
                    metrics = [read_gpu_metrics(i) for i, _ in selected]
                    live.update(build_dashboard(metrics, elapsed, duration_s, status, mode_labels[mode]))
                    time.sleep(0.5)
                    break

                # ── Collect metrics ──
                metrics = [read_gpu_metrics(i) for i, _ in selected]

                # ── Thermal check ──
                for m in metrics:
                    if m and m["temp_c"] >= TEMP_LIMIT_C:
                        status = f"🛑 ABORTADO: GPU {m['idx']} atingiu {m['temp_c']}°C!"
                        abort_event.set()
                        live.update(build_dashboard(metrics, elapsed, duration_s, status, mode_labels[mode]))
                        time.sleep(1)
                        raise SystemExit(status)

                # ── Snapshot for report ──
                if now - last_snap >= REPORT_INTERVAL:
                    report["snapshots"].append({
                        "ts": datetime.datetime.now().isoformat(),
                        "elapsed_s": round(elapsed, 1),
                        "gpus": [m for m in metrics if m],
                    })
                    last_snap = now

                # ── Render ──
                live.update(build_dashboard(metrics, elapsed, duration_s, status, mode_labels[mode]))
                time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        abort_event.set()
        status = "Interrompido pelo usuário (CTRL+C)"
        console.print(f"\n[bold yellow]{status}[/bold yellow]")
    except SystemExit as e:
        abort_event.set()
        console.print(f"\n[bold red]{e}[/bold red]")

    # ── Cleanup ──
    abort_event.set()
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()

    end_ts = time.time()
    report["result"] = status
    report["test_ended"] = datetime.datetime.now().isoformat()
    report["total_elapsed_s"] = round(end_ts - start_ts, 1)

    # ── Peak metrics summary ──
    if report["snapshots"]:
        for idx, name in selected:
            gpu_snaps = [
                g for s in report["snapshots"] for g in s["gpus"] if g["idx"] == idx
            ]
            if gpu_snaps:
                report[f"gpu_{idx}_peak"] = {
                    "max_temp_c": max(g["temp_c"] for g in gpu_snaps),
                    "max_power_w": max(g["power_w"] for g in gpu_snaps),
                    "max_mem_used_gb": max(g["mem_used_gb"] for g in gpu_snaps),
                    "avg_util_gpu": round(
                        sum(g["util_gpu"] for g in gpu_snaps) / len(gpu_snaps), 1
                    ),
                }

    pynvml.nvmlShutdown()

    # ── Save JSON ──
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gpu_report_{ts_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.print()
    console.print(Panel.fit(
        f"[bold green]Teste finalizado![/bold green]\n"
        f"Status: {status}\n"
        f"Duração total: {str(datetime.timedelta(seconds=int(end_ts - start_ts)))}\n"
        f"Relatório: [link=file://{out_path}]{out_path}[/link]",
        title="Resultado",
        border_style="green",
    ))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
