#!/bin/bash
clear
# ─── GPU Stress Tester Launcher ───
# Ativa o venv e executa o script principal.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Verifica se o venv existe
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "⚙️  Criando ambiente virtual e instalando dependências..."
    python3 -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$SCRIPT_DIR/venv/bin/activate"
fi

python3 "$SCRIPT_DIR/gpu_stress.py"
