# 🔥 NVIDIA RTX GPU Stress Tester

Ferramenta de linha de comando para estresse e monitoramento de GPUs NVIDIA RTX, com dashboard visual em tempo real no terminal.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Funcionalidades

- **3 modos de estresse**: Compute (CUDA/Tensor cores), VRAM (alocação máxima), Misto
- **Dashboard TUI** em tempo real com [Rich](https://github.com/Textualize/rich) — resiliente a redimensionamento de janela
- **Suporte multi-GPU** — testa múltiplas placas em paralelo
- **Proteção térmica** automática a 95°C
- **Relatório JSON** com snapshots periódicos e métricas de pico
- **Menu interativo** para configuração rápida

## 📊 Métricas Monitoradas

| Métrica     | Descrição                            |
| ----------- | ------------------------------------ |
| Temperatura | °C em tempo real com cores dinâmicas |
| GPU Load    | Utilização dos cores (%)             |
| VRAM        | Uso de memória de vídeo (GB / %)     |
| Power Draw  | Consumo em Watts                     |
| Fan Speed   | Velocidade do cooler (%)             |
| Clock Core  | Frequência dos CUDA cores (MHz)      |
| Clock Mem   | Frequência da memória (MHz)          |

## 🛠 Requisitos

- **OS**: Ubuntu 24.04 (ou qualquer Linux com drivers NVIDIA)
- **GPU**: NVIDIA com suporte CUDA (testado em RTX 3090 / 4060 Ti)
- **Drivers**: NVIDIA driver compatível + `nvidia-smi` funcional
- **Python**: 3.10+

## 🚀 Instalação

```bash
git clone git@github.com:rigonijunior/gpu_stress.git
cd gpu_stress

# Criar venv e instalar dependências
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Nota:** O PyTorch com CUDA pode ser grande (~2 GB). Certifique-se de ter espaço e uma boa conexão.

## ▶️ Uso

### Via launcher (recomendado)

```bash
bash run.sh
```

### Manualmente

```bash
source venv/bin/activate
python3 gpu_stress.py
```

O menu interativo vai guiar você:

1. **Selecione as GPUs** — marque com espaço, confirme com Enter
2. **Tipo de estresse** — Compute, VRAM ou Misto
3. **Duração** — 5m, 15m, 30m, 1h, indefinido ou personalizado

### Atalhos durante o teste

| Tecla    | Ação                             |
| -------- | -------------------------------- |
| `CTRL+C` | Abortar teste e salvar relatório |

## 📝 Relatório JSON

Ao finalizar, um arquivo `gpu_report_YYYYMMDD_HHMMSS.json` é gerado contendo:

```json
{
  "test_started": "2026-02-21T13:00:00",
  "config": {
    "gpus": [[0, "NVIDIA GeForce RTX 3090"]],
    "mode": "compute",
    "duration_requested_s": 300
  },
  "snapshots": [
    {
      "ts": "2026-02-21T13:00:05",
      "elapsed_s": 5.0,
      "gpus": [{ "temp_c": 72, "util_gpu": 99, "power_w": 320.5, "...": "..." }]
    }
  ],
  "gpu_0_peak": {
    "max_temp_c": 81,
    "max_power_w": 350.2,
    "max_mem_used_gb": 23.4,
    "avg_util_gpu": 98.7
  },
  "result": "Concluído ✅"
}
```

## 🔒 Segurança

- **Limite térmico de 95°C** — o teste é automaticamente abortado se qualquer GPU atingir essa temperatura
- Todos os processos de estresse são encerrados de forma limpa ao parar
- O relatório é sempre salvo, mesmo em caso de interrupção

## 🏗 Arquitetura

```
gpu_stress.py          # Script principal (~500 linhas)
├── read_gpu_metrics() # Leitura de sensores via pynvml (NVML)
├── _worker_compute()  # Estresse FP32+FP16 com 4 CUDA streams
├── _worker_vram()     # Alocação máxima + R/W contínuo
├── _worker_mix()      # Combinação compute + VRAM
├── build_dashboard()  # Renderização TUI com Rich Layout
└── main()             # Menu interativo + loop de monitoramento
```

## 📄 Licença

MIT License — use como quiser.
