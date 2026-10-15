# ECG Time-Series Forecasting
### Unified Baselines and Foundation Model Exploration on Physiological Signals

This repository is an independent research exploration of time-series modeling on ECG signals.

Instead of treating model implementation as the final goal, this project asks a simple question:

> **How do different time-series modeling paradigms — from persistence and CNN/LSTM baselines to Transformers, patch-based models, and pretrained foundation models — behave under a unified ECG forecasting setting?**

The project currently provides a reproducible forecasting pipeline for the MIT-BIH Arrhythmia Database, with multiple model families implemented under the same data split, preprocessing, training, and evaluation protocol.

---

## Research Motivation

ECG is a representative physiological time series containing both strong local structures, such as QRS complexes, and longer temporal dependencies.

My previous projects involved CNN/LSTM sequence modeling, IoT sensor data, and Mamba-based long-range modeling. This project extends those experiences toward modern time-series architectures and foundation models.

Rather than asking only which model obtains the lowest error, I am interested in:

1. **Do more complex sequence models consistently outperform simple persistence baselines?**
2. **How do CNN, recurrent, attention-based, and patch-based models differ when modeling ECG signals?**
3. **How sensitive are conclusions to input length, forecasting horizon, and patient-level distribution shift?**
4. **Can pretrained time-series foundation models transfer meaningfully to physiological signals without task-specific training?**

---

## Current Experimental Scope

| Model | Modeling paradigm | Status |
|---|---|---|
| Persistence | Naive forecasting baseline | ✅ Implemented |
| 1D-CNN | Local temporal feature extraction | ✅ Implemented |
| LSTM | Recurrent sequence modeling | ✅ Implemented |
| Transformer | Full-token attention | ✅ Implemented |
| PatchTST | Patch-based Transformer | ✅ Implemented |
| Chronos | Pretrained zero-shot time-series model | 🧪 Adapter implemented, further validation required |
| TimesNet | Temporal 2D variation modeling | ⏳ Planned |
| TimesFM | Pretrained time-series foundation model | ⏳ Planned |

The current task is **raw ECG waveform forecasting**, rather than heartbeat classification or medical diagnosis.

Classification, few-shot learning, anomaly detection, and patient-level adaptation are treated as future research questions and require separate task definitions and evaluation protocols.

---

## Experimental Pipeline

```text
MIT-BIH ECG
    │
    ▼
Record-level Split
    │
    ├── Training records
    ├── Validation records
    └── Held-out test records
    │
    ▼
Training-only Normalization
    │
    ▼
Sliding Windows
Input: 256 samples
Forecast: 64 samples
    │
    ▼
┌─────────────┬───────┬──────┬─────────────┬──────────┐
│ Persistence │  CNN  │ LSTM │ Transformer │ PatchTST │
└─────────────┴───────┴──────┴─────────────┴──────────┘
    │
    ▼
Unified Evaluation
MSE / MAE / Patient-level Error Analysis
