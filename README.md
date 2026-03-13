<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ffffff,30:ffe0f0,60:ffb3d9,100:ff69b4&height=250&section=header&text=PULSEML&fontSize=100&fontColor=c2185b&fontAlignY=38&desc=ML%20Monitoring%20%2B%20Data%20Drift%20Detection%20Platform&descAlignY=62&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=22&duration=3000&pause=800&color=E91E8C&center=true&vCenter=true&multiline=true&width=800&height=120&lines=Real-Time+ML+Model+Monitoring;Data+Drift+Detection+%7C+Auto+Alerts;XGBoost+%2B+Evidently+AI+%2B+FastAPI;Is+Your+Model+Still+Working%3F+PulseML+Knows.)](https://git.io/typing-svg)

<br/>

<img src="https://img.shields.io/badge/Python-3.11-ff69b4?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0.3-ffb3d9?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Evidently-AI-ff69b4?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FastAPI-0.110-ffb3d9?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.40-ff69b4?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-LIVE-ff1493?style=for-the-badge"/>

<br/><br/>

> ### *"ML models don't fail loudly. They decay in silence — until it's too late."*
> PulseML watches your model 24/7. Drift detected. Alerts fired. Model saved.

<br/>

[![Live Demo](https://img.shields.io/badge/LIVE_DEMO-Try_PulseML-ff69b4?style=for-the-badge)](https://pulseml-siddhantchandorkar.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai/PulseML)

</div>

---

## WHAT IS PULSEML?

```
╔══════════════════════════════════════════════════════════════════════╗
║     PULSEML — ML Monitoring & Data Drift Detection Platform         ║
║     "Your model trained yesterday. But the world changed today."    ║
║                                                                      ║
║     MONITORS:  Data Drift | Model Performance | Feature Stats       ║
║     DETECTS:   PSI Shift | KS-Test Violations | F1 Degradation     ║
║     ALERTS:    Automated threshold-based firing                     ║
║     DATASET:   IEEE-CIS Fraud Detection — 590,540 transactions      ║
╚══════════════════════════════════════════════════════════════════════╝
```

Most ML projects stop at model training. **PulseML starts where everyone else stops.**

> It answers the question every production ML engineer faces daily: *"Is my model still working?"*

---

## THE PROBLEM

```
You train a model. It hits 94% accuracy. You deploy it.
Three months later — fraud slips through. Revenue drops.
Nobody noticed. The model was "working" the whole time.

The data had drifted. The model had not adapted.
This is the silent killer of production ML systems.
```

| Failure Mode | Industry Impact |
|--------------|----------------|
| **Data Drift** | Input distribution shifts — model predictions become unreliable |
| **Concept Drift** | Fraud patterns evolve — yesterday's model misses today's attacks |
| **Silent Degradation** | F1 drops 15% before anyone checks the dashboard |
| **No Alerting** | Engineers find out from customers, not from systems |

**PulseML solves all four.**

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│              IEEE-CIS Fraud Dataset (590,540 transactions)           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────┐
               │       DATA LOADER          │  ← Load, merge, preprocess
               │    data_loader.py          │     stratified split
               └──────────────┬─────────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │      XGBOOST MODEL         │  ← Fraud classification
               │      model.py              │     scale_pos_weight for imbalance
               └──────────────┬─────────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │    EVIDENTLY MONITOR       │  ← PSI + KS drift detection
               │    monitor.py              │     performance tracking
               └──────────────┬─────────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │      ALERT ENGINE          │  ← Threshold-based firing
               │      alerts.py             │     drift share + F1 drop
               └──────────────┬─────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
         ┌─────────────────┐   ┌──────────────────┐
         │ STREAMLIT DASH  │   │   FASTAPI REST    │
         │ Live monitoring │   │   7 endpoints     │
         │ Plotly charts   │   │   JSON responses  │
         └─────────────────┘   └──────────────────┘
```

---

## DRIFT DETECTION SYSTEM

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   PSI < 0.10      No significant drift detected                     │
│   PSI 0.10-0.20   Minor drift — monitor closely                     │
│   PSI > 0.20      ALERT FIRED — model retraining recommended        │
│                                                                      │
│   KS p-value < 0.05   Distribution shift detected — ALERT           │
│   F1 drop > 5%        Performance degradation — ALERT               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## FEATURES

| Feature | Description |
|---------|-------------|
| **Data Drift Detection** | PSI + KS-test via Evidently AI — statistical rigor |
| **Performance Monitoring** | F1, Accuracy, Precision, Recall, ROC-AUC in real-time |
| **Automated Alerts** | Fires when drift exceeds threshold or F1 degrades |
| **Interactive Dashboard** | Streamlit UI — drift gauges, performance charts, reports |
| **REST API** | 7 FastAPI endpoints for programmatic integration |
| **Fraud Dataset** | IEEE-CIS — 590K transactions, 394 features, 3.5% fraud rate |
| **Class Imbalance** | Handled via XGBoost `scale_pos_weight` |

---

## DATASET

**IEEE-CIS Fraud Detection** — Kaggle

| Property | Value |
|----------|-------|
| Transactions | 590,540 |
| Features | 394 |
| Fraud Rate | 3.5% (severe imbalance) |
| Task | Binary classification |
| Imbalance Strategy | XGBoost `scale_pos_weight` |

---

## API ENDPOINTS

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/prepare` | Load and preprocess data |
| POST | `/train` | Train XGBoost model |
| GET | `/drift` | Run drift detection |
| GET | `/performance` | Run performance monitoring |
| GET | `/monitor` | Full monitoring pipeline |
| GET | `/alerts` | Get active alerts |

---

## TECH STACK

| Layer | Technology | Version | Why |
|-------|-----------|---------|-----|
| **Model** | XGBoost | 2.0.3 | Best-in-class gradient boosting for tabular fraud data |
| **Monitoring** | Evidently AI | Latest | Purpose-built for ML drift detection |
| **Dashboard** | Streamlit | 1.40 | Fast, clean data monitoring UI |
| **API** | FastAPI + Uvicorn | 0.110 | Async, typed, production-ready |
| **Visualization** | Plotly | Latest | Interactive drift and performance charts |
| **Data** | pandas + scikit-learn | Latest | Data pipeline and preprocessing |

---

## PROJECT STRUCTURE

```
PulseML/
├── src/
│   ├── config.py         # Centralized configuration
│   ├── data_loader.py    # Data pipeline — load, merge, preprocess
│   ├── model.py          # XGBoost training + evaluation
│   ├── monitor.py        # Evidently drift + performance reports
│   └── alerts.py         # Threshold-based automated alerting
├── api/
│   └── main.py           # FastAPI REST endpoints
├── app.py                # Streamlit monitoring dashboard
├── requirements.txt      # Pinned dependencies
├── .env.example          # Environment variable template
└── README.md
```

---

## QUICK START

```bash
# 1. Clone
git clone https://github.com/siddhantchandorkar752-ai/PulseML.git
cd PulseML

# 2. Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Add dataset (download from Kaggle IEEE-CIS)
# Place in data/ folder:
# data/train_transaction.csv
# data/train_identity.csv

# 4. Configure
cp .env.example .env

# 5. Run Dashboard
streamlit run app.py

# 6. Run API (optional)
uvicorn api.main:app --reload
```

---

## WHAT I LEARNED

- ML models fail **silently** — monitoring is not optional, it is survival
- Statistical drift detection: PSI measures distribution shift, KS-test catches feature-level divergence
- Building end-to-end MLOps pipelines — raw data to live production monitoring
- Severe class imbalance in fraud detection requires more than oversampling — loss weighting changes everything
- Production-grade modular Python: one concern per file, one responsibility per class

---

## WHY THIS STANDS OUT

```
Average ML project:   Jupyter notebook → 94% accuracy → done.

PulseML:              Training → Deployment → Monitoring → Alerting → Dashboard → API
                      This is what production ML actually looks like.
```

---

## ROADMAP

- [ ] Automated retraining trigger on drift detection
- [ ] Email + Slack alert integration
- [ ] Multi-model comparison dashboard
- [ ] Docker + Kubernetes deployment config
- [ ] Grafana integration for enterprise monitoring

---

## LICENSE

MIT License — free to use, modify, distribute.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:fff0f7,50:ffb3d9,100:fff0f7&height=70&text=Siddhant%20Chandorkar&fontSize=30&fontColor=c2185b&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-ff69b4?style=for-the-badge&logo=github&logoColor=white)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-siddhantchandorkar-ffb3d9?style=for-the-badge&logo=huggingface&logoColor=c2185b)](https://huggingface.co/siddhantchandorkar)

<br/>

*"Most engineers build models. I build systems that keep models alive."*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff69b4,40:ffb3d9,100:ffffff&height=130&section=footer&text=PULSEML%20v1.0&fontSize=32&fontColor=c2185b&fontAlignY=68&animation=fadeIn" width="100%"/>

</div>
