                  ┌─────────────────────────────┐
                  │         Streamlit UI         │
                  │(Unified AI Platform Console) │
                  └──────────────┬───────────────┘
                                 │
               ┌────────────────┴────────────────┐
               │                                 │
       ┌───────▼────────┐               ┌────────▼────────┐
       │ RAG Assistant  │               │  Search Engine  │
       │ (LLM + Vector) │               │ (Semantic Embds)│
       └───────▲────────┘               └────────▲────────┘
               │                                 │
               └──────┬────────────┬────────────┘
                      │            │
              ┌───────▼───────┐   ┌▼───────────────┐
              │ Anomaly Detect │   │  MLflow + DVC  │
              │ (LSTM + API)   │   │  (MLOps Core)  │
              └───────────────┘   └────────────────┘



unified-ai-platform/
│
├── rag_assistant/
│   ├── retriever.py
│   ├── generate.py
│   └── config/
│
├── semantic_search/
│   ├── embed.py
│   ├── search.py
│
├── anomaly_detection/
│   ├── model_lstm_ae.py
│   ├── train.py
│   ├── infer.py
│   └── mlruns/
│
├── mlops/
│   ├── dvc.yaml
│   ├── params.yaml
│   ├── pipeline.py
│
├── streamlit_dashboard/
│   ├── app.py
│   └── components/
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── docker-compose.yml
│
├── .github/workflows/
│   └── ci_cd.yml        ← GitHub Actions for GitOps
│
└── README.md

Folder Summary
Folder	Purpose
rag_assistant/	All code related to RAG + LLM query answering
semantic_search/	Embedding & semantic retrieval service
anomaly_detection/	LSTM Autoencoder, training, inference
mlops/	DVC, MLflow, and CI/CD integration scripts
streamlit_dashboard/	Unified front-end dashboard
docker/	Docker setup for API and UI
.github/workflows/	GitOps automation using GitHub Actions



┌────────────────────────────┐
│        GitOps (CI/CD)      │  ← Deployment brain
│  — GitHub Actions, ArgoCD  │
└────────────┬───────────────┘
             │
┌────────────┴───────────────┐
│   MLOps Pipeline (DVC)     │  ← Orchestrates experiments, data & training
│  — DVC tracks data/models  │
└────────────┬───────────────┘
             │
┌────────────┴───────────────┐
│     MLflow Tracking        │  ← Tracks experiments, metrics, params
│  — Version registry + UI   │
└────────────┬───────────────┘
             │
┌────────────┴───────────────┐
│  Model Services & APIs     │  ← FastAPI, Streamlit, etc.
│  — Served via Docker/K8s   │
└────────────────────────────┘
