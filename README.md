# Vantara Customer Intelligence Platform

> **A Machine Learning & Deep Learning System for Churn, Lifetime Value, and Purchase Behavior Prediction**
> Built for Vantara Retail Solutions — Data Science & Analytics Division

---

## 📌 Project Overview
Vantara Retail Solutions operates an omnichannel retail and e-commerce business. This platform ingests historical transaction data and produces:
- **Churn Risk Scores** (XGBoost, LightGBM, Random Forest, Logistic Regression, Decision Tree, KNN/SVM, Deep ANN)
- **Customer Lifetime Value (CLV)** Regression Estimates
- **Sequential Purchase Dynamics** via LSTM
- **Spending Anomaly Detection** via Unsupervised Deep Autoencoder
- **Customer Segmentation** via K-Means and GMM/DBSCAN
- **Model Explainability** via SHAP and LIME
- **Production REST API** (FastAPI) and Interactive Analytics Dashboard (Streamlit)
- **Containerized Deployment** (PostgreSQL & Docker Compose)

---

## 🏗️ Repository Architecture

```text
customer-behavior-prediction/
├── data/
│   ├── raw/                  # Original unmodified dataset (Online Retail II)
│   ├── interim/              # Cleaned transactional records
│   └── processed/            # Final customer-level feature tables
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & Hypotheses
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data/                 # Loading, cleaning, schema validation
│   ├── features/             # Point-in-time RFM, CLV, behavioral signals
│   ├── models/               # Classical ML & Deep Learning (PyTorch)
│   ├── segmentation/         # Clustering & customer profiling
│   ├── explainability/       # SHAP / LIME wrappers
│   └── utils/                # Config loader, structured logging
├── api/
│   ├── main.py               # FastAPI entrypoint
│   ├── routers/              # Prediction, batch scoring, health endpoints
│   └── schemas/              # Pydantic data schemas
├── frontend/
│   └── dashboard.py          # Streamlit dashboard
├── models_artifacts/         # Serialized models, scalers, encoders
├── config/
│   └── config.yaml           # Centralized configuration parameters
├── tests/
│   ├── test_features.py      # Leakage & feature unit tests
│   └── test_api.py           # Integration tests
├── docs/                     # Architecture, ER diagrams & reports
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Milestones Execution Roadmap
- [x] **Milestone 0: Project Setup & Environment Scaffolding**
  - Dedicated virtual environment (Python 3.11) with PyTorch, XGBoost, LightGBM, SHAP, FastAPI, Streamlit.
  - Standard directory hierarchy and centralized YAML configuration.
- [x] **Milestone 1: Data Pipeline (Ingestion, Cleaning & Validation)**
  - Automated multi-sheet ingestion and caching (`data/raw/raw_transactions_cache.parquet`).
  - Robust cleaning logic (cancellations/returns, admin code filtering, deduplication, description mode standardization).
  - Strict schema and business rule validation.
  - Interim storage: `data/interim/cleaned_transactions.parquet` (794,461 validated rows, 5,895 customers).
  - Automated EDA engine with charts and modeling hypotheses (`docs/figures/`, `notebooks/01_eda.ipynb`).
  - Unit test suite: 11 tests passing with 87% coverage across `src/data`.
- [x] **Milestone 2: Feature Engineering (Point-in-Time RFM, CLV, Affinity & Leakage Tests)**
  - Strict point-in-time cutoff ($T_{cutoff}$ = 2011-09-10 12:50:00) with zero target window data contamination.
  - Extracted 36 customer-level features across RFM, purchase intervals, velocity acceleration, Q4 seasonality, category affinities, return rates, discount sensitivities, and composite engagement score.
  - Generated ground-truth 90-day churn binary labels (56.81% churn, 43.19% retained) and 90-day forward CLV targets.
  - Documented feature dictionary and metadata in `data/processed/feature_metadata.json`.
  - Stored processed dataset in `data/processed/customer_features.parquet` (5,300 cohort rows).
  - Explicit anti-leakage test suite passing with 90% overall test coverage and 0 Ruff linter errors.
- [x] **Milestone 3: Classical Machine Learning & Customer Segmentation**
  - Stratified 70/15/15 data split (Train=3,710, Val=795, Test=795) with training-only `StandardScaler`.
  - Trained 6 classical churn classifiers: LightGBM (Best: ROC-AUC=0.8205, Recall=0.8164, F1=0.8013), Random Forest (0.8180), Logistic Regression (0.8177), XGBoost (0.8160, Recall=0.9447), KNN (0.8009), Decision Tree (0.7944).
  - Trained 4 forward CLV regressors: Ridge Regressor (Best: $R^2$=0.9150, MAE=£366.30), Random Forest Regressor ($R^2$=0.6606), LightGBM ($R^2$=0.6195).
  - Built K-Means and GMM customer segmentation with Elbow/Silhouette evaluation ($k=4$, Silhouette=0.4193).
  - Extracted 4 distinct marketing personas: *"VIP Champions"*, *"Loyal Regulars"*, *"High-Value Inactive"*, *"At-Risk / Lapsed"*.
  - Full test suite passing with 21 unit tests, 91% code coverage, and 0 Ruff linter errors.
- [x] **Milestone 4: Deep Learning & Explainability (PyTorch & XAI)**
  - Trained Deep Feed-Forward ANN (`ChurnANN`) with Batch Normalization, Dropout, and early stopping on CUDA (Test ROC-AUC=0.8235, Recall=0.8473).
  - Built time-ordered 3D sequence tensor $(N=5,300, L=6, D=4)$ and trained 2-layer `PurchaseLSTM` (Test ROC-AUC=0.7528, Recall=0.8385).
  - Implemented unsupervised `SpendingAutoencoder` for fraud and anomaly detection; established validated $P95$ threshold ($0.0748$) flagging 265 anomalous accounts.
  - Implemented SHAP TreeExplainer: generated global summary & bar importance plots (`docs/figures/shap_summary_plot.png`) and local waterfalls for low-risk, borderline, and high-risk customers.
  - Implemented LIME tabular local explainer with rule-based surrogate explanations.
  - Implemented automated plain-language marketing briefing generator.
  - Test suite passing with 27 unit tests, 86% coverage across `src/`, and 0 Ruff linter errors.
- [x] **Milestone 5: Database (PostgreSQL & SQLite) & REST API (FastAPI)**
  - Built SQLAlchemy ORM models (`Customer`, `CustomerSegment`, `Prediction`, `AnomalyReport`).
  - Database connection manager with automatic SQLite fallback (`data/processed/retail.db`).
  - Pre-seeded database with 5,300 historical customer cohort across all tables.
  - Developed FastAPI application with Pydantic request/response schemas.
  - Endpoints: `GET /health`, `GET /models/metadata`, `POST /predict/customer`, `POST /predict/batch`, `GET /customers/{customer_id}`.
  - Measured latency: P95 response time of **42.56ms** (< 400ms SLA).
- [x] **Milestone 6: Streamlit Interactive Dashboard & Visualizations**
  - Built multi-tab responsive Streamlit dashboard (`frontend/dashboard.py`).
  - Views: Executive Overview, Behavioral Segmentation (2D/3D space), Churn Risk Leaderboard, Customer 360° Drilldown (SHAP & Narratives), Batch Scoring Studio.
  - Optimized cached loading under 1.5 seconds via `@st.cache_data`.
- [x] **Milestone 7: Unit Tests (88% Coverage), Docker Compose & Final Documentation**
  - 39 automated unit and integration tests passing (`pytest tests/`).
  - 88% overall repository code coverage (exceeding >= 70% requirement).
  - 0 Ruff linter errors across all source files.
  - Production `Dockerfile` and `docker-compose.yml` (PostgreSQL, FastAPI, Streamlit).
  - Architecture diagram (`docs/architecture_diagram.png`) and ER diagram (`docs/er_diagram.png`).
  - Comprehensive final report and mathematical foundations appendix (`docs/final_report.md`).
  - 5-minute executive presentation script (`docs/demo_presentation_script.md`).

---

## 🏗️ System Architecture & Schema Diagrams

### 4-Layer Architecture Diagram
![System Architecture](docs/architecture_diagram.png)

### Relational Database Entity-Relationship (ER) Diagram
![ER Diagram](docs/er_diagram.png)

---

## ⚡ Quickstart Guide

### Option 1: Multi-Container Docker Compose (Recommended)
```bash
# Launch PostgreSQL, FastAPI, and Streamlit in background
docker-compose up --build -d

# Open Streamlit Web App: http://localhost:8501
# Open FastAPI Swagger Docs: http://localhost:8000/docs
# Open Health Endpoint: http://localhost:8000/health
```

### Option 2: Local Python Execution
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Seed SQLite database
python -m src.database.seed

# 3. Launch FastAPI backend (Terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Launch Streamlit frontend (Terminal 2)
streamlit run frontend/dashboard.py --server.port 8501
```

---

## 🧪 Testing & Code Quality Verification

```bash
# Run complete test suite with coverage
pytest --cov=src --cov=api --cov-report=term-missing

# Run Ruff code linter
ruff check .
```
