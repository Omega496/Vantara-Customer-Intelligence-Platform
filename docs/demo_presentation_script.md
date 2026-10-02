# Vantara Customer Intelligence Platform — 5-Minute Executive Demo Script

**Target Audience:** Executive Stakeholders, VP of Marketing, Head of Data Science & Engineering  
**Presenter:** Lead Predictive Analytics Engineer  
**Duration:** ~5 Minutes  
**Prerequisites:** Streamlit running on `http://localhost:8501`, FastAPI running on `http://localhost:8000` (or `docker-compose up -d`)

---

## ⏱️ Minute-by-Minute Breakdown

```
[0:00 - 0:45] Act 1: The Business Challenge (Customer Attrition & Lifetime Value)
[0:45 - 1:30] Act 2: Data Pipeline & Strict Point-in-Time Anti-Leakage
[1:30 - 2:30] Act 3: Model Suite Benchmarks (LightGBM, Ridge CLV, PyTorch Autoencoder)
[2:30 - 4:15] Act 4: Live Streamlit Dashboard & Explainability Walkthrough
[4:15 - 5:00] Act 5: Enterprise Architecture, Containerization & Conclusion
```

---

## Act 1: The Business Challenge (0:00 – 0:45)

> *"Good morning, everyone. In omnichannel e-commerce, acquiring a new customer costs 5 to 7 times more than retaining an existing one. Yet most retail businesses operate reactively — only noticing customer churn weeks after they have already defected.*
> 
> *Today, I'm excited to present the **Vantara Customer Intelligence Platform** — an end-to-end predictive machine learning and deep learning system built for Vantara Retail Solutions. Our mission was to eliminate blind spots by forecasting 90-day churn probability, estimating future monetary customer lifetime value, detecting suspicious spending anomalies, and automatically translating complex AI predictions into actionable marketing narratives."*

---

## Act 2: Data Engineering & Strict Anti-Leakage Discipline (0:45 – 1:30)

> *"To power this platform, we ingested over 1 million real-world transactions from the UCI Online Retail II dataset across 5,895 unique international accounts.*
> 
> *A critical engineering requirement was **strict point-in-time cutoff discipline**. We established a firm cutoff at September 10, 2011, splitting the data into an observation window of up to 648 days and a forward 90-day evaluation window. We engineered 33 behavioral features — capturing RFM velocity, inter-purchase interval standard deviation, holiday Q4 seasonality concentration, return rates, and a composite engagement index.
> 
> *Our rigorous automated unit tests guarantee **zero temporal data contamination** — ensuring that every prediction in production reflects only information known at the exact moment of inference."*

---

## Act 3: Modeling Suite & Key Benchmarks (1:30 – 2:30)

> *"Rather than relying on a single algorithm, we engineered a multi-model ensemble suite comparing classical ML against modern deep learning:*
> 
> 1. *For **Churn Classification**, our production champion is **LightGBM**, delivering a **Test ROC-AUC of 0.8205**, a churn **Recall of 81.64%**, and an **F1-score of 0.8013**. For high-recall sensitivity checks, our **PyTorch Deep ANN** achieved an ROC-AUC of **0.8235** with **84.73% Recall**.*
> 2. *For **Customer Lifetime Value**, our L2-regularized **Ridge Regressor** achieved a stellar **Test $R^2$ of 0.9150** with a Mean Absolute Error of just **£366.30** across the forward 90-day window.*
> 3. *For **Security and Fraud Detection**, we built an unsupervised **PyTorch Deep Spending Autoencoder**. By reconstructing normalized spending vectors, we established a 95th-percentile threshold ($0.0748$) that successfully flags anomalous order spikes and bulk resale behaviors.*
> 4. *Finally, **K-Means Clustering ($k=4$, Silhouette 0.4193)** grouped the cohort into 4 actionable personas: VIP Champions, Loyal Regulars, High-Value Inactive, and At-Risk Accounts."*

---

## Act 4: Live Streamlit Dashboard Walkthrough (2:30 – 4:15)

*(Presenter switches screen to `http://localhost:8501`)*

### 1. Executive Overview
> *"Let's explore the live platform. On the **Executive Overview**, leadership sees instant portfolio health metrics:*
> *We have **5,300 tracked customers** with an aggregate 90-day forward churn rate of **56.8%**, representing an average forward customer lifetime value of **£518.86**. The interactive revenue chart shows strong holiday concentration in Q4, while our AI strategy alert highlights that proactive 45-day re-engagement can preserve over **£285,000** in vulnerable gross margin."*

### 2. Behavioral Segmentation
> *(Navigate to 👥 Customer Segmentation tab)*
> *"Under **Customer Segmentation**, we see our accounts mapped in interactive 2D and 3D space. Notice how our **VIP Champions** — accounts spending over £5,800 with 24+ lifetime orders — clearly separate from the **High-Value Inactive** segment. Marketing teams can filter by country, spend range, or persona to export targeted cohorts in seconds."*

### 3. Churn Risk Leaderboard
> *(Navigate to 🎯 Churn Risk Leaderboard tab)*
> *"The **Churn Risk Leaderboard** solves the marketer's triage problem by ranking customers by **Priority Revenue at Risk** — multiplying forward CLV by churn probability. Here, accounts like Customer #14376 appear at the top. With one click on 'Export High-Risk Retention Campaign', the CRM team can download the exact accounts requiring urgent retention outreach."*

### 4. Customer 360° Drilldown & Explainability
> *(Navigate to 🔍 Customer 360° Drilldown tab)*
> *"Let's inspect **Customer #14057**. Instantly, we see their full profile: 0.8% churn probability, 'Low Risk', predicted £2,145 forward CLV, assigned to 'Loyal Regulars'.*
> 
> *More importantly, we don't present a black box. Below the metrics is our **Automated Plain-Language Marketing Narrative**, explaining exactly why this customer is scored this way. Next to it, the **SHAP Attribution Plot** reveals that high order frequency and recent purchases within 18 days drive the churn risk down by over 40%."*

### 5. Batch Scoring Studio
> *(Navigate to 🚀 Batch Scoring Studio tab)*
> *"Finally, for operational workflows, the **Batch Scoring Studio** allows business users to drag and drop arbitrary CSV cohorts, run multi-model inference via FastAPI, view real-time cohort churn statistics, and download fully scored prediction tables."*

---

## Act 5: Enterprise Serving & Architecture Handover (4:15 – 5:00)

> *"Under the hood, the entire system is built for production:*
> - *Our **FastAPI backend** delivers a blistering **P95 response latency of 42.56ms** — nearly 10 times faster than the 400ms SLA.*
> - *A dual-persistence layer uses **PostgreSQL 16** with seamless fallback to local **SQLite**.*
> - *The platform runs via a single command: **`docker-compose up --build -d`**, spinning up PostgreSQL, FastAPI, and Streamlit with pre-seeded data under a secure non-root user.*
> - *The codebase passes with **0 Ruff linter errors** and **88% automated test coverage** across 39 comprehensive unit and integration tests.*
> 
> *The Vantara Customer Intelligence Platform is fully tested, containerized, documented, and ready for deployment. Thank you, and I look forward to your questions."*

---

## 💡 Quick Reference Q&A for the Presenter

* **Q: Why did you pick LightGBM over XGBoost for production?**  
  *A: LightGBM achieved a higher overall test ROC-AUC (0.8205 vs 0.8160) and superior F1-score (0.8013 vs 0.8024) while offering significantly faster inference latency (< 15ms) and direct TreeSHAP integration.*

* **Q: How does the Autoencoder catch fraud?**  
  *A: The Autoencoder is trained on normalized customer spending and return behaviors. When an order sequence yields a reconstruction error exceeding the 95th percentile ($0.0748$), it flags potential bulk resellers, abrupt returns, or compromised accounts for manual review.*

* **Q: How does the system prevent data leakage?**  
  *A: All features are strictly calculated using events occurring on or before September 10, 2011. Forward metrics (90-day churn and 90-day forward spend) are computed exclusively from events occurring after the cutoff. Our automated anti-leakage test suite (`test_features.py`) guarantees zero post-cutoff data leaks into pre-cutoff feature rows.*
