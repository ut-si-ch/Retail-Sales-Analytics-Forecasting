#  Retail Sales Analytics & Forecasting (Walmart Case Study)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest%2C%20XGBoost%2C%20Clustering-green)
![EDA](https://img.shields.io/badge/EDA-Seaborn%2C%20Matplotlib-orange)
![TimeSeries](https://img.shields.io/badge/Forecasting-ARIMA%2C%20Prophet%2C%20ML-purple)

> **Goal:** Optimize retail store performance, forecast demand, detect anomalies, and enhance customer experience through segmentation and personalized strategies.

---

##  Table of Contents
1. [Project Overview](#--project-overview)
2. [Dataset](#-dataset)
3. [Business Problem](#-business-problem)
4. [End-to-End Workflow](#-end-to-end-workflow)
    - [1. Project Setup](#1-project-setup)
    - [2. Data Ingestion & Exploration](#2-data-ingestion--exploration)
    - [3. Data Preprocessing & Feature Engineering](#3-data-preprocessing--feature-engineering)
    - [4. Exploratory Data Analysis](#4-exploratory-data-analysis)
    - [5. Anomaly Detection](#5-anomaly-detection)
    - [6. Customer Segmentation](#6-customer-segmentation)
    - [7. Market Basket Analysis](#7-market-basket-analysis)
    - [8. Demand Forecasting](#8-demand-forecasting)
    - [9. Impact of External Factors](#9-impact-of-external-factors)
    - [10. Personalization & Strategy](#10-personalization--strategy)
    - [11. Visualization & Reporting](#11-visualization--reporting)
5. [Key Results](#-key-results)
6. [Final Conclusions](#-final-conclusions)
7. [Future Scope](#-future-scope)

---

## Project Overview
Retail businesses face challenges in managing **seasonality, external shocks, and customer preferences**.  
This project applies **data analytics + ML** to:
- Detect sales anomalies  
- Forecast demand  
- Segment stores/departments  
- Analyze product affinities (market basket)  
- Understand macroeconomic impacts  

---

## Dataset
The dataset includes:
- **Sales data:** Weekly sales by store & department  
- **Markdowns:** Promotional markdowns (MarkDown1–5)  
- **External factors:** CPI, Fuel Price, Unemployment  
- **Metadata:** Store size, type, holiday indicators  

---

## Business Problem
- How can Walmart optimize **inventory & operations** across stores?  
- Which **departments co-occur** frequently → cross-selling opportunities?  
- What role do **external factors** (fuel, unemployment, CPI) play in sales fluctuations?  
- How can we build a **forecasting engine** to plan for holiday seasons?  

---

## End-to-End Workflow

### 1. Project Setup
- Virtual environment setup using Anaconda  
- Libraries installed: pandas, numpy, matplotlib, seaborn, sklearn, mlxtend, xgboost  

---

### 2. Data Ingestion & Exploration
- Merged datasets on **Store, Date, IsHoliday**  
- Explored nulls, outliers, distributions  
- Visualized sales trends, holiday effects, and correlations  

---

### 3. Data Preprocessing & Feature Engineering
- Imputed missing values (median for MarkDown, CPI, Unemployment)  
- Converted Date → Year, Month, Week, Quarter, DayOfWeek  
- Added lag features, rolling mean & std (4, 12 weeks)  
- Encoded categorical variables (Store Type)  

---

### 4. Exploratory Data Analysis
- **Sales Trends Over Time**
   
    - **Total Weekly Sales**: Strong seasonality with clear Q4 spikes in 2011 and 2012.<img width="1142" height="580" alt="Total Weekly Sales " src="https://github.com/user-attachments/assets/8646cc48-fbf2-4f1a-9c84-c7f3aa2323ab" />
    - **Monthly Sales Trend**: Peaks in November and December confirm holiday-driven demand.<img width="1153" height="580" alt="download" src="https://github.com/user-attachments/assets/966f1988-53cb-4b3c-a536-09de45a708aa" />
    - **Time Series Decomposition**: Seasonal and trend components extracted; residuals show random noise.<img width="1133" height="757" alt="download" src="https://github.com/user-attachments/assets/bd54671e-c323-494e-a85f-c994b6fe844d" />


-  **Store-Level Performance**:
  
    -  **Weekly Sales by Store (Top 10)**: Stores 1 and 2 consistently outperform others.<img width="1317" height="624" alt="download" src="https://github.com/user-attachments/assets/1efab117-aa41-43a7-a50d-20b8aadb6d2e" />

    - **Total Sales by Store**: Store 4, 20, 13, 14 leads overall; Store 33,5 among the lowest.<img width="1153" height="547" alt="download" src="https://github.com/user-attachments/assets/419e7253-91d7-405e-bcc9-be09ce20b32a" />

    - **Average Weekly Sales by Store Type**: Type A stores outperform B and C.<img width="553" height="470" alt="download" src="https://github.com/user-attachments/assets/3b409c71-4111-4663-beba-2a56e1aff070" />

    - **Sales by Store Size Category**: Larger stores generate higher average weekly sales.<img width="630" height="470" alt="download" src="https://github.com/user-attachments/assets/87876fbb-00f5-4314-a297-1f22ce140a6d" />

   
-  **Holiday Impact**:

    - **Holiday vs Non-Holiday Sales (Bar Chart)**: Holiday weeks show slightly higher average sales.<img width="553" height="466" alt="download" src="https://github.com/user-attachments/assets/1e09432f-069a-4fe5-9a05-6843e0cf5603" />

    - **Holiday vs Non-Holiday Sales (Box Plot)**: Wider spread and higher median during holidays.<img width="691" height="547" alt="download" src="https://github.com/user-attachments/assets/8f4db9e1-2913-4b26-a1d0-2830149fe59f" />

- **Markdown Strategy Analysis**
  
  - **Weekly Sales vs MarkDowns (Scatter Grid)**: MarkDown1 and MarkDown4 show broader influence.<img width="1190" height="590" alt="download" src="https://github.com/user-attachments/assets/3dd629d4-a45a-4317-8906-913da740211d" />

  - **MarkDown Correlation Heatmap**: MarkDown4 has the strongest correlation with sales (0.073); MarkDown1 and MarkDown4 are highly correlated (0.84).<img width="696" height="524" alt="download" src="https://github.com/user-attachments/assets/cca4d7ee-93de-4cbd-9409-7984349baa60" />
  
-  **Department-Level Insights**:

   - **Top 10 Departments by Total Sales**: Departments 92, 95, and 13 dominate overall sales.<img width="832" height="543" alt="download" src="https://github.com/user-attachments/assets/034f2fd7-432a-4cbe-86d2-d5253b0537a4" />

- **External Factors & Economic Indicators**

    - **Sales vs Fuel Price, CPI, Unemployment (Scatter Plots)**: No strong linear relationship observed.<img width="1789" height="489" alt="download" src="https://github.com/user-attachments/assets/0ff4283a-c7b3-44e1-a9de-408be0f0a6a6" />
    - **Pair Plot (Sales vs External Factors)**: Weak visual patterns; distributions show multimodality.<img width="986" height="1023" alt="download" src="https://github.com/user-attachments/assets/78694f57-747e-4822-accf-fa498d97798e" />

  - **Correlation Heatmap (External Factors)**: All correlations with sales are weak; CPI–Unemployment shows moderate negative correlation (-0.3).<img width="637" height="526" alt="download" src="https://github.com/user-attachments/assets/a9074d49-ca65-4ae1-a1df-b00d5e31ea7e" />

- **Forecasting Models**
   - **ARIMA Forecast**: Captures overall trend but smooths out seasonal spikes.<img width="1004" height="526" alt="download" src="https://github.com/user-attachments/assets/69a9ca56-58cd-4a96-b230-468c8eba58a9" />
   - **SARIMA Forecast**: Accurately models seasonal peaks and periodic fluctuations.<img width="1004" height="526" alt="download" src="https://github.com/user-attachments/assets/fb94b1ce-6dcd-4491-8857-8cad6c2b6bd8" />

  - **Prophet Forecast**: Captures trend and seasonality with confidence intervals; strong fit for retail cycles.
  - **Prophet Seasonal Components**: Weekly peaks around Sunday; yearly patterns align with holidays; steady upward trend.<img width="886" height="1190" alt="download" src="https://github.com/user-attachments/assets/7ea41016-614d-4671-a7ee-d523c3d784ba" />

---

### 5. Anomaly Detection
- **Statistical**: Z-score, IQR  
- **ML-based**: Isolation Forest vs Autoencoder  
    - Autoencoder performed better (F1 = 0.20 vs IF F1 = 0.12)  
- Identified unusual sales spikes around holidays/events  

---

### 6. Customer Segmentation
- K-Means clustering (k=10 optimal) → segmented stores/departments  
- Silhouette score validated segmentation quality  

---

### 7. Market Basket Analysis
- Applied **FP-Growth** on department-level sales → found frequent itemsets  
- Key co-occurring pairs:  
  - (91, 92, 95)  
  - (1, 2, 3, 21)  
  - (7, 13, 14, 16)  
- Actionable for **cross-selling & bundle offers**  

---

### 8. Demand Forecasting
- **Models Tested**: Linear Regression, RF, GBM, XGBoost  
- **Results**:  
  - Linear Regression → RMSE = 3400  
  - Random Forest → RMSE = 2986, R² = 0.98 - Champion Model  
  - GBM → RMSE = 3202  
  - XGBoost → RMSE = 3152  
- RF selected as best-performing baseline  

---

### 9. Impact of External Factors
- Regression model coefficients:  
  - CPI → -19 impact  
  - Fuel Price → -308 impact  
  - Unemployment → -435 impact ❗  
- **Scenario Analysis**:  
  - Fuel +10% → Sales ↓ 104  
  - Unemployment +1% → Sales ↓ 435  
  - CPI +2 → Sales ↓ 38  

---

### 10. Personalization & Strategy
- Store-specific markdown strategies  
- Targeted campaigns for clusters  
- Bundle promotions for frequent itemsets  

---

### 11. Visualization & Reporting
- EDA plots (trend, seasonality, holiday vs non-holiday, heatmaps)  
- Store/department bar charts  
- Scenario simulation outputs  
- Ready for dashboard integration (Streamlit / Power BI)  

---

## Key Results
-  **Best Model**: Random Forest (RMSE ≈ 2986, R² = 0.98)  
-  **Anomalies Detected**: Holiday/event-driven spikes  
-  **Top Item Pairs**: (91,92,95), (1,2,3,21), (7,13,14,16)  
-  **External Factors**: Unemployment strongest negative impact  
-  **Segmentation**: 10 store/department clusters identified  

---

## Final Conclusions
- Holidays are critical demand drivers → inventory must be optimized  
- Unemployment strongly reduces sales → proactive discounting needed  
- Department cross-selling opportunities → bundle offers boost revenue  
- Forecasting pipeline (RF baseline) is reliable for short-term planning  

---

## Future Scope
- Deploy deep learning (LSTM/Transformers) for long-term forecasting  
- Real-time forecasting using Kafka + Spark  
- Customer-level personalization using recommender systems  
- Stress-test extreme economic scenarios  
- Build interactive dashboards for business users  

---

📎 **Linkedin Profile:** *[[Uttam Singh Chaudhary]](https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b)*  

