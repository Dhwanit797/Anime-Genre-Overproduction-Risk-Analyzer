# 🎌 Anime Genre Overproduction Risk Analyzer

## 📌 Description

A data science project that analyzes long-term anime genre trends to detect **overproduction risk** — situations where popularity rises while quality declines due to excessive production.

Using **Pandas + Machine Learning**, the project identifies genres that are likely being overproduced based on audience growth, score trends, and production volume over time.

---

## 🚀 Key Insights

* Detects **fake hype cycles**: popularity ↑ but quality ↓
* Identifies genres prone to **overproduction after breakout success**
* Assigns an **overproduction risk score** to each genre-year
* Highlights long-term genre sustainability vs short-term trends

---

## 🧠 Approach

1. Extract release year from messy date fields
2. Normalize popularity and scores **year-wise**
3. Aggregate data at **genre–year level**
4. Engineer trend features:

   * YoY production growth
   * 3-year rolling popularity & score trends
5. Define overproduction logic:

   * Popularity ↑
   * Scores ↓
   * Production ↑
6. Train a **Logistic Regression (imbalanced-aware)** model
7. Generate probabilistic **risk scores**
8. Visualize genre behavior and lifecycle patterns

---

## 📊 Visualizations

* Overproduction risk distribution
* Popularity vs quality drift (trend quadrant)
* Genre-level overproduction leaderboard
* Case study: genre lifecycle over time (e.g., Drama)

All visuals are generated directly from the analysis pipeline.

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python anime_overproduction_analyzer.py
```

---

## ⚠️ Notes

* Dataset sourced from **MyAnimeList (Kaggle)**
* AI assistance was used **only for visualization guidance**, not for analysis logic
* All feature engineering and modeling decisions were made manually

---

## 🔮 Future Scope

* Time-series forecasting of genre saturation
* Studio-level overproduction analysis
* Multi-label genre interaction effects
* Replace logistic regression with tree-based models

---

## 👤 Author

**Dhwanit**
Computer Engineering Student
Focused on data analysis, ML, and system-level thinking
