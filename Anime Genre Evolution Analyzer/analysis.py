import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Load & Clean Data
# -------------------------------

df = pd.read_csv("anime.csv")

df = df[
    ["MAL_ID", "Name", "Genres", "Score", "Members", "Favorites",
     "Popularity", "Aired", "Premiered", "Studios", "Type"]
]

# Extract release year
df["release_year"] = df["Premiered"].str.extract(r"(\d{4})")[0]
df["release_year"] = df["release_year"].fillna(
    df["Aired"].str.extract(r"(\d{4})")[0]
)
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

# Keep only TV anime
df = df[df["Type"] == "TV"]

# Numeric coercion
for col in ["Score", "Members", "Favorites"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Genres", "Score", "Members", "Favorites", "release_year"])

# Explode genres
df["Genres"] = df["Genres"].str.split(", ")
df = df.explode("Genres")

# -------------------------------
# 2. Normalization (Year-based)
# -------------------------------

df["members_norm"] = df["Members"] / df.groupby("release_year")["Members"].transform("mean")
df["score_norm"] = df["Score"] / df.groupby("release_year")["Score"].transform("mean")

# -------------------------------
# 3. Genre-Year Aggregation
# -------------------------------

genre_year = (
    df.groupby(["Genres", "release_year"])
    .agg(
        anime_count=("MAL_ID", "count"),
        avg_score=("Score", "mean"),
        avg_members=("Members", "mean"),
        total_members=("Members", "sum"),
        avg_favorites=("Favorites", "mean"),
        avg_score_norm=("score_norm", "mean"),
        avg_members_norm=("members_norm", "mean")
    )
    .reset_index()
)

# -------------------------------
# 4. Trend Engineering
# -------------------------------

genre_year["anime_count_yoy"] = (
    genre_year.groupby("Genres")["anime_count"].pct_change()
)

genre_year["members_3yr_roll"] = (
    genre_year.groupby("Genres")["avg_members"]
    .transform(lambda x: x.rolling(3).mean())
)

genre_year["score_3yr_roll"] = (
    genre_year.groupby("Genres")["avg_score"]
    .transform(lambda x: x.rolling(3).mean())
)

genre_year["members_trend"] = (
    genre_year.groupby("Genres")["members_3yr_roll"].diff()
)

genre_year["score_trend"] = (
    genre_year.groupby("Genres")["score_3yr_roll"].diff()
)

genre_year = genre_year.dropna()

# -------------------------------
# 5. Label: Overproduction
# -------------------------------

genre_year["overproduced"] = (
    (genre_year["members_trend"] >= 0) &
    (genre_year["score_trend"] <= 0) &
    (genre_year["anime_count_yoy"] >= 0)
).astype(int)

# -------------------------------
# 6. ML Dataset
# -------------------------------

features = [
    "anime_count",
    "anime_count_yoy",
    "avg_score_norm",
    "avg_members_norm",
    "score_trend",
    "members_trend"
]

X = genre_year[features]
y = genre_year["overproduced"]
indices = X.index

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 7. Logistic Regression (Balanced)
# -------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_probs = model.predict_proba(X_test_scaled)[:, 1]

# -------------------------------
# 8. Risk Scoring
# -------------------------------

threshold = 0.30
y_pred = (y_probs >= threshold).astype(int)

risk_df = genre_year.loc[idx_test].copy()
risk_df["overproduction_risk"] = y_probs
risk_df["risk_flag"] = y_pred

# -------------------------------
# 9. Evaluation
# -------------------------------

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -------------------------------
# 10. Genre-Level Risk
# -------------------------------

genre_risk = (
    risk_df.groupby("Genres")["overproduction_risk"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop Overproduction-Prone Genres")
print(genre_risk.head(10))

# -------------------------------
# 11. VISUALIZATIONS
# -------------------------------

# 1. Risk distribution
plt.figure()
plt.hist(risk_df["overproduction_risk"], bins=30)
plt.title("Distribution of Overproduction Risk Scores")
plt.xlabel("Risk Probability")
plt.ylabel("Count")
plt.show()

# 2. Popularity vs Quality Trend
plt.figure()
plt.scatter(
    genre_year["members_trend"],
    genre_year["score_trend"],
    alpha=0.5
)
plt.axhline(0)
plt.axvline(0)
plt.xlabel("Members Trend (Popularity)")
plt.ylabel("Score Trend (Quality)")
plt.title("Popularity vs Quality Drift")
plt.show()

# 3. Genre Risk Leaderboard
plt.figure()
plt.barh(genre_risk.head(10).index, genre_risk.head(10).values)
plt.xlabel("Average Overproduction Risk")
plt.title("Top Genres by Overproduction Risk")
plt.show()

# 4. Genre Case Study
genre_name = "Drama"

genre_ts = genre_year[genre_year["Genres"] == genre_name]

plt.figure()
plt.plot(genre_ts["release_year"], genre_ts["avg_members"], label="Avg Members")
plt.plot(genre_ts["release_year"], genre_ts["avg_score"], label="Avg Score")
plt.xlabel("Year")
plt.ylabel("Value")
plt.title(f"{genre_name}: Popularity vs Quality Over Time")
plt.legend()
plt.show()
