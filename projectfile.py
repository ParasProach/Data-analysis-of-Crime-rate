
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv("python dataset.csv")


df = data.copy()



print("Dataset Shape:", df.shape)
print("\nColumn Info:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())




numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())


categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())


df["REPORT_DAT"] = pd.to_datetime(df["REPORT_DAT"])
df["START_DATE"] = pd.to_datetime(df["START_DATE"])
df["END_DATE"] = pd.to_datetime(df["END_DATE"], errors='coerce')


plt.figure(figsize=(8,5))
sns.countplot(data=df, x="SHIFT", order=df["SHIFT"].value_counts().index)
plt.title("Crime Count by Shift")
plt.xlabel("Shift")
plt.ylabel("Number of Crimes")
plt.show()



plt.figure(figsize=(8,5))
sns.histplot(df["LATITUDE"], bins=30, kde=True)
plt.title("Distribution of Latitude")
plt.show()



plt.figure(figsize=(6,6))
df["METHOD"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Crime Method Distribution")
plt.ylabel("")
plt.show()



plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="SHIFT", y="LONGITUDE")
plt.title("Longitude Distribution Across Shifts")
plt.show()


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


df["Month"] = df["REPORT_DAT"].dt.to_period("M").astype(str)
monthly_crimes = df["Month"].value_counts().sort_index()
plt.figure(figsize=(10,5))
monthly_crimes.plot(marker='o')
plt.title("Monthly Trend of Reported Crimes")
plt.xlabel("Month")
plt.ylabel("Crime Count")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Let's predict LATITUDE based on LONGITUDE
X = df[["LONGITUDE"]]
y = df["LATITUDE"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print("\nR² Score (LATITUDE vs LONGITUDE):", round(score, 4))


plt.figure(figsize=(8,5))
sns.scatterplot(x=X_test["LONGITUDE"], y=y_test, label="Actual")
sns.lineplot(x=X_test["LONGITUDE"], y=y_pred, color="red", label="Predicted")
plt.title("Regression: Latitude vs Longitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
