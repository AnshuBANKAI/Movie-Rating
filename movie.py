import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("movie.csv", encoding='latin1')
print("Initial Data Sample:")
print(df.head())

# Select and clean necessary columns
df = df[['Genre', 'Director', 'Actor 1', 'Rating', 'Votes', 'Year', 'Duration', 'Actor 2', 'Actor 3']]

# Convert columns to numeric
df['Votes'] = pd.to_numeric(df['Votes'].astype(str).str.replace(',', ''), errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

# Drop rows with missing numeric values
df.dropna(subset=['Rating', 'Votes', 'Year', 'Duration'], inplace=True)

# Impute missing categorical values
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']] = cat_imputer.fit_transform(
    df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
)

# Encode categorical features with separate encoders
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor1 = LabelEncoder()
le_actor2 = LabelEncoder()
le_actor3 = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actor 1'] = le_actor1.fit_transform(df['Actor 1'])
df['Actor 2'] = le_actor2.fit_transform(df['Actor 2'])
df['Actor 3'] = le_actor3.fit_transform(df['Actor 3'])

# Prepare features and target
X = df.drop('Rating', axis=1)
y = df['Rating']

# Scale numeric features (optional for tree models, useful for others)
scaler = StandardScaler()
X[['Votes', 'Duration']] = scaler.fit_transform(X[['Votes', 'Duration']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("Test R2 Score:", r2_score(y_test, y_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-validated R2 scores:", cv_scores)
print("Mean CV R2 Score:", np.mean(cv_scores))

# --- Helper function for label transformation ---
def safe_transform(le, label):
    if label in le.classes_:
        return le.transform([label])[0]
    else:
        print(f"Warning: '{label}' not found. Using fallback: '{le.classes_[0]}'")
        return le.transform([le.classes_[0]])[0]

# Create a sample movie input
sample = pd.DataFrame([{
    'Genre': safe_transform(le_genre, 'Action'),
    'Director': safe_transform(le_director, 'Christopher Nolan'),
    'Actor 1': safe_transform(le_actor1, 'Leonardo DiCaprio'),
    'Actor 2': safe_transform(le_actor2, 'Joseph Gordon-Levitt'),
    'Actor 3': safe_transform(le_actor3, 'Elliot Page'),
    'Votes': 1500000,
    'Year': 2010,
    'Duration': 148
}])

# Scale numeric features in the sample
sample[['Votes', 'Duration']] = scaler.transform(sample[['Votes', 'Duration']])

# Ensure column order matches training data
sample = sample[X.columns]

# Predict rating
predicted_rating = model.predict(sample)[0]
print(f"Predicted Rating for Sample Movie: {predicted_rating:.2f}")

# Plot feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Print top 5 important features
top_features = sorted(zip(importances, features), reverse=True)
print("\nTop 5 Important Features:")
for imp, feat in top_features[:5]:
    print(f"{feat}: {imp:.4f}")
