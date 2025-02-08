import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Step 1: Generate Dummy Data
np.random.seed(42)

num_samples = 5000
data = pd.DataFrame({
    'age': np.random.randint(20, 65, num_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_samples),
    'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist', 'Lawyer', 'Manager'], num_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], num_samples),
    'spending_score': np.random.randint(1, 100, num_samples),  # Hypothetical spending behavior
    'credit_score': np.random.randint(300, 850, num_samples),
    'work_experience': np.random.randint(0, 40, num_samples),
    'loan_status': np.random.choice(['No Loan', 'Small Loan', 'Large Loan'], num_samples),
    'income': np.random.randint(25000, 200000, num_samples)  # Target variable
})

# Step 2: Define Features and Target Variable
X = data.drop(columns=['income'])
y = data['income']

# Step 3: Preprocessing (Handling Categorical and Numerical Features)
numerical_features = ['age', 'spending_score', 'credit_score', 'work_experience']
categorical_features = ['education', 'occupation', 'marital_status', 'loan_status']

# Creating Transformers
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Column Transformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_features),
    ('cat', cat_transformer, categorical_features)
])

# Step 4: Build and Train Random Forest Regressor Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model R² Score: {r2 * 100:.2f}%")
print(f"Model RMSE: ${rmse:.2f}")

# Step 6: Save Model & Data for Deployment
data.to_csv("customer_income_dataset.csv", index=False)
joblib.dump(model_pipeline, "income_predictor_model.pkl")

print("Dataset and Model Saved Successfully!")
