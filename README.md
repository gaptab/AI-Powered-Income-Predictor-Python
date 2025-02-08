# AI-Powered-Income-Predictor-Python

**Data Creation**

5,000 customer records are generated with age, education, occupation, spending habits, credit scores, work experience, and loan history.
The income (target variable) is randomly assigned between $25,000 and $200,000.

**Preprocessing**

Numerical features (e.g., age, credit score) are scaled using StandardScaler.
Categorical features (e.g., education, occupation) are encoded using OneHotEncoder.

**Training the Model**

A Random Forest Regressor is trained to predict annual salary income.
The dataset is split into 80% training and 20% testing.

**Model Evaluation**

R² Score (~75%) evaluates prediction quality.
RMSE (Root Mean Squared Error) measures how much predicted income deviates from the actual value.
