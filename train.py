from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


from load_data import load_and_clean_data
X, target = load_and_clean_data('players.csv')
# --------------------------------------------------

# 1. The Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42, stratify=target)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 2. Setup the Preprocessing rules
categorical_features = ['POS']
numerical_features = [col for col in X.columns if col != 'POS'] 

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Apply the rules to our data!
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nData successfully preprocessed!")

# 4. Train the Classic ML Model (Random Forest)
print("\n--- Training Classic ML Model (Random Forest) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# 5. Predict and Evaluate
rf_predictions = rf_model.predict(X_test_processed)

print("Accuracy Score:", accuracy_score(y_test, rf_predictions))
print("\nDetailed Classification Report:")
print(classification_report(y_test, rf_predictions))