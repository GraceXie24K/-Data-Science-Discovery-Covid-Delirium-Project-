import numpy as np
import pandas as pd
import os

def load_table(path, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, **kwargs)
    if ext in ('.xls', '.xlsx'):
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported extension: {ext}")

demographic_df = load_table('/users/audreysu/AudreyCovidProject/delirium cohort demographics.xlsx', engine='openpyxl')

demographic_df = demographic_df.head(128)

demographic_df = demographic_df.rename(columns={"Delirium at any time during hospitalization": "Delirium"})

target = "Delirium"

demographic_df = demographic_df.dropna(subset=[target]).reset_index(drop=True)
print("Shape after dropping rows with missing target:", demographic_df.shape)

# 2️⃣ Identify categorical and numeric columns (exclude target)
cat_cols = demographic_df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_impute_cols = [c for c in cat_cols if c != target]

num_cols = demographic_df.select_dtypes(include=[np.number]).columns.tolist()
num_impute_cols = [c for c in num_cols if c != target]

print("Categorical columns (will fill mode):", cat_impute_cols)
print("Numeric columns (will impute mean):", num_impute_cols)

# 3️⃣ Fill categorical NaNs with mode (most frequent value)
for c in cat_impute_cols:
    mode_vals = demographic_df[c].mode(dropna=True)
    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
    demographic_df[c] = demographic_df[c].fillna(fill_val)

# 4️⃣ Impute numeric NaNs with mean, only if there are rows and columns
if num_impute_cols and len(demographic_df) > 0:
    # Filter to only include columns that actually exist in combined_df
    existing_num_cols = [col for col in num_impute_cols if col in demographic_df.columns]
    
    if existing_num_cols:
        # Use a safer approach - impute one column at a time
        for col in existing_num_cols:
            demographic_df[col] = demographic_df[col].fillna(demographic_df[col].mean())
        print(f"Imputed {len(existing_num_cols)} numeric columns using fillna")
    else:
        print("No existing numeric columns found for imputation")
else:
    print("Skipping numeric imputation: no numeric columns or no rows.")


# 5️⃣ Optional: double-check no NaNs remain
missing_counts = demographic_df.isna().sum()
print("Missing values after imputation:\n", missing_counts[missing_counts > 0])

demographic_df = demographic_df.drop(columns = ["Conv Plasma Date Started", "Conv Plasma HD ended"])


from imblearn.over_sampling import SMOTE

# Separate features and target
X = demographic_df.drop(columns=[target])
y = demographic_df[target]

# One-hot encode categorical features
cat_cols_to_encode = [col for col in cat_impute_cols if col in X.columns]
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)

# Sanitize column names for LightGBM
import re
X_encoded.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in X_encoded.columns]

# Apply SMOTE
try:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    # Convert back to DataFrame
    oversampled_df = pd.DataFrame(X_resampled, columns=X_encoded.columns)
    oversampled_df[target] = y_resampled

    print("\nShape of oversampled data:", oversampled_df.shape)
    print("\nValue counts of target variable in oversampled data:")
    print(oversampled_df[target].value_counts())

except ImportError:
    print("\nPlease install the 'imbalanced-learn' library to use SMOTE.")
    print("You can install it by running: pip install imbalanced-learn")
    oversampled_df = None

if oversampled_df is not None:
    try:
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import roc_curve, roc_auc_score
        import matplotlib.pyplot as plt
        import xgboost as xgb
        import lightgbm as lgb

        X = oversampled_df.drop(columns=[target])
        y = oversampled_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            'Logistic Regression': LogisticRegression(solver='liblinear'),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': xgb.XGBClassifier(eval_metric='logloss'),
            'KNN': KNeighborsClassifier(),
            'LightGBM': lgb.LGBMClassifier(),
            'Decision Tree': DecisionTreeClassifier()
        }

        param_grids = {
            'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'LightGBM': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'learning_rate': [0.05, 0.1]},
            'Decision Tree': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        }

        best_estimators = {}
        plt.figure(figsize=(10, 8))

        for name, model in models.items():
            print(f"\nTraining {name}...")
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_estimators[name] = grid_search.best_estimator_
            
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"Best AUC for {name}: {auc:.4f}")
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Models')
        plt.legend()
        plt.show()


        for name, model in best_estimators.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.show()

    except ImportError as e:
        print(f"\nPlease install the following library: {e.name}")
        print(f"You can install it by running: pip install {e.name}")
