import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
import warnings
import contextlib
import os
import shap
import joblib

# Suppress all warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# --- Robust loader for CSV / Excel ---
def load_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    if ext in ('.xls', '.xlsx'):
        return pd.read_excel(path, engine='openpyxl')
    raise ValueError(f"Unsupported extension: {ext}")

# Enhanced data preprocessing
def transform_colname(col):
    match = re.search(r"MVIR1HS(\d+)", col)
    if match:
        code = match.group(1)
        return 1000 + int(code)
    return col

print("Loading data...")
admission_df = load_table('/users/audreysu/AudreyCovidProject/admission_norm_gene_exp_df.csv')
demographic_df = load_table('/users/audreysu/AudreyCovidProject/delirium cohort demographics.xlsx')
print((demographic_df.shape))
gene_symbols = load_table('/users/audreysu/AudreyCovidProject/gene_symbols.csv')

# Data preprocessing
admission_df = admission_df.rename(columns={admission_df.columns[0]: 'SampleID'})
gene_symbols = gene_symbols.rename(columns={gene_symbols.columns[0]: 'SampleID'})

gene_map = dict(zip(gene_symbols["gene_ids"], gene_symbols["gene_symbols"]))
admission_df["Sample_ID_mapped"] = admission_df["SampleID"].map(gene_map).fillna(admission_df["SampleID"])

transcriptomic_df = admission_df.drop(columns=['SampleID'])
transcriptomic_df.columns = [transform_colname(col) for col in transcriptomic_df.columns]

demographic_df = demographic_df.head(128)
# demographic_df = demographic_df

transcriptomic_df = transcriptomic_df.set_index("Sample_ID_mapped")
transcriptomic_df = transcriptomic_df.T
transcriptomic_df = transcriptomic_df.reset_index().rename(columns={'index': 'Master Record ID'})

merged_df = pd.merge(demographic_df, transcriptomic_df, on='Master Record ID', how='outer')
merged_df = merged_df.rename(columns={"Delirium at any time during hospitalization": "Delirium"})

target = "Delirium"
merged_df = merged_df.dropna(subset=[target]).reset_index(drop=True)

# print(merged_df)

print(f"Original data shape: {merged_df.shape}")
print(f"Target distribution: {merged_df[target].value_counts()}")

# Handle missing values more intelligently
cat_cols = merged_df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_impute_cols = [c for c in cat_cols if c != target]

num_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
num_impute_cols = [c for c in num_cols if c != target]

# Fill categorical NaNs
for c in cat_impute_cols:
    mode_vals = merged_df[c].mode(dropna=True)
    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
    merged_df[c] = merged_df[c].fillna(fill_val)

# Handle missing numeric values with more sophisticated approach
missing_percentage = merged_df[num_impute_cols].isna().sum() / len(merged_df)
cols_to_drop = missing_percentage[missing_percentage > 0.3].index.tolist()  # Lowered threshold

if cols_to_drop:
    merged_df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} columns with >30% missing values.")
    num_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    num_impute_cols = [c for c in num_cols if c != target]

# Impute remaining numeric NaNs with median
if num_impute_cols and len(merged_df) > 0:
    for col in num_impute_cols:
        if merged_df[col].isna().any():
            median_val = merged_df[col].median()
            merged_df[col].fillna(median_val, inplace=True)

print(merged_df)

if "Diagnosis" in merged_df.columns:
    print("Removing 'Diagnosis' column as it's nearly identical to Delirium target")
    merged_df = merged_df.drop(columns=["Diagnosis"])

if "WHO Scale" in merged_df.columns:
    print("Removing 'WHO Scale' column as it's nearly identical to Delirium target")
    merged_df = merged_df.drop(columns=["WHO Scale"])

if "Subject" in merged_df.columns:
    print("Removing 'Subject' column as it's random to Delirium target")
    merged_df = merged_df.drop(columns=["Subject"])

# Check similarity between "Late_del" and "Delirium" target
if "Late_del" in merged_df.columns:
    print("Removing 'Late_del' column as it's correlated to Delirium target")
    merged_df = merged_df.drop(columns=["Late_del"])
# Data preparation
X = merged_df.drop(columns=[target, 'Master Record ID'])
y = merged_df[target]

# One-hot encode categorical features
X_cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
X_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X = pd.get_dummies(X, columns=X_cat_cols, drop_first=True)

print(f"Features after encoding: {X.shape[1]}")

# Enhanced feature selection strategies
def advanced_feature_selection(X, y, n_features=100):
    """Multiple feature selection strategies"""
    print(f"Performing advanced feature selection to select {n_features} features...")
    
    # Strategy 1: Variance threshold (remove low variance features)
    from sklearn.feature_selection import VarianceThreshold
    selector_var = VarianceThreshold(threshold=0.01)
    X_var_selected = selector_var.fit_transform(X)
    var_features = X.columns[selector_var.get_support()]
    print(f"Variance threshold selected {len(var_features)} features")
    
    # Strategy 2: Mutual information (captures non-linear relationships)
    if len(var_features) > n_features:
        mi_scores = mutual_info_classif(X[var_features], y, random_state=42)
        mi_features = var_features[np.argsort(mi_scores)[-n_features:]]
        print(f"Mutual info selected {len(mi_features)} features")
        return mi_features
    else:
        return var_features

# Apply advanced feature selection
selected_features = advanced_feature_selection(X, y, n_features=200)

# Print all selected features without truncation
print(f"Selected features ({len(selected_features)} total):")
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't wrap long strings
pd.set_option('display.max_colwidth', None)  # Show full content of each cell

# Convert to list and print each feature
selected_features_list = selected_features.tolist()
for i, feature in enumerate(selected_features_list):
    print(f"{i+1:3d}. {feature}")

X_selected = X[selected_features]

print(f"Final feature set: {X_selected.shape[1]} features")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Enhanced scaling - try multiple scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MinMaxScaler': MinMaxScaler()
}

# Test different scalers
best_scaler = None
best_score = 0

for scaler_name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Quick test with simple model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    avg_score = scores.mean()
    
    if avg_score > best_score:
        best_score = avg_score
        best_scaler = scaler_name
    
    print(f"{scaler_name}: CV AUC = {avg_score:.4f}")

print(f"Selected scaler: {best_scaler}")

# Apply best scaler
scaler = scalers[best_scaler]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Enhanced sampling strategies
sampling_strategies = {
    'SMOTE': SMOTE(random_state=42, k_neighbors=3),
    'ADASYN': ADASYN(random_state=42, n_neighbors=3),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
    'SMOTEENN': SMOTEENN(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42)
}

# Test different sampling strategies
best_sampler = None
best_sampler_score = 0

for sampler_name, sampler in sampling_strategies.items():
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        
        # Quick test
        lr = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(lr, X_resampled, y_resampled, cv=5, scoring='roc_auc')
        avg_score = scores.mean()
        
        if avg_score > best_sampler_score:
            best_sampler_score = avg_score
            best_sampler = sampler_name
        
        print(f"{sampler_name}: CV AUC = {avg_score:.4f}")
    except Exception as e:
        print(f"{sampler_name}: Failed - {e}")

print(f"Selected sampler: {best_sampler}")

# Apply best sampler
# Note: sampling will be performed inside the CV pipeline to avoid data leakage.
# Keep the selected sampler object for use inside the pipeline (do not pre-resample).
sampler = sampling_strategies[best_sampler]
print(f"Selected sampler (to be used inside CV pipeline): {best_sampler}")

# Enhanced model configurations with more sophisticated hyperparameters
models = {
    "Logistic Regression": {
        "model": LogisticRegression(solver='liblinear', random_state=42, max_iter=2000),
        "params": {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        }
    },
    "LightGBM": {
    "model": lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
    "params": {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 50, 100],
        'max_depth': [5, 10, 15, -1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0, 0.1, 1, 10]
    }
},
    "XGBoost": {
    "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
    "params": {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0, 0.1, 1, 10],
        'scale_pos_weight': [1, 3, 5, 10]  # Handle class imbalance
    }
},
    "SVM": {
        "model": SVC(probability=True, random_state=42, class_weight='balanced'),
        "params": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    }
}

# Enhanced training with cross-validation and ensemble methods
trained_models = {}
roc_curves_data = {}
cv_scores = {}

print("\n" + "="*60)
print("ENHANCED MODEL TRAINING AND OPTIMIZATION")
print("="*60)

for name, config in models.items():
    print(f"\n--- Training {name} ---")

    # Build an imblearn pipeline so scaling and sampling happen inside CV folds
    pipe_scaler = scaler
    pipe_sampler = sampler
    pipeline = ImbPipeline(steps=[('scaler', pipe_scaler), ('sampler', pipe_sampler), ('est', config['model'])])

    # Prefix estimator params for the pipeline
    est_params = {f"est__{k}": v for k, v in config['params'].items()}

    # Use a modest number of iterations to keep tuning time reasonable
    random_search = RandomizedSearchCV(
        pipeline,
        est_params,
    n_iter=30,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Fit on the original training set; sampler inside pipeline will resample during CV
    random_search.fit(X_train, y_train)

    best_pipeline = random_search.best_estimator_
    trained_models[name] = best_pipeline

    # Cross-validation score
    cv_score = random_search.best_score_
    cv_scores[name] = cv_score

    # Test set performance: pipeline will scale X_test appropriately; sampler is bypassed during predict
    try:
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        test_recall = recall_score(y_test, (y_pred_proba > 0.5).astype(int))
        test_precision = precision_score(y_test, (y_pred_proba > 0.5).astype(int))
        test_f1 = f1_score(y_test, (y_pred_proba > 0.5).astype(int))
    except Exception:
        test_auc = np.nan
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        test_recall = np.nan
        test_precision = np.nan
        test_f1 = np.nan

    roc_curves_data[name] = {"fpr": fpr, "tpr": tpr, "auc": test_auc, "recall": test_recall, "precision": test_precision, "f1": test_f1}

    print(f"Best parameters: {random_search.best_params_}")
    print(f"CV AUC: {cv_score:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Save the tuned pipeline
    try:
        os.makedirs('models', exist_ok=True)
        pipeline_path = f"models/{name.replace(' ', '_')}_pipeline.pkl"
        joblib.dump(best_pipeline, pipeline_path)
        roc_curves_data[name]['pipeline_path'] = pipeline_path
        print(f"Saved pipeline for {name} to {pipeline_path}")
    except Exception as e:
        print(f"Failed to save pipeline for {name}: {e}")

# Create ensemble models
print("\n--- Creating Ensemble Models ---")

# Voting Classifier (Soft Voting)
voting_classifier = VotingClassifier(
    estimators=[(name, model) for name, model in trained_models.items()],
    voting='soft'
)

voting_classifier.fit(X_train, y_train)
y_pred_voting = voting_classifier.predict_proba(X_test)[:, 1]
voting_auc = roc_auc_score(y_test, y_pred_voting)
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_pred_voting)
voting_recall = recall_score(y_test, (y_pred_voting > 0.5).astype(int))
voting_precision = precision_score(y_test, (y_pred_voting > 0.5).astype(int))
voting_f1 = f1_score(y_test, (y_pred_voting > 0.5).astype(int))
roc_curves_data["Voting Ensemble"] = {"fpr": fpr_voting, "tpr": tpr_voting, "auc": voting_auc, "recall": voting_recall, "precision": voting_precision, "f1": voting_f1}
trained_models["Voting Ensemble"] = voting_classifier  # Add to trained_models for SHAP analysis
print(f"Voting Ensemble AUC: {voting_auc:.4f}")
print(f"Voting Ensemble Recall: {voting_recall:.4f}")
print(f"Voting Ensemble Precision: {voting_precision:.4f}")
print(f"Voting Ensemble F1: {voting_f1:.4f}")

# Stacking Classifier
base_models = [(name, model) for name, model in trained_models.items()]
meta_model = LogisticRegression(random_state=42)

stacking_classifier = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stacking_classifier.fit(X_train, y_train)
y_pred_stacking = stacking_classifier.predict_proba(X_test)[:, 1]
stacking_auc = roc_auc_score(y_test, y_pred_stacking)
fpr_stacking, tpr_stacking, _ = roc_curve(y_test, y_pred_stacking)
stacking_recall = recall_score(y_test, (y_pred_stacking > 0.5).astype(int))
stacking_precision = precision_score(y_test, (y_pred_stacking > 0.5).astype(int))
stacking_f1 = f1_score(y_test, (y_pred_stacking > 0.5).astype(int))
roc_curves_data["Stacking Ensemble"] = {"fpr": fpr_stacking, "tpr": tpr_stacking, "auc": stacking_auc, "recall": stacking_recall, "precision": stacking_precision, "f1": stacking_f1}
trained_models["Stacking Ensemble"] = stacking_classifier  # Add to trained_models for SHAP analysis
print(f"Stacking Ensemble AUC: {stacking_auc:.4f}")
print(f"Stacking Ensemble Recall: {stacking_recall:.4f}")
print(f"Stacking Ensemble Precision: {stacking_precision:.4f}")
print(f"Stacking Ensemble F1: {stacking_f1:.4f}")

# Results summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame({
    'Model': list(roc_curves_data.keys()),
    'Test AUC': [data['auc'] for data in roc_curves_data.values()],
    'Recall': [data.get('recall', np.nan) for data in roc_curves_data.values()],
    'Precision': [data.get('precision', np.nan) for data in roc_curves_data.values()],
    'F1': [data.get('f1', np.nan) for data in roc_curves_data.values()],
    'CV AUC': [cv_scores.get(name, 'N/A') for name in roc_curves_data.keys()]
})

results_df = results_df.sort_values('Test AUC', ascending=False)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('graph2/model_performance_results.csv', index=False)

# Save per-model ROC points (FPR/TPR) to CSVs with x/y columns for downstream use
os.makedirs('graph2', exist_ok=True)
for name, data in roc_curves_data.items():
    try:
        fpr = data.get('fpr')
        tpr = data.get('tpr')
        if fpr is None or tpr is None:
            print(f"Warning: no ROC points available for {name}")
            continue
        roc_df = pd.DataFrame({'FPR': np.asarray(fpr), 'TPR': np.asarray(tpr)})
        roc_df['x'] = roc_df['FPR']
        roc_df['y'] = roc_df['TPR']
        safe_name = name.replace(' ', '_').replace('/', '_')
        roc_df.to_csv(f'graph2/{safe_name}_roc_points.csv', index=False)
        print(f"💾 Saved ROC points for {name} to graph2/{safe_name}_roc_points.csv")
    except Exception as e:
        print(f"❌ Failed to save ROC points for {name}: {e}")

# Enhanced ROC plotting
plt.figure(figsize=(12, 10))

# Color palette for better visualization
colors = plt.cm.Set3(np.linspace(0, 1, len(roc_curves_data)))

for i, (name, data) in enumerate(roc_curves_data.items()):
    plt.plot(data["fpr"], data["tpr"], 
             label=f'{name} (AUC = {data["auc"]:.4f})',
             color=colors[i], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison - Enhanced Models', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph2/enhanced_models_roc_curves.png', dpi=300, bbox_inches='tight')

# Individual ROC curves
for name, data in roc_curves_data.items():
    plt.figure(figsize=(8, 6))
    plt.plot(data["fpr"], data["tpr"], 
             label=f'{name} (AUC = {data["auc"]:.4f})', 
             color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'graph2/{name.replace(" ", "_").replace("-", "_")}_roc_curve.png', dpi=300, bbox_inches='tight')

print(f"\nEnhanced model training complete!")
print(f"Results saved to: graph2/model_performance_results.csv")
print(f"ROC curves saved as PNG files")
print(f"Best performing model: {results_df.iloc[0]['Model']} (AUC: {results_df.iloc[0]['Test AUC']:.4f})")

# Feature importance analysis for best model
best_model_name = results_df.iloc[0]['Model']
print(f"\nFeature importance analysis for {best_model_name}:")
def _unwrap_estimator(obj):
    """If obj is an imblearn/sklearn Pipeline, return the final estimator (named 'est' or last step).
    Otherwise return obj as-is."""
    try:
        if hasattr(obj, 'named_steps'):
            # prefer 'est' step used in our pipelines
            if 'est' in obj.named_steps:
                return obj.named_steps['est']
            # fallback to last step
            steps = list(obj.named_steps.items())
            if steps:
                return steps[-1][1]
    except Exception:
        pass
    return obj


if best_model_name == "XGBoost":
    xgb_pipe = trained_models.get("XGBoost")
    xgb_model = _unwrap_estimator(xgb_pipe)
    importance = getattr(xgb_model, 'feature_importances_', None)

elif best_model_name == "LightGBM":
    lgb_pipe = trained_models.get("LightGBM")
    lgb_model = _unwrap_estimator(lgb_pipe)
    importance = getattr(lgb_model, 'feature_importances_', None)

elif best_model_name == "Random Forest":
    rf_pipe = trained_models.get("Random Forest")
    rf_model = _unwrap_estimator(rf_pipe)
    importance = getattr(rf_model, 'feature_importances_', None)

elif best_model_name == "Gradient Boosting":
    gb_pipe = trained_models.get("Gradient Boosting")
    gb_model = _unwrap_estimator(gb_pipe)
    importance = getattr(gb_model, 'feature_importances_', None)

else:
    print(f"Feature importance not available for {best_model_name}")
    importance = None

if importance is not None:
    # Get top features
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print(f"Top 10 most important features:")
    print(feature_importance_df.head(10))
    
    # Save feature importance
    feature_importance_df.to_csv(f'graph2/{best_model_name.replace(" ", "_")}_feature_importance.csv', index=False)
    
    # Show top 5 genes specifically
    print(f"\nTop 5 most predictive genes for delirium:")
    top_5_genes = feature_importance_df.head(5)
    for i, (_, row) in enumerate(top_5_genes.iterrows(), 1):
        print(f"{i}. {row['Feature']} (Importance: {row['Importance']:.6f})")

# Comprehensive visualization of all models' test AUCs
print("\n--- Generating Comprehensive Visualizations ---")

# 1. Bar chart of all test AUCs
plt.figure(figsize=(14, 8))
models_list = results_df['Model'].tolist()
aucs_list = results_df['Test AUC'].tolist()

# Create color-coded bars
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
bars = plt.bar(range(len(models_list)), aucs_list, color=colors[:len(models_list)], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (bar, auc) in enumerate(zip(bars, aucs_list)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.xlabel('Models', fontsize=14, fontweight='bold')
plt.ylabel('Test AUC Score', fontsize=14, fontweight='bold')
plt.title('All Models Test AUC Performance Comparison', fontsize=16, fontweight='bold')
plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graph2/all_models_test_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # Disabled interactive display for automated execution

# 2. Horizontal bar chart for better readability
plt.figure(figsize=(12, 10))
y_pos = np.arange(len(models_list))
bars = plt.barh(y_pos, aucs_list, color=colors[:len(models_list)], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for i, (bar, auc) in enumerate(zip(bars, aucs_list)):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
             f'{auc:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)

plt.yticks(y_pos, models_list)
plt.xlabel('Test AUC Score', fontsize=14, fontweight='bold')
plt.title('All Models Test AUC Performance (Horizontal View)', fontsize=16, fontweight='bold')
plt.xlim(0, 1.0)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('graph2/all_models_test_auc_horizontal.png', dpi=300, bbox_inches='tight')
plt.close()  # Disabled interactive display for automated execution

# 3. Performance ranking visualization
plt.figure(figsize=(10, 8))
ranked_models = results_df.sort_values('Test AUC', ascending=True)
y_pos = np.arange(len(ranked_models))
bars = plt.barh(y_pos, ranked_models['Test AUC'], color=plt.cm.viridis(np.linspace(0, 1, len(ranked_models))), alpha=0.8)

# Add ranking numbers and AUC values
for i, (idx, row) in enumerate(ranked_models.iterrows()):
    width = row['Test AUC']
    plt.text(width + 0.01, i, f'#{len(ranked_models)-i} - {width:.4f}', 
             ha='left', va='center', fontweight='bold', fontsize=10)

plt.yticks(y_pos, ranked_models['Model'])
plt.xlabel('Test AUC Score', fontsize=14, fontweight='bold')
plt.title('Models Ranked by Test AUC Performance', fontsize=16, fontweight='bold')
plt.xlim(0, 1.0)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('graph2/all_models_auc_ranking.png', dpi=300, bbox_inches='tight')
plt.close()  # Disabled interactive display for automated execution

# Create graph folder if it doesn't exist
os.makedirs('graph2', exist_ok=True)

# Feature importance and SHAP analysis for selected individual models
print(f"\n=== FEATURE IMPORTANCE AND SHAP ANALYSIS ===")

# Get top 5 individual models (excluding ensemble models) and select specific ones for SHAP
all_individual_models = results_df[~results_df['Model'].str.contains('Ensemble')].head(5)
print(f"Top 5 individual models: {all_individual_models['Model'].tolist()}")

# Select models for SHAP analysis: XGBoost, Gradient Boosting, LightGBM, Logistic Regression, Random Forest, SVM, and Ensemble models
# Include SVM even if it's not in top 5, and get all models for SHAP selection
all_models_for_shap = results_df  # Include all models (individual + ensemble)
shap_models = all_models_for_shap[all_models_for_shap['Model'].isin(['XGBoost', 'Gradient Boosting', 'LightGBM', 'Logistic Regression', 'Random Forest', 'SVM', 'Voting Ensemble', 'Stacking Ensemble'])]
print(f"Models selected for SHAP analysis: {shap_models['Model'].tolist()}")
print(f"Generating SHAP plots for all selected models")

# If Logistic Regression is not in individual models, add it from the full results
if 'Logistic Regression' not in shap_models['Model'].values:
    lr_model = results_df[results_df['Model'] == 'Logistic Regression']
    if not lr_model.empty:
        shap_models = pd.concat([shap_models, lr_model], ignore_index=True)
        print(f"Added Logistic Regression to SHAP analysis")
        print(f"Updated models for SHAP analysis: {shap_models['Model'].tolist()}")

# Sort by AUC for consistent ranking
shap_models = shap_models.sort_values('Test AUC', ascending=False).reset_index(drop=True)
print(f"Models sorted by AUC rank (high to low): {shap_models['Model'].tolist()}")

# Helper function to check if a feature is in the top 10 of a model's SHAP values
def _feature_in_model_top10(feature, model_name):
    """Helper function to check if a feature is in the top 10 of a model's SHAP values.
    Handles cases where the CSV file may not exist or be readable."""
    try:
        shap_csv_path = f'graph2/{model_name.replace(" ", "_")}_shap_values.csv'
        if os.path.exists(shap_csv_path):
            shap_df = pd.read_csv(shap_csv_path)
            top_10_features = shap_df.head(10)['Feature'].tolist()
            return feature in top_10_features
        return False
    except Exception as e:
        print(f"Warning: Could not read SHAP values for model {model_name}: {e}")
        return False

for rank, (_, model_row) in enumerate(shap_models.iterrows(), 1):
    model_name = model_row['Model']
    model_auc = model_row['Test AUC']
    
    # Get actual AUC rank from the sorted results
    auc_rank = shap_models[shap_models['Model'] == model_name].index[0] + 1
    print(f"\n--- Rank {rank}: {model_name} (AUC: {model_auc:.4f}) ---")
    
    # Get the actual model object
    if model_name in trained_models:
        model = trained_models[model_name]
        print(f"✅ Successfully retrieved {model_name} model from trained_models")
    else:
        print(f"❌ Model {model_name} not found in trained_models")
        continue

    # Unwrap pipeline to get the underlying estimator for feature-importance based explainers
    orig_model = model
    estimator_for_explain = _unwrap_estimator(orig_model)

    # Prepare X_test_df for SHAP: if the stored model is a pipeline with a scaler, use its scaler to transform
    if hasattr(orig_model, 'named_steps') and 'scaler' in orig_model.named_steps:
        try:
            X_test_scaled_for_shap = orig_model.named_steps['scaler'].transform(X_test)
            X_test_df = pd.DataFrame(X_test_scaled_for_shap, columns=selected_features)
        except Exception:
            # Fallback to global scaler
            if scaler is not None:
                X_test_scaled_for_shap = scaler.transform(X_test)
                X_test_df = pd.DataFrame(X_test_scaled_for_shap, columns=selected_features)
            else:
                X_test_df = pd.DataFrame(X_test, columns=selected_features)
    else:
        if scaler is not None:
            X_test_scaled_for_shap = scaler.transform(X_test)
            X_test_df = pd.DataFrame(X_test_scaled_for_shap, columns=selected_features)
        else:
            X_test_df = pd.DataFrame(X_test, columns=selected_features)
    
    # Feature importance analysis
    if model_name in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"]:
        # Get feature importance for tree-based models
        # model may be a Pipeline — use the unwrapped estimator
        importance = getattr(estimator_for_explain, 'feature_importances_', None)
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"Top 10 most important features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Print top 5 most predictive genes for delirium
        print(f"Top 5 most predictive genes for delirium:")
        for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
            print(f"{i}. {row['Feature']} (Importance: {row['Importance']:.6f})")
        
        # Save feature importance to CSV
        feature_importance_df.to_csv(f'graph2/{model_name.replace(" ", "_")}_feature_importance.csv', index=False)
        print(f"Feature importance saved to: graph2/{model_name.replace(' ', '_')}_feature_importance.csv")
        # Save top-20 predictive genes for this model
        try:
            top20 = feature_importance_df.head(20)
            top20.to_csv(f'graph2/{model_name.replace(" ", "_")}_top20_genes.csv', index=False)
            print(f"💾 Saved top-20 genes for {model_name} to graph2/{model_name.replace(' ', '_')}_top20_genes.csv")
        except Exception as e:
            print(f"❌ Failed to save top-20 genes for {model_name}: {e}")
        
    elif model_name == "Logistic Regression":
        # Get feature importance for logistic regression
        # The stored `model` may be an imblearn/sklearn Pipeline — unwrap to the final estimator if necessary
        try:
            if hasattr(model, 'named_steps'):
                est = model.named_steps.get('est', None)
                if est is None:
                    # fallback to last step
                    steps = list(model.named_steps.items())
                    est = steps[-1][1] if steps else model
            else:
                est = model

            coef = getattr(est, 'coef_', None)
            if coef is None:
                # If coef_ not present, try to access `coef_` on the pipeline itself as last resort
                coef = getattr(model, 'coef_', None)

            if coef is None:
                print(f"Warning: could not find coef_ for Logistic Regression model; skipping feature-importance for this model.")
                importance = np.zeros(len(selected_features))
            else:
                # Ensure we have a 1D array matching feature list
                importance = np.abs(np.asarray(coef).ravel())
                if importance.shape[0] < len(selected_features):
                    # pad with zeros if mismatch
                    padded = np.zeros(len(selected_features))
                    padded[:importance.shape[0]] = importance
                    importance = padded

        except Exception as e:
            print(f"Error extracting Logistic Regression coefficients: {e}")
            importance = np.zeros(len(selected_features))

        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print(f"Top 10 most important features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Print top 5 most predictive genes for delirium
        print(f"Top 5 most predictive genes for delirium:")
        for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
            print(f"{i}. {row['Feature']} (Importance: {row['Importance']:.6f})")
        
        # Save feature importance to CSV
        feature_importance_df.to_csv(f'graph2/{model_name.replace(" ", "_")}_feature_importance.csv', index=False)
        print(f"Feature importance saved to: graph2/{model_name.replace(' ', '_')}_feature_importance.csv")
        
    elif model_name == "SVM":
        # For SVM, we'll use SHAP values to estimate importance
        print(f"SVM feature importance will be calculated using SHAP values")
    
    # SHAP Analysis for each model
    print(f"Generating SHAP plots for {model_name}...")
    try:
        # Prepare data for SHAP
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        X_test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        # Create SHAP explainer based on model type
        if model_name in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"]:
            # Tree-based models
            print(f"🔍 Creating TreeExplainer for {model_name}...")
            try:
                explainer = shap.TreeExplainer(estimator_for_explain)
                print(f"✅ TreeExplainer created successfully for {model_name}")
                
                print(f"🔍 Computing SHAP values for {model_name}...")
                shap_values = explainer.shap_values(X_test_df)
                print(f"✅ SHAP values computed successfully for {model_name}")
                
                # For tree-based models, shap_values might be a list or 3D array
                if isinstance(shap_values, list):
                    print(f"📊 SHAP values is a list with {len(shap_values)} elements")
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    print(f"📊 Using element {1 if len(shap_values) > 1 else 0} from SHAP values list")
                else:
                    shap_array = np.array(shap_values)
                    print(f"📊 SHAP values is a single array with shape {shap_array.shape}")
                    
                    # Handle 3D arrays (samples, features, classes) - common with Random Forest
                    if len(shap_array.shape) == 3:
                        print(f"📊 Detected 3D SHAP array, extracting class 1 (positive class) for binary classification")
                        shap_values = shap_array[:, :, 1]  # Extract SHAP values for class 1 (delirium)
                        print(f"📊 Extracted 2D SHAP values with shape {shap_values.shape}")
                    else:
                        print(f"📊 Using SHAP values as-is with shape {shap_values.shape}")
            except Exception as e:
                print(f"❌ Error in TreeExplainer for {model_name}: {str(e)}")
                raise
                
        elif model_name == "Logistic Regression":
            # Linear models
            print(f"🔍 Creating LinearExplainer for {model_name}...")
            try:
                explainer = shap.LinearExplainer(estimator_for_explain, X_test_df)
                print(f"✅ LinearExplainer created successfully for {model_name}")
                
                print(f"🔍 Computing SHAP values for {model_name}...")
                shap_values = explainer.shap_values(X_test_df)
                print(f"✅ SHAP values computed successfully for {model_name}")
            except Exception as e:
                print(f"❌ Error in LinearExplainer for {model_name}: {str(e)}")
                raise
            
        elif model_name == "SVM":
            # Kernel models - use KernelExplainer
            print(f"🔍 Creating KernelExplainer for {model_name}...")
            try:
                # Create background dataset using kmeans for better performance
                X_train_summary = shap.kmeans(X_train_scaled, 50)
                explainer = shap.KernelExplainer(orig_model.predict_proba, X_train_summary)
                print(f"✅ KernelExplainer created successfully for {model_name}")
                
                print(f"🔍 Computing SHAP values for {model_name}...")
                shap_values = explainer.shap_values(X_test_df)
                print(f"✅ SHAP values computed successfully for {model_name}")
                
                # Handle KernelExplainer output format for binary classification
                if isinstance(shap_values, list):
                    print(f"📊 SVM SHAP values is a list with {len(shap_values)} elements")
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    print(f"📊 Using element {1 if len(shap_values) > 1 else 0} from SVM SHAP values list")
                else:
                    shap_array = np.array(shap_values)
                    print(f"📊 SVM SHAP values is a single array with shape {shap_array.shape}")
                    
                    # Handle 3D arrays (samples, features, classes) - common with KernelExplainer
                    if len(shap_array.shape) == 3:
                        print(f"📊 Detected 3D SVM SHAP array, extracting class 1 (positive class) for binary classification")
                        shap_values = shap_array[:, :, 1]  # Extract SHAP values for class 1 (delirium)
                        print(f"📊 Extracted 2D SVM SHAP values with shape {shap_values.shape}")
                    else:
                        print(f"📊 Using SVM SHAP values as-is with shape {shap_values.shape}")
            except Exception as e:
                print(f"❌ Error in KernelExplainer for {model_name}: {str(e)}")
                raise
            
        elif model_name in ["Voting Ensemble", "Stacking Ensemble"]:
            # Ensemble models - use KernelExplainer
            print(f"🔍 Creating KernelExplainer for {model_name}...")
            try:
                # Create background dataset using kmeans for better performance
                X_train_summary = shap.kmeans(X_train_scaled, 50)
                explainer = shap.KernelExplainer(orig_model.predict_proba, X_train_summary)
                print(f"✅ KernelExplainer created successfully for {model_name}")
                
                print(f"🔍 Computing SHAP values for {model_name}...")
                shap_values = explainer.shap_values(X_test_df)
                print(f"✅ SHAP values computed successfully for {model_name}")
                
                # Handle KernelExplainer output format for binary classification
                if isinstance(shap_values, list):
                    print(f"📊 {model_name} SHAP values is a list with {len(shap_values)} elements")
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    print(f"📊 Using element {1 if len(shap_values) > 1 else 0} from {model_name} SHAP values list")
                else:
                    shap_array = np.array(shap_values)
                    print(f"📊 {model_name} SHAP values is a single array with shape {shap_array.shape}")
                    
                    # Handle 3D arrays (samples, features, classes) - common with KernelExplainer
                    if len(shap_array.shape) == 3:
                        print(f"📊 Detected 3D {model_name} SHAP array, extracting class 1 (positive class) for binary classification")
                        shap_values = shap_array[:, :, 1]  # Extract SHAP values for class 1 (delirium)
                        print(f"📊 Extracted 2D {model_name} SHAP values with shape {shap_values.shape}")
                    else:
                        print(f"📊 Using {model_name} SHAP values as-is with shape {shap_values.shape}")
            except Exception as e:
                print(f"❌ Error in KernelExplainer for {model_name}: {str(e)}")
                raise
            
        else:
            # Default to KernelExplainer for other models
            print(f"🔍 Creating KernelExplainer for {model_name}...")
            try:
                # Create background dataset using kmeans for better performance
                X_train_summary = shap.kmeans(X_train_scaled, 50)
                explainer = shap.KernelExplainer(orig_model.predict_proba, X_train_summary)
                print(f"✅ KernelExplainer created successfully for {model_name}")
                
                print(f"🔍 Computing SHAP values for {model_name}...")
                shap_values = explainer.shap_values(X_test_df)
                print(f"✅ SHAP values computed successfully for {model_name}")
            except Exception as e:
                print(f"❌ Error in KernelExplainer for {model_name}: {str(e)}")
                raise
        
        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, feature_names=selected_features, show=False)
        plt.title(f'SHAP Summary Plot - {model_name} (Rank {rank}, AUC: {model_auc:.4f})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph2/{model_name.replace(" ", "_")}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()  # Disabled interactive display for automated execution
        
        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, feature_names=selected_features, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name} (Rank {rank}, AUC: {model_auc:.4f})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph2/{model_name.replace(" ", "_")}_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()  # Disabled interactive display for automated execution
        
        # 3. Force Plot for a sample prediction
        sample_idx = 0  # First test sample
        force_plot_success = False
        try:
            # Handle expected_value for different model types
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)) and len(np.array(expected_value).shape) > 0:
                # For multi-class models, use the positive class expected value
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            # Use new SHAP v0.20+ syntax
            shap.plots.force(expected_value, shap_values[sample_idx],
                           X_test_df.iloc[sample_idx], show=False)
            plt.title(f'SHAP Force Plot - Sample {sample_idx} ({model_name}, Rank {rank})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'graph2/{model_name.replace(" ", "_")}_shap_force.png', dpi=300, bbox_inches='tight')
            plt.close()  # Disabled interactive display for automated execution
            force_plot_success = True
        except Exception as e:
            print(f"Force plot failed for {model_name}: {str(e)}")
        
        # 4. Dependence Plot for top feature
        dependence_plot_success = False
        try:
            # Get top feature by mean absolute SHAP value
            mean_shap_abs = np.mean(np.abs(shap_values), axis=0)
            top_feature_idx = np.argmax(mean_shap_abs)
            top_feature = selected_features[top_feature_idx]
            
            shap.dependence_plot(top_feature, shap_values, X_test_df, show=False)
            plt.title(f'SHAP Dependence Plot - {top_feature} ({model_name}, Rank {rank})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'graph2/{model_name.replace(" ", "_")}_shap_dependence.png', dpi=300, bbox_inches='tight')
            plt.close()  # Disabled interactive display for automated execution
            dependence_plot_success = True
        except Exception as e:
            print(f"Dependence plot failed for {model_name}: {str(e)}")
        
        # Print top 10 SHAP values for this model
        print(f"\n--- Top 10 SHAP Values for {model_name} ---")
        try:
            # Calculate mean absolute SHAP values across all samples
            mean_shap_abs = np.mean(np.abs(shap_values), axis=0)
            
            # Create DataFrame with SHAP values
            shap_df = pd.DataFrame({
                'Feature': selected_features,
                'Mean_ABS_SHAP': mean_shap_abs
            }).sort_values('Mean_ABS_SHAP', ascending=False)
            
            print("Top 10 features by SHAP importance:")
            print(shap_df.head(10).to_string(index=False))
            
            # Save SHAP values to CSV
            shap_df.to_csv(f'graph2/{model_name.replace(" ", "_")}_shap_values.csv', index=False)
            print(f"SHAP values saved to: graph2/{model_name.replace(' ', '_')}_shap_values.csv")
            # Save top-20 SHAP features as predictive genes
            try:
                top20_shap = shap_df.head(20)
                top20_shap.to_csv(f'graph2/{model_name.replace(" ", "_")}_top20_genes.csv', index=False)
                print(f"💾 Saved top-20 SHAP genes for {model_name} to graph2/{model_name.replace(' ', '_')}_top20_genes.csv")
            except Exception as e:
                print(f"❌ Failed to save top-20 SHAP genes for {model_name}: {e}")
            
        except Exception as e:
            print(f"Failed to calculate SHAP values summary: {str(e)}")
        
        print(f"✅ SHAP plots for {model_name} saved to graph2/ folder")
        
    except Exception as e:
        print(f"❌ SHAP analysis failed for {model_name}: {str(e)}")

# Analyze feature frequency across all models' top 10 SHAP features
print(f"\n=== FEATURE FREQUENCY ANALYSIS ACROSS ALL MODELS ===")
print("Analyzing which features appear most often in top 10 SHAP features...")

# Count feature appearances across models
feature_counts = {}
for model_name in shap_models['Model'].values:
    try:
        # Load the SHAP values CSV for this model
        shap_csv_path = f'graph2/{model_name.replace(" ", "_")}_shap_values.csv'
        if os.path.exists(shap_csv_path):
            shap_df = pd.read_csv(shap_csv_path)
            top_10_features = shap_df.head(10)['Feature'].tolist()
            
            for feature in top_10_features:
                if feature not in feature_counts:
                    feature_counts[feature] = []
                feature_counts[feature].append(model_name)
            
            print(f"{model_name}: {len(top_10_features)} features processed")
        else:
            print(f"Warning: SHAP CSV not found for {model_name}")
    except Exception as e:
        print(f"Warning: Error processing {model_name}: {str(e)}")

# Calculate average SHAP values for tie-breaking
feature_avg_shap = {}
for feature, models_list in feature_counts.items():
    shap_values_list = []
    for model_name in models_list:
        try:
            shap_csv_path = f'graph2/{model_name.replace(" ", "_")}_shap_values.csv'
            if os.path.exists(shap_csv_path):
                shap_df = pd.read_csv(shap_csv_path)
                if feature in shap_df['Feature'].values:
                    # Get SHAP value for this feature in this model
                    feature_shap = shap_df[shap_df['Feature'] == feature]['Mean_ABS_SHAP'].iloc[0]
                    shap_values_list.append(feature_shap)
        except Exception as e:
            continue
    
    if shap_values_list:
        feature_avg_shap[feature] = np.mean(shap_values_list)
    else:
        feature_avg_shap[feature] = 0.0

# Sort features by frequency first, then by average SHAP values (tie-breaker)
sorted_features = sorted(feature_counts.items(), 
                        key=lambda x: (x[1], feature_avg_shap[x[0]]), 
                        reverse=True)

print(f"\n📊 FEATURE FREQUENCY RANKING (Top 5 Most Common)")
print("="*70)
print(f"{'Rank':<4} {'Feature':<20} {'Appearances':<12} {'Avg_SHAP':<12} {'Models':<20}")
print("-" * 70)

for rank, (feature, models_list) in enumerate(sorted_features[:5], 1):
    count = len(models_list)
    models_str = ', '.join(models_list[:3])  # Show first 3 models
    if len(models_list) > 3:
        models_str += f" (+{len(models_list)-3} more)"
    avg_shap = feature_avg_shap[feature]
    print(f"{rank:<4} {feature:<20} {count:<12} {avg_shap:<12.4f} {models_str:<20}")

# Save enhanced feature frequency analysis
enhanced_frequency_df = pd.DataFrame([
    {
        'Feature': feature,
        'Appearances': len(models_list),
        'Avg_SHAP': feature_avg_shap[feature],
        'Models': ', '.join([
            m for m in shap_models['Model'].values
            if _feature_in_model_top10(feature, m)
        ])
    }
    for feature, models_list in sorted_features
])

enhanced_frequency_df.to_csv('graph2/feature_frequency_analysis.csv', index=False)
print(f"\n📁 Enhanced feature frequency analysis saved to: graph2/feature_frequency_analysis.csv")
print("Includes tie-breaking based on average SHAP values across all models")

print("Comprehensive visualizations saved to 'graph2' folder:")
print("- graph2/all_models_test_auc_comparison.png")
print("- graph2/all_models_test_auc_horizontal.png")
print("- graph2/all_models_auc_ranking.png")
print("- All individual ROC curve PNGs in graph2/ folder")
print("- graph2/enhanced_models_roc_curves.png")
print("- Feature importance CSV files in graph2/ folder")
print("- SHAP plots for 8 selected models: XGBoost, Gradient Boosting, LightGBM, Logistic Regression, Random Forest, SVM, Voting Ensemble, Stacking Ensemble")
print("- SHAP values CSV files with top features for each model")
print("- Generated SHAP plots for all selected models including Random Forest, SVM, and Ensemble models")