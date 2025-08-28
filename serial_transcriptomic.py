import warnings
import contextlib
import os
import shap

# Suppress all warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

# Set random seeds for reproducibility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import re

np.random.seed(42)

# --- Ensure serialGraph folder exists ---
def ensure_serial_graph_folder():
    """Create serialGraph folder if it doesn't exist"""
    serial_graph_folder = 'serialGraph'
    if not os.path.exists(serial_graph_folder):
        os.makedirs(serial_graph_folder)
        print(f"✅ Created {serial_graph_folder}/ folder")
    else:
        print(f"✅ {serial_graph_folder}/ folder already exists")

# Create serialGraph folder
ensure_serial_graph_folder()

print("=== SERIAL TRANSCRIPTOMIC ANALYSIS ===")
print("Using serial transcriptomic data (any time during hospitalization)")

# --- Data Loading and Preprocessing ---
def load_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    if ext in ('.xls', '.xlsx'):
        return pd.read_excel(path, engine='openpyxl')
    raise ValueError(f"Unsupported extension: {ext}")

print("Loading serial transcriptomic data...")
demographic_df = load_table('/users/audreysu/AudreyCovidProject/delirium cohort demographics.xlsx')
serial_df = load_table('/users/audreysu/AudreyCovidProject/serial_norm_gene_exp_df.csv')
serial_annotation_df = load_table('/users/audreysu/AudreyCovidProject/serial_samples_annotation.csv')

print(f"Demographic data shape: {demographic_df.shape}")
print(f"Serial transcriptomic data shape: {serial_df.shape}")
print(f"Serial annotation data shape: {serial_annotation_df.shape}")

# Data preprocessing for serial data
demographic_df = demographic_df.head(128)  # Use first 128 patients as in original

# Process serial transcriptomic data
serial_df = serial_df.rename(columns={serial_df.columns[0]: "gene"})
serial_df.set_index("gene", inplace=True)
serial_df = serial_df.T
serial_df = serial_df.reset_index()
serial_df = serial_df.rename(columns={serial_df.columns[0]: "X"})

# Merge serial data with annotation
merged_serial = pd.merge(serial_df, serial_annotation_df.drop(columns=["Delirium"]), on="X")

# Transform sample IDs to match demographic data format
def transform_value(val):
    match = re.search(r"MVIR1.HS(\d+)", str(val))
    if match:
        code = match.group(1)
        return 1000 + int(code)
    return val

merged_serial["X"] = merged_serial["X"].apply(transform_value)
merged_serial.rename(columns={"X": "Master Record ID"}, inplace=True)

# Merge with demographic data
combined_df = pd.merge(demographic_df, merged_serial, on='Master Record ID', how='outer')
combined_df.rename(columns={"Delirium at any time during hospitalization": "Delirium"}, inplace=True)

print(f"Combined data shape: {combined_df.shape}")
print(f"Target distribution: {combined_df['Delirium'].value_counts()}")

# Handle missing values
target = "Delirium"
combined_df = combined_df.dropna(subset=[target]).reset_index(drop=True)

cat_cols = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_impute_cols = [c for c in cat_cols if c != target]

num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
num_impute_cols = [c for c in num_cols if c != target]

# Fill categorical NaNs
for c in cat_impute_cols:
    mode_vals = combined_df[c].mode(dropna=True)
    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
    combined_df[c] = combined_df[c].fillna(fill_val)

# Handle missing numeric values
missing_percentage = combined_df[num_impute_cols].isna().sum() / len(combined_df)
cols_to_drop = missing_percentage[missing_percentage > 0.3].index.tolist()

if cols_to_drop:
    print(f"Dropping {len(cols_to_drop)} columns with >30% missing values")
    combined_df = combined_df.drop(columns=cols_to_drop)

# Fill remaining numeric NaNs with median
for col in num_impute_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Prepare features and target
# Remove "Diagnosis" column as it's nearly identical to Delirium target
if "Diagnosis" in combined_df.columns:
    print("Removing 'Diagnosis' column as it's nearly identical to Delirium target")
    combined_df = combined_df.drop(columns=["Diagnosis"])

# Check similarity between "Late_del" and "Delirium" target
if "Late_del" in combined_df.columns:
    print("\n=== Checking 'Late_del' vs 'Delirium' similarity ===")
    print(f"Late_del value counts:\n{combined_df['Late_del'].value_counts()}")
    print(f"Delirium value counts:\n{combined_df['Delirium'].value_counts()}")
    
    # Calculate correlation/similarity
    if combined_df['Late_del'].dtype in ['int64', 'float64']:
        correlation = combined_df['Late_del'].corr(combined_df['Delirium'])
        print(f"Correlation between Late_del and Delirium: {correlation:.4f}")
        
        # Check overlap
        overlap = (combined_df['Late_del'] == combined_df['Delirium']).sum()
        total = len(combined_df)
        overlap_percentage = (overlap / total) * 100
        print(f"Exact overlap: {overlap}/{total} ({overlap_percentage:.1f}%)")
        
        if overlap_percentage > 80:
            print("⚠️  WARNING: Late_del and Delirium are very similar (>80% overlap)")
            print("This could cause data leakage. Removing Late_del to prevent data leakage.")
            combined_df = combined_df.drop(columns=["Late_del"])
        elif overlap_percentage > 60:
            print("⚠️  CAUTION: Late_del and Delirium have moderate overlap (>60%)")
        else:
            print("✅ Late_del and Delirium have acceptable overlap")

X = combined_df.drop(columns=[target] + cat_impute_cols)
y = combined_df[target]

# Convert target to numeric if needed
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})

print(f"Features after encoding: {X.shape[1]}")

# Advanced feature selection for serial data
def advanced_feature_selection(X, y, n_features=200):
    """Multiple feature selection strategies for serial transcriptomic data"""
    print(f"Performing advanced feature selection to select {n_features} features...")
    
    # Strategy 1: Variance threshold (remove low variance features)
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
sampler = sampling_strategies[best_sampler]
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)

print(f"Resampled training set: {X_train_resampled.shape}")
print(f"Resampled target distribution: {np.bincount(y_train_resampled)}")

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
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
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
        "model": SVC(probability=True, random_state=42),
        "params": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': ['balanced', None]
        }
    }
}

print("\n" + "="*60)
print("ENHANCED MODEL TRAINING AND OPTIMIZATION")
print("="*60)

# Results storage
results = {}
roc_curves_data = {}

# Train and optimize each model
for model_name, model_config in models.items():
    print(f"\n--- Training {model_name} ---")
    
    # Use RandomizedSearchCV for faster optimization
    search = RandomizedSearchCV(
        model_config["model"],
        model_config["params"],
        n_iter=50,  # Number of parameter combinations to try
        cv=5,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the model
    search.fit(X_train_resampled, y_train_resampled)
    
    # Get best model and score
    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_auc = search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"CV AUC: {cv_auc:.4f}")
    
    # Test set performance
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[model_name] = {
        'model': best_model,
        'best_params': best_params,
        'cv_auc': cv_auc,
        'test_auc': test_auc
    }
    
    # Store ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_curves_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": test_auc}
    
    print(f"Test AUC: {test_auc:.4f}")

# Create ensemble models
print("\n--- Creating Ensemble Models ---")

# Voting Classifier (soft voting)
voting_clf = VotingClassifier(
    estimators=[(name, results[name]['model']) for name in results.keys()],
    voting='soft'
)

voting_clf.fit(X_train_resampled, y_train_resampled)
y_pred_voting = voting_clf.predict_proba(X_test_scaled)[:, 1]
voting_auc = roc_auc_score(y_test, y_pred_voting)

# Stacking Classifier
estimators = [(name, results[name]['model']) for name in results.keys()]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_clf.fit(X_train_resampled, y_train_resampled)
y_pred_stacking = stacking_clf.predict_proba(X_test_scaled)[:, 1]
stacking_auc = roc_auc_score(y_test, y_pred_stacking)

print(f"Voting Ensemble AUC: {voting_auc:.4f}")
print(f"Stacking Ensemble AUC: {stacking_auc:.4f}")

# Store ensemble results
results['Voting Ensemble'] = {'test_auc': voting_auc, 'cv_auc': 'N/A'}
results['Stacking Ensemble'] = {'test_auc': stacking_auc, 'cv_auc': 'N/A'}

# Create results DataFrame
results_df = pd.DataFrame([
    {
        'Model': name,
        'Test AUC': results[name]['test_auc'],
        'CV AUC': results[name]['cv_auc']
    }
    for name in results.keys()
])

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('serialGraph/serial_transcriptomic_results.csv', index=False)
print(f"\nSerial transcriptomic analysis complete!")
print(f"Results saved to: serialGraph/serial_transcriptomic_results.csv")

# Feature importance and SHAP analysis for selected individual models
print(f"\n=== FEATURE IMPORTANCE AND SHAP ANALYSIS ===")

# Get top 5 individual models (excluding ensemble models) and select specific ones for SHAP
all_individual_models = results_df[~results_df['Model'].str.contains('Ensemble')].head(5)
print(f"Top 5 individual models: {all_individual_models['Model'].tolist()}")

# Select models for SHAP analysis: XGBoost, Gradient Boosting, LightGBM, Logistic Regression
# Skip Random Forest and SVM as requested
shap_models = all_individual_models[all_individual_models['Model'].isin(['XGBoost', 'Gradient Boosting', 'LightGBM', 'Logistic Regression'])]
print(f"Models selected for SHAP analysis: {shap_models['Model'].tolist()}")
print(f"Skipping Random Forest and SVM SHAP plots as requested")

# If Logistic Regression is not in individual models, add it from the full results
if 'Logistic Regression' not in shap_models['Model'].values:
    lr_model = results_df[results_df['Model'] == 'Logistic Regression']
    if not lr_model.empty:
        shap_models = pd.concat([shap_models, lr_model], ignore_index=True)
        print(f"Added Logistic Regression to SHAP analysis")
        print(f"Updated models for SHAP analysis: {shap_models['Model'].tolist()}")

# Sort models by Test AUC (high to low) for proper ranking
shap_models = shap_models.sort_values('Test AUC', ascending=False).reset_index(drop=True)
print(f"Models sorted by AUC rank (high to low): {shap_models['Model'].tolist()}")

for rank, (_, model_row) in enumerate(shap_models.iterrows(), 1):
    model_name = model_row['Model']
    model_auc = model_row['Test AUC']
    
    # Get actual AUC rank from the sorted results
    auc_rank = shap_models[shap_models['Model'] == model_name].index[0] + 1
    print(f"\n--- AUC Rank {auc_rank}: {model_name} (AUC: {model_auc:.4f}) ---")
    
    # Get the actual model object
    model = results[model_name]['model']
    
    # Feature importance analysis
    if model_name in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"]:
        # Get feature importance for tree-based models
        importance = model.feature_importances_
        
    elif model_name == "Logistic Regression":
        # Get feature importance for linear models
        importance = np.abs(model.coef_[0])
        
    elif model_name == "SVM":
        # For SVM, we'll use SHAP values to estimate importance
        importance = None
    else:
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
        feature_importance_df.to_csv(f'serialGraph/{model_name.replace(" ", "_")}_feature_importance.csv', index=False)
        
        # Show top 5 genes specifically
        print(f"Top 5 most predictive genes for delirium:")
        top_5_genes = feature_importance_df.head(5)
        for i, (_, row) in enumerate(top_5_genes.iterrows(), 1):
            print(f"{i}. {row['Feature']} (Importance: {row['Importance']:.6f})")
    
    # SHAP Analysis for each model
    print(f"Generating SHAP plots for {model_name}...")
    
    try:
        # Prepare data for SHAP
        X_test_scaled = scaler.transform(X_test)
        X_test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        # Create SHAP explainer based on model type
        if model_name in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"]:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_df)
            
            # For tree-based models, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
        elif model_name == "Logistic Regression":
            # Linear models
            explainer = shap.LinearExplainer(model, X_test_df)
            shap_values = explainer.shap_values(X_test_df)
            
        elif model_name == "SVM":
            # Kernel models - use KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test_df, 100))
            shap_values = explainer.shap_values(X_test_df)
            
        else:
            # Default to KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test_df, 100))
            shap_values = explainer.shap_values(X_test_df)
        
        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, feature_names=selected_features, show=False)
        plt.title(f'SHAP Summary Plot - {model_name} (AUC Rank {auc_rank}, AUC: {model_auc:.4f})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'serialGraph/{model_name.replace(" ", "_")}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, feature_names=selected_features, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name} (AUC Rank {auc_rank}, AUC: {model_auc:.4f})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'serialGraph/{model_name.replace(" ", "_")}_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Force Plot for a sample prediction
        sample_idx = 0  # First test sample
        force_plot_success = False
        try:
            # Use new SHAP v0.20+ syntax
            shap.plots.force(explainer.expected_value, shap_values[sample_idx], 
                           X_test_df.iloc[sample_idx], show=False)
            force_plot_success = True
        except Exception as e:
            print(f"Force plot failed for {model_name}: {str(e)}")
            # Continue without force plot
        
        if force_plot_success:
            plt.title(f'SHAP Force Plot - Sample {sample_idx} ({model_name}, AUC Rank {auc_rank})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'serialGraph/{model_name.replace(" ", "_")}_shap_force.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Dependence Plot for top feature
        try:
            if importance is not None:
                top_feature = feature_importance_df.iloc[0]['Feature']
            else:
                # For SVM, use first feature as fallback
                top_feature = selected_features[0]
                
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(top_feature, shap_values, X_test_df, show=False)
            plt.title(f'SHAP Dependence Plot - {top_feature} ({model_name}, AUC Rank {auc_rank})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'serialGraph/{model_name.replace(" ", "_")}_shap_dependence.png', dpi=300, bbox_inches='tight')
            plt.show()
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
            shap_df.to_csv(f'serialGraph/{model_name.replace(" ", "_")}_shap_values.csv', index=False)
            print(f"SHAP values saved to: serialGraph/{model_name.replace(' ', '_')}_shap_values.csv")
            
        except Exception as e:
            print(f"Failed to calculate SHAP values summary: {str(e)}")
        
        print(f"✅ SHAP plots for {model_name} saved to serialGraph/ folder")
        
    except Exception as e:
        print(f"❌ SHAP analysis failed for {model_name}: {str(e)}")
        print("This might be due to model type incompatibility or data issues.")

# Analyze feature frequency across all models' top 10 SHAP features
print("\n=== FEATURE FREQUENCY ANALYSIS ACROSS ALL MODELS ===")
print("Analyzing which features appear most often in top 10 SHAP features...")

# Dictionary to store feature counts
feature_counts = {}
all_top_features = []

# Collect all top 10 features from each model
for model_name in shap_models['Model'].values:
    try:
        # Load the SHAP values CSV for this model
        shap_csv_path = f'serialGraph/{model_name.replace(" ", "_")}_shap_values.csv'
        if os.path.exists(shap_csv_path):
            shap_df = pd.read_csv(shap_csv_path)
            # Get top 10 features
            top_10_features = shap_df.head(10)['Feature'].tolist()
            all_top_features.extend(top_10_features)
            
            # Count occurrences
            for feature in top_10_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
                
            print(f"{model_name}: {len(top_10_features)} features processed")
        else:
            print(f"Warning: SHAP CSV not found for {model_name}")
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")

# Calculate average SHAP values for tie-breaking
feature_avg_shap = {}
for feature in feature_counts.keys():
    shap_values_list = []
    for model_name in shap_models['Model'].values:
        try:
            shap_csv_path = f'serialGraph/{model_name.replace(" ", "_")}_shap_values.csv'
            if os.path.exists(shap_csv_path):
                shap_df = pd.read_csv(shap_csv_path)
                if feature in shap_df['Feature'].values:
                    # Get SHAP value for this feature in this model
                    feature_shap = shap_df[shap_df['Feature'] == feature]['Mean_ABS_SHAP'].iloc[0]
                    shap_values_list.append(feature_shap)
        except:
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
print("=" * 70)
print(f"{'Rank':<4} {'Feature':<20} {'Appearances':<12} {'Avg_SHAP':<12} {'Models':<20}")
print("-" * 70)

for rank, (feature, count) in enumerate(sorted_features[:5], 1):
    # Find which models this feature appears in
    models_with_feature = []
    for model_name in shap_models['Model'].values:
        try:
            shap_csv_path = f'serialGraph/{model_name.replace(" ", "_")}_shap_values.csv'
            if os.path.exists(shap_csv_path):
                shap_df = pd.read_csv(shap_csv_path)
                top_10_features = shap_df.head(10)['Feature'].tolist()
                if feature in top_10_features:
                    models_with_feature.append(model_name)
        except:
            continue
    
    models_str = ", ".join(models_with_feature)
    avg_shap = feature_avg_shap[feature]
    print(f"{rank:<4} {feature:<20} {count:<12} {avg_shap:<12.4f} {models_str:<20}")

# Save the enhanced frequency analysis to CSV
enhanced_frequency_df = pd.DataFrame([
    {
        'Feature': feature, 
        'Appearances': count, 
        'Avg_SHAP': feature_avg_shap[feature],
        'Models': ', '.join([m for m in shap_models['Model'].values if feature in pd.read_csv(f'serialGraph/{m.replace(" ", "_")}_shap_values.csv').head(10)['Feature'].tolist()])
    }
    for feature, count in sorted_features
])

enhanced_frequency_df.to_csv('serialGraph/feature_frequency_analysis.csv', index=False)
print(f"\n📁 Enhanced feature frequency analysis saved to: serialGraph/feature_frequency_analysis.csv")
print("Includes tie-breaking based on average SHAP values across all models")

# Plot ROC curves
plt.figure(figsize=(12, 8))
for name, data in roc_curves_data.items():
    plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC={data['auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Serial Transcriptomic Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('serialGraph/serial_transcriptomic_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Individual ROC curves
for name, data in roc_curves_data.items():
    plt.figure(figsize=(8, 6))
    plt.plot(data['fpr'], data['tpr'], 'b-', linewidth=2, label=f"{name} (AUC={data['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'serialGraph/{name.replace(" ", "_").replace("-", "_")}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Comprehensive visualization of all models' test AUCs
print("\n--- Generating Comprehensive Visualizations ---")

# 1. Bar chart of all test AUCs
plt.figure(figsize=(14, 8))
models_list = results_df['Model'].tolist()
aucs_list = results_df['Test AUC'].tolist()

# Create color-coded bars
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

bars = plt.bar(range(len(models_list)), aucs_list, color=colors[:len(models_list)], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (bar, auc) in enumerate(zip(bars, aucs_list)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Test AUC Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison - Serial Transcriptomic Analysis', fontsize=14, fontweight='bold')
plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('serialGraph/serial_transcriptomic_test_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Horizontal bar chart
plt.figure(figsize=(12, 10))
y_pos = np.arange(len(models_list))
bars = plt.barh(y_pos, aucs_list, color=colors[:len(models_list)], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for i, (bar, auc) in enumerate(zip(bars, aucs_list)):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{auc:.3f}', ha='left', va='center', fontweight='bold')

plt.yticks(y_pos, models_list)
plt.xlabel('Test AUC Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Ranking - Serial Transcriptomic Analysis', fontsize=14, fontweight='bold')
plt.xlim(0, 1.1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('serialGraph/serial_transcriptomic_test_auc_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. AUC ranking chart
plt.figure(figsize=(12, 8))
sorted_results = results_df.sort_values('Test AUC', ascending=True)
y_pos = np.arange(len(sorted_results))
bars = plt.barh(y_pos, sorted_results['Test AUC'], color=colors[:len(sorted_results)], alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for i, (bar, auc) in enumerate(zip(bars, sorted_results['Test AUC'])):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{auc:.3f}', ha='left', va='center', fontweight='bold')

plt.yticks(y_pos, sorted_results['Model'])
plt.xlabel('Test AUC Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Ranking (Ascending) - Serial Transcriptomic Analysis', fontsize=14, fontweight='bold')
plt.xlim(0, 1.1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('serialGraph/serial_transcriptomic_auc_ranking.png', dpi=300, bbox_inches='tight')
plt.show()

print("All results and visualizations saved to 'serialGraph' folder:")
print("- serialGraph/serial_transcriptomic_results.csv")
print("- serialGraph/serial_transcriptomic_test_auc_comparison.png")
print("- serialGraph/serial_transcriptomic_test_auc_horizontal.png")
print("- serialGraph/serial_transcriptomic_auc_ranking.png")
print("- All individual ROC curve PNGs in serialGraph/ folder")
print("- serialGraph/serial_transcriptomic_roc_curves.png")
print("- Feature importance CSV files in serialGraph/ folder")
print("- SHAP plots for 4 selected models ranked by AUC performance (high to low)")
print("- SHAP values CSV files with top features for each model")
print("- Skipped Random Forest and SVM SHAP plots as requested")
