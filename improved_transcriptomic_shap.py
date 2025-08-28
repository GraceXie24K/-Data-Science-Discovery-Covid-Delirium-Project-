import os
import re
import numpy as np
import pandas as pd
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
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix
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
gene_symbols = load_table('/users/audreysu/AudreyCovidProject/gene_symbols.csv')

# Data preprocessing
admission_df = admission_df.rename(columns={admission_df.columns[0]: 'SampleID'})
gene_symbols = gene_symbols.rename(columns={gene_symbols.columns[0]: 'SampleID'})

gene_map = dict(zip(gene_symbols["gene_ids"], gene_symbols["gene_symbols"]))
admission_df["Sample_ID_mapped"] = admission_df["SampleID"].map(gene_map).fillna(admission_df["SampleID"])

transcriptomic_df = admission_df.drop(columns=['SampleID'])
transcriptomic_df.columns = [transform_colname(col) for col in transcriptomic_df.columns]

demographic_df = demographic_df.head(128)
demographic_df = demographic_df.iloc[:, [0, -1]]

transcriptomic_df = transcriptomic_df.set_index("Sample_ID_mapped")
transcriptomic_df = transcriptomic_df.T
transcriptomic_df = transcriptomic_df.reset_index().rename(columns={'index': 'Master Record ID'})

merged_df = pd.merge(demographic_df, transcriptomic_df, on='Master Record ID', how='outer')
merged_df = merged_df.rename(columns={"Delirium at any time during hospitalization": "Delirium"})

target = "Delirium"
merged_df = merged_df.dropna(subset=[target]).reset_index(drop=True)

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
    
    # Suppress warnings specifically for LightGBM training
    if name == "LightGBM":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use RandomizedSearchCV for faster optimization
            random_search = RandomizedSearchCV(
                config["model"], 
                config["params"], 
                n_iter=50,  # More iterations for better optimization
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc', 
                n_jobs=-1, 
                verbose=1,
                random_state=42
            )
            
            random_search.fit(X_train_resampled, y_train_resampled)
    else:
        # Use RandomizedSearchCV for faster optimization
        random_search = RandomizedSearchCV(
            config["model"], 
            config["params"], 
            n_iter=50,  # More iterations for better optimization
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', 
            n_jobs=-1, 
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = random_search.best_estimator_
    trained_models[name] = best_model
    
    # Cross-validation score
    cv_score = random_search.best_score_
    cv_scores[name] = cv_score
    
    # Test set performance
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    roc_curves_data[name] = {"fpr": fpr, "tpr": tpr, "auc": test_auc}
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"CV AUC: {cv_score:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

# Create ensemble models
print("\n--- Creating Ensemble Models ---")

# Voting Classifier (Soft Voting)
voting_classifier = VotingClassifier(
    estimators=[(name, model) for name, model in trained_models.items()],
    voting='soft'
)

voting_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_voting = voting_classifier.predict_proba(X_test_scaled)[:, 1]
voting_auc = roc_auc_score(y_test, y_pred_voting)
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_pred_voting)

roc_curves_data["Voting Ensemble"] = {"fpr": fpr_voting, "tpr": tpr_voting, "auc": voting_auc}
print(f"Voting Ensemble AUC: {voting_auc:.4f}")

# Stacking Classifier
base_models = [(name, model) for name, model in trained_models.items()]
meta_model = LogisticRegression(random_state=42)

stacking_classifier = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stacking_classifier.fit(X_train_resampled, y_train_resampled)
y_pred_stacking = stacking_classifier.predict_proba(X_test_scaled)[:, 1]
stacking_auc = roc_auc_score(y_test, y_pred_stacking)
fpr_stacking, tpr_stacking, _ = roc_curve(y_test, y_pred_stacking)

roc_curves_data["Stacking Ensemble"] = {"fpr": fpr_stacking, "tpr": tpr_stacking, "auc": stacking_auc}
print(f"Stacking Ensemble AUC: {stacking_auc:.4f}")

# Results summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

results_df = pd.DataFrame({
    'Model': list(roc_curves_data.keys()),
    'Test AUC': [data['auc'] for data in roc_curves_data.values()],
    'CV AUC': [cv_scores.get(name, 'N/A') for name in roc_curves_data.keys()]
})

results_df = results_df.sort_values('Test AUC', ascending=False)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('graph/model_performance_results.csv', index=False)

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
plt.savefig('graph/enhanced_models_roc_curves.png', dpi=300, bbox_inches='tight')

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
    plt.savefig(f'graph/{name.replace(" ", "_").replace("-", "_")}_roc_curve.png', dpi=300, bbox_inches='tight')

print(f"\nEnhanced model training complete!")
print(f"Results saved to: graph/model_performance_results.csv")
print(f"ROC curves saved as PNG files")
print(f"Best performing model: {results_df.iloc[0]['Model']} (AUC: {results_df.iloc[0]['Test AUC']:.4f})")

# Feature importance analysis for best model
best_model_name = results_df.iloc[0]['Model']
print(f"\nFeature importance analysis for {best_model_name}:")

# Get the actual best model object
best_model = trained_models[best_model_name]

if best_model_name == "XGBoost":
    # Get XGBoost feature importance
    xgb_model = trained_models["XGBoost"]
    importance = xgb_model.feature_importances_
    
elif best_model_name == "LightGBM":
    # Get LightGBM feature importance
    lgb_model = trained_models["LightGBM"]
    importance = lgb_model.feature_importances_
    
elif best_model_name == "Random Forest":
    # Get Random Forest feature importance
    rf_model = trained_models["Random Forest"]
    importance = rf_model.feature_importances_
    
elif best_model_name == "Gradient Boosting":
    # Get Gradient Boosting feature importance
    gb_model = trained_models["Gradient Boosting"]
    importance = gb_model.feature_importances_
    
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
    feature_importance_df.to_csv(f'graph/{best_model_name.replace(" ", "_")}_feature_importance.csv', index=False)
    
    # Show top 5 genes specifically
    print(f"\nTop 5 most predictive genes for delirium:")
    top_5_genes = feature_importance_df.head(5)
    for i, (_, row) in enumerate(top_5_genes.iterrows(), 1):
        print(f"{i}. {row['Feature']} (Importance: {row['Importance']:.6f})")
    
    # SHAP Analysis for the best model
    print(f"\n--- Generating SHAP Plots for {best_model_name} ---")
    
    try:
        # Prepare data for SHAP
        X_test_scaled = scaler.transform(X_test)
        X_test_df = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        # Create SHAP explainer based on model type
        if best_model_name in ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"]:
            # Tree-based models
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_df)
            
            # For tree-based models, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
        elif best_model_name == "Logistic Regression":
            # Linear models
            explainer = shap.LinearExplainer(best_model, X_test_df)
            shap_values = explainer.shap_values(X_test_df)
            
        elif best_model_name == "SVM":
            # Kernel models - use KernelExplainer
            explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_test_df, 100))
            shap_values = explainer.shap_values(X_test_df)
            
        else:
            # Default to KernelExplainer for other models
            explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_test_df, 100))
            shap_values = explainer.shap_values(X_test_df)
        
        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, feature_names=selected_features, show=False)
        plt.title(f'SHAP Summary Plot - {best_model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph/{best_model_name.replace(" ", "_")}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_df, feature_names=selected_features, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {best_model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph/{best_model_name.replace(" ", "_")}_shap_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Force Plot for a sample prediction
        sample_idx = 0  # First test sample
        plt.figure(figsize=(12, 8))
        shap.force_plot(explainer.expected_value, shap_values[sample_idx], 
                       X_test_df.iloc[sample_idx], show=False)
        plt.title(f'SHAP Force Plot - Sample {sample_idx} ({best_model_name})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph/{best_model_name.replace(" ", "_")}_shap_force.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Dependence Plot for top feature
        top_feature = feature_importance_df.iloc[0]['Feature']
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature, shap_values, X_test_df, show=False)
        plt.title(f'SHAP Dependence Plot - {top_feature} ({best_model_name})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph/{best_model_name.replace(" ", "_")}_shap_dependence.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"SHAP plots saved to graph/ folder:")
        print(f"- graph/{best_model_name.replace(' ', '_')}_shap_summary.png")
        print(f"- graph/{best_model_name.replace(' ', '_')}_shap_importance.png")
        print(f"- graph/{best_model_name.replace(' ', '_')}_shap_force.png")
        print(f"- graph/{best_model_name.replace(' ', '_')}_shap_dependence.png")
        
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("This might be due to model type incompatibility or data issues.")

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
plt.savefig('graph/all_models_test_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

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
plt.savefig('graph/all_models_test_auc_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()

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
plt.savefig('graph/all_models_auc_ranking.png', dpi=300, bbox_inches='tight')
plt.show()

print("All results and visualizations saved to 'graph' folder:")
print("- graph/model_performance_results.csv")
print("- graph/all_models_test_auc_comparison.png")
print("- graph/all_models_test_auc_horizontal.png")
print("- graph/all_models_auc_ranking.png")
print("- All individual ROC curve PNGs in graph/ folder")
print("- graph/enhanced_models_roc_curves.png")
print("- Feature importance CSV files in graph/ folder")
print("- SHAP plots for best model in graph/ folder")
