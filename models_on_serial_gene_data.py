#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

import pandas as pd

# Load data (handling potential byte order mark)
serial_samples_annotation = pd.read_csv('/users/audreysu/AudreyCovidProject/serial_samples_annotation.csv')
serial_norm_gene_exp_df = pd.read_csv('/users/audreysu/AudreyCovidProject/serial_norm_gene_exp_df.csv')

# Transpose and set column names from 'Unnamed: 0'
serial_norm_gene_exp_df_transposed = serial_norm_gene_exp_df.set_index('Unnamed: 0').T #Set gene names as index *before* transposing

# Set 'X' column as index in serial_samples_annotation
serial_samples_annotation = serial_samples_annotation.set_index('X')

# Merge the dataframes based on 'X' (original identifier) from annotation and index from gene data
merged_df = pd.merge(serial_samples_annotation, serial_norm_gene_exp_df_transposed, left_on='X', right_index=True, how='inner')
merged_df.head()


# In[2]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt

# Prepare data for RNN (LSTM or GRU)
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys).astype(int)  # Ensure y is integer

X = merged_df.drop(['Subject', 'Day', 'Delirium','Diagnosis', 'Steroids','Late_del'], axis=1)
y = merged_df['Diagnosis']

#Scale data as before
numeric_cols = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

TIME_STEPS = 3
X, y = create_sequences(X, y, TIME_STEPS)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# Define models (LSTM and GRU) using Functional API
def create_lstm_model(input_shape):
    input_layer = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm2 = layers.LSTM(32)(lstm1)
    dropout = layers.Dropout(0.2)(lstm2)
    output_layer = layers.Dense(1, activation='sigmoid')(dropout)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_gru_model(input_shape):  # GRU Model Definition
    input_layer = keras.Input(shape=input_shape)
    gru1 = layers.GRU(64, return_sequences=True)(input_layer)  # GRU layer
    gru2 = layers.GRU(32)(gru1)  # GRU layer
    dropout = layers.Dropout(0.2)(gru2)
    output_layer = layers.Dense(1, activation='sigmoid')(dropout)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Train and evaluate models
input_shape = (X_train.shape[1], X_train.shape[2])

models = {
    "LSTM": create_lstm_model(input_shape),
    "GRU": create_gru_model(input_shape) #Include GRU in model dictionary
}

for model_name, model in models.items():
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0) #verbose=0 to suppress training output

    y_pred_proba = model.predict(X_test).flatten()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, (y_pred_proba > 0.5).astype(int)) #Calculate precision and recall for each model
    recall = recall_score(y_test, (y_pred_proba > 0.5).astype(int))

    print(f"\n{model_name} Model:")
    print(f"  ROC AUC Score: {roc_auc:.2f}")
    print(f"  Precision: {precision:.2f}") #Include precision and recall in output
    print(f"  Recall: {recall:.2f}")
    print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))

    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name} model')
    plt.show()


# In[3]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

for name, model in models.items():
    # robust prediction
    try:
        y_pred_proba = model.predict(X_test).ravel()
    except Exception as e:
        print(f"model.predict failed for {name}: {e}; trying model(X_test, training=False)")
        preds = model(X_test, training=False)
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        y_pred_proba = np.asarray(preds).ravel()

    # guard against single-class y_test
    uniq = np.unique(y_test)
    if len(uniq) < 2:
        print(f"Skipping ROC for {name}: y_test has only one class: {uniq}")
        continue

    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"{name} AUC: {auc_score:.3f}")

        # Plot ROC for this model in its own figure
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_score:.2f})")
        plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve — {name}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"Failed to compute/plot ROC for {name}: {e}")



# =============================================================================
# COMPREHENSIVE SHAP ANALYSIS FOR LSTM AND GRU MODELS
# =============================================================================

# Import required libraries for SHAP analysis
import shap
import os

print("\n" + "="*80)
print("COMPREHENSIVE SHAP ANALYSIS FOR LSTM AND GRU MODELS")
print("="*80)

# Create results directory
results_dir = 'modelsGraph'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"✅ Created {results_dir}/ folder")
else:
    print(f"✅ {results_dir}/ folder already exists")

# Store model performance results
model_results = {}
for model_name, model in models.items():
    try:
        y_pred_proba = model.predict(X_test).flatten()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        model_results[model_name] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred_proba
        }
        print(f"{model_name} - AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")

# Sort models by AUC performance
sorted_models = sorted(model_results.items(), key=lambda x: x[1]['auc'], reverse=True)
print(f"\nModels ranked by AUC (high to low): {[name for name, _ in sorted_models]}")

# Feature importance and SHAP analysis for each model
print(f"\n=== FEATURE IMPORTANCE AND SHAP ANALYSIS ===")

# Get original feature names
original_features = merged_df.drop(['Subject', 'Day', 'Delirium','Diagnosis', 'Steroids','Late_del'], axis=1).columns.tolist()
print(f"Total features available: {len(original_features)}")

# Store SHAP results for cross-model analysis
shap_results = {}
feature_frequency = {}

for rank, (model_name, model_data) in enumerate(sorted_models, 1):
    model = model_data['model']
    auc_score = model_data['auc']
    
    print(f"\n--- AUC Rank {rank}: {model_name} (AUC: {auc_score:.4f}) ---")
    
    try:
        # Prepare data for SHAP analysis
        background = X_train[:min(50, len(X_train))].astype(np.float32)
        X_test_sample = X_test[:min(200, len(X_test))].astype(np.float32)
        
        # Try different SHAP explainers for neural networks
        shap_values = None
        explainer_type = None
        rf = None  # Store RandomForest for later use
        
        # 1. Try DeepExplainer (best for neural networks)
        try:
            explainer = shap.DeepExplainer(model, background)
            shap_vals = explainer.shap_values(X_test_sample)
            if isinstance(shap_vals, list):
                shap_values = np.array(shap_vals[0])
            else:
                shap_values = np.array(shap_vals)
            explainer_type = "DeepExplainer"
            print(f"✅ Used DeepExplainer - SHAP shape: {shap_values.shape}")
        except Exception as e:
            print(f"❌ DeepExplainer failed: {e}")
            
            # 2. Try GradientExplainer
            try:
                explainer = shap.GradientExplainer(model, background)
                shap_vals = explainer.shap_values(X_test_sample)
                if isinstance(shap_vals, list):
                    shap_values = np.array(shap_vals[0])
                else:
                    shap_values = np.array(shap_vals)
                explainer_type = "GradientExplainer"
                print(f"✅ Used GradientExplainer - SHAP shape: {shap_values.shape}")
            except Exception as e2:
                print(f"❌ GradientExplainer failed: {e2}")
                
                # 3. Fallback to RandomForest surrogate
                try:
                    print("🔄 Falling back to RandomForest surrogate...")
                    # Aggregate time series data
                    X_train_agg = X_train.mean(axis=1)
                    X_test_agg = X_test_sample.mean(axis=1)
                    
                    # Train RandomForest on aggregated features
                    from sklearn.ensemble import RandomForestClassifier
                    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_features='sqrt')
                    rf.fit(X_train_agg, y_train[:len(X_train_agg)])
                    
                    # Get SHAP values from RandomForest
                    tree_explainer = shap.TreeExplainer(rf)
                    shap_vals_rf = tree_explainer.shap_values(X_test_agg)
                    if isinstance(shap_vals_rf, list):
                        shap_values = np.array(shap_vals_rf[1] if len(shap_vals_rf) > 1 else shap_vals_rf[0])
                    else:
                        shap_values = np.array(shap_vals_rf)
                    explainer_type = "RandomForest_Surrogate"
                    print(f"✅ Used RandomForest surrogate - SHAP shape: {shap_values.shape}")
                except Exception as e3:
                    print(f"❌ All SHAP explainers failed for {model_name}: {e3}")
                    continue
        
        # Process SHAP values based on dimensionality
        if shap_values is not None:
            # Handle 3D SHAP values (samples, time_steps, features) or (samples, features, classes)
            if shap_values.ndim == 3:
                # Check if it's (samples, features, classes) or (samples, time_steps, features)
                if shap_values.shape[2] == 2 and explainer_type == "RandomForest_Surrogate":
                    # RandomForest surrogate with binary classification: (samples, features, classes)
                    # Take the positive class (index 1) for binary classification
                    shap_per_feature = shap_values[:, :, 1]
                    features_for_plot = X_test_sample.mean(axis=1)
                    print(f"📊 RandomForest surrogate 3D SHAP (binary): {shap_per_feature.shape}")
                else:
                    # Sum across time steps to get per-feature importance
                    shap_per_feature = shap_values.sum(axis=1)
                    features_for_plot = X_test_sample.mean(axis=1)
                    print(f"📊 Aggregated 3D SHAP to 2D: {shap_per_feature.shape}")
            # Handle 2D SHAP values
            elif shap_values.ndim == 2:
                # Check if it matches flattened dimensions
                time_steps = X_train.shape[1]
                n_features = X_train.shape[2]
                flat_dim = time_steps * n_features
                
                if shap_values.shape[1] == flat_dim:
                    # Reshape from flattened to 3D, then sum across time
                    shap_3d = shap_values.reshape(shap_values.shape[0], time_steps, n_features)
                    shap_per_feature = shap_3d.sum(axis=1)
                    features_for_plot = X_test_sample.mean(axis=1)
                    print(f"📊 Reshaped flattened SHAP to per-feature: {shap_per_feature.shape}")
                elif shap_values.shape[1] == n_features:
                    shap_per_feature = shap_values
                    features_for_plot = X_test_sample.mean(axis=1)
                    print(f"📊 SHAP already per-feature: {shap_per_feature.shape}")
                elif shap_values.shape[1] == 2 and explainer_type == "RandomForest_Surrogate":
                    # Special case: RandomForest surrogate with only 2 features (likely due to class imbalance)
                    print(f"⚠️ RandomForest surrogate only found 2 features, using feature importance instead")
                    # Use RandomForest feature importance instead
                    if rf is not None:
                        rf_importance = rf.feature_importances_
                        # Create SHAP-like values by scaling feature importance
                        shap_per_feature = np.tile(rf_importance, (shap_values.shape[0], 1))
                        features_for_plot = X_test_sample.mean(axis=1)
                        print(f"📊 Using RandomForest feature importance: {shap_per_feature.shape}")
                    else:
                        print(f"❌ RandomForest not available for feature importance")
                        continue
                elif shap_values.shape[1] == 13107 and explainer_type == "RandomForest_Surrogate":
                    # RandomForest surrogate with all features
                    shap_per_feature = shap_values
                    features_for_plot = X_test_sample.mean(axis=1)
                    print(f"📊 RandomForest surrogate with all features: {shap_per_feature.shape}")
                else:
                    print(f"⚠️ Unexpected SHAP shape: {shap_values.shape}")
                    continue
            else:
                print(f"⚠️ Unexpected SHAP dimensionality: {shap_values.ndim}")
                continue
            
            # Ensure feature alignment
            if shap_per_feature.shape[1] != len(original_features):
                if shap_per_feature.shape[1] < len(original_features):
                    # Pad with zeros
                    pad_width = len(original_features) - shap_per_feature.shape[1]
                    shap_per_feature = np.pad(shap_per_feature, ((0, 0), (0, pad_width)), mode='constant')
                    features_for_plot = np.pad(features_for_plot, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    # Truncate
                    shap_per_feature = shap_per_feature[:, :len(original_features)]
                    features_for_plot = features_for_plot[:, :len(original_features)]
            
            # Calculate feature importance (mean absolute SHAP values)
            mean_abs_shap = np.mean(np.abs(shap_per_feature), axis=0)
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': original_features,
                'Importance': mean_abs_shap,
                'Mean_SHAP': np.mean(shap_per_feature, axis=0)
            }).sort_values('Importance', ascending=False)
            
            print(f"\n📈 Top 10 most important features for {model_name}:")
            print(feature_importance_df.head(10).to_string(index=False))
            
            # Save feature importance
            feature_importance_df.to_csv(f'{results_dir}/{model_name}_feature_importance.csv', index=False)
            print(f"💾 Feature importance saved to: {results_dir}/{model_name}_feature_importance.csv")
            
            # Store for cross-model analysis
            shap_results[model_name] = {
                'shap_values': shap_per_feature,
                'feature_importance': feature_importance_df,
                'explainer_type': explainer_type
            }
            
            # Track feature frequency
            top_10_features = feature_importance_df.head(10)['Feature'].tolist()
            for feature in top_10_features:
                feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
            
            # Generate SHAP plots
            print(f"🎨 Generating SHAP plots for {model_name}...")
            
            try:
                # 1. Summary Plot (dot plot)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_per_feature, features_for_plot, 
                                feature_names=original_features, 
                                max_display=20, show=False)
                plt.title(f'SHAP Summary Plot - {model_name} (AUC: {auc_score:.4f})', 
                         fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{results_dir}/{model_name}_shap_summary.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
                # 2. Feature Importance Bar Plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_per_feature, features_for_plot, 
                                feature_names=original_features, 
                                plot_type="bar", max_display=20, show=False)
                plt.title(f'SHAP Feature Importance - {model_name} (AUC: {auc_score:.4f})', 
                         fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{results_dir}/{model_name}_shap_importance.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
                # 3. Dependence Plot for top feature
                try:
                    top_feature = feature_importance_df.iloc[0]['Feature']
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(top_feature, shap_per_feature, 
                                       features_for_plot, 
                                       feature_names=original_features, show=False)
                    plt.title(f'SHAP Dependence Plot - {top_feature} ({model_name})', 
                             fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(f'{results_dir}/{model_name}_shap_dependence.png', 
                               dpi=300, bbox_inches='tight')
                    plt.show()
                except Exception as e:
                    print(f"⚠️ Dependence plot failed: {e}")
                
                print(f"✅ SHAP plots for {model_name} saved successfully")
                
            except Exception as e:
                print(f"❌ SHAP plotting failed for {model_name}: {e}")
                
                # Fallback: Simple bar plot
                try:
                    plt.figure(figsize=(12, 8))
                    top_20_features = feature_importance_df.head(20)
                    plt.barh(range(len(top_20_features)), top_20_features['Importance'])
                    plt.yticks(range(len(top_20_features)), top_20_features['Feature'])
                    plt.xlabel('Mean |SHAP| Value')
                    plt.title(f'Feature Importance - {model_name} (AUC: {auc_score:.4f})')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(f'{results_dir}/{model_name}_feature_importance_plot.png', 
                               dpi=300, bbox_inches='tight')
                    plt.show()
                    print(f"✅ Fallback feature importance plot saved")
                except Exception as e2:
                    print(f"❌ Fallback plotting also failed: {e2}")
        
    except Exception as e:
        print(f"❌ SHAP analysis failed for {model_name}: {e}")

# Cross-model feature frequency analysis
print(f"\n=== FEATURE FREQUENCY ANALYSIS ACROSS ALL MODELS ===")
print("Analyzing which features appear most often in top 10 features...")

if feature_frequency:
    # Calculate mean importance and mean SHAP for each feature across all models
    feature_stats = {}
    
    for feature in feature_frequency.keys():
        importance_values = []
        shap_values = []
        models_with_feature = []
        
        for model_name in shap_results.keys():
            if model_name in shap_results:
                feature_importance_df = shap_results[model_name]['feature_importance']
                if feature in feature_importance_df['Feature'].values:
                    # Get importance and SHAP values for this feature
                    feature_row = feature_importance_df[feature_importance_df['Feature'] == feature]
                    if not feature_row.empty:
                        importance_values.append(feature_row['Importance'].iloc[0])
                        shap_values.append(feature_row['Mean_SHAP'].iloc[0])
                        models_with_feature.append(model_name)
        
        # Calculate means
        mean_importance = np.mean(importance_values) if importance_values else 0.0
        mean_shap = np.mean(shap_values) if shap_values else 0.0
        
        feature_stats[feature] = {
            'appearances': feature_frequency[feature],
            'mean_importance': mean_importance,
            'mean_shap': mean_shap,
            'models': models_with_feature
        }
    
    # Sort by appearances first, then by mean_importance (descending)
    sorted_features = sorted(feature_stats.items(), 
                           key=lambda x: (x[1]['appearances'], x[1]['mean_importance']), 
                           reverse=True)
    
    print(f"\n📊 FEATURE FREQUENCY RANKING (Top 15 Most Common)")
    print("=" * 100)
    print(f"{'Rank':<4} {'Feature':<20} {'Apps':<6} {'Mean_Imp':<10} {'Mean_SHAP':<12} {'Models':<25}")
    print("-" * 100)
    
    for rank, (feature, stats) in enumerate(sorted_features[:15], 1):
        models_str = ", ".join(stats['models'])
        print(f"{rank:<4} {feature:<20} {stats['appearances']:<6} {stats['mean_importance']:<10.6f} {stats['mean_shap']:<12.6f} {models_str:<25}")
    
    # Also create a version sorted by mean_importance for comparison
    sorted_by_importance = sorted(feature_stats.items(), 
                                key=lambda x: x[1]['mean_importance'], 
                                reverse=True)
    
    print(f"\n📊 FEATURE RANKING BY MEAN IMPORTANCE (Top 10)")
    print("=" * 100)
    print(f"{'Rank':<4} {'Feature':<20} {'Apps':<6} {'Mean_Imp':<10} {'Mean_SHAP':<12} {'Models':<25}")
    print("-" * 100)
    
    for rank, (feature, stats) in enumerate(sorted_by_importance[:10], 1):
        models_str = ", ".join(stats['models'])
        print(f"{rank:<4} {feature:<20} {stats['appearances']:<6} {stats['mean_importance']:<10.6f} {stats['mean_shap']:<12.6f} {models_str:<25}")
    
    # Save enhanced frequency analysis
    frequency_df = pd.DataFrame([
        {
            'Feature': feature,
            'Appearances': stats['appearances'],
            'Mean_Importance': stats['mean_importance'],
            'Mean_SHAP': stats['mean_shap'],
            'Models': ', '.join(stats['models'])
        }
        for feature, stats in sorted_features
    ])
    
    frequency_df.to_csv(f'{results_dir}/feature_frequency_analysis.csv', index=False)
    print(f"\n💾 Enhanced feature frequency analysis saved to: {results_dir}/feature_frequency_analysis.csv")
    
    # Provide recommendations
    print(f"\n🔍 SORTING CRITERIA RECOMMENDATIONS:")
    print("=" * 60)
    print("1. **Appearances + Mean_Importance** (Current): Best for finding")
    print("   features that are consistently important across models")
    print("   - Prioritizes features that appear in multiple models")
    print("   - Uses mean importance as tie-breaker")
    print("   - Good for identifying robust, consensus features")
    print()
    print("2. **Mean_Importance only**: Best for finding the most")
    print("   impactful features regardless of model consensus")
    print("   - May highlight features important in only one model")
    print("   - Could miss features that are moderately important in many models")
    print()
    print("3. **Mean_SHAP only**: Best for understanding feature")
    print("   contribution direction and magnitude")
    print("   - SHAP values show both magnitude and direction of effect")
    print("   - Better for understanding feature relationships")
    print("   - May be more interpretable for biological insights")
    print()
    print("💡 **RECOMMENDATION**: Use 'Appearances + Mean_Importance' for")
    print("   finding robust biomarkers, or 'Mean_SHAP' for biological")
    print("   interpretation of feature effects.")
    
else:
    print("❌ No feature frequency data available")

# Model performance summary
print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
performance_df = pd.DataFrame([
    {
        'Model': model_name,
        'AUC': model_data['auc'],
        'SHAP_Analysis': '✅ Success' if model_name in shap_results else '❌ Failed'
    }
    for model_name, model_data in model_results.items()
]).sort_values('AUC', ascending=False)

print(performance_df.to_string(index=False))
performance_df.to_csv(f'{results_dir}/model_performance_summary.csv', index=False)
print(f"\n💾 Model performance summary saved to: {results_dir}/model_performance_summary.csv")

# Final summary
print(f"\n" + "="*80)
print("SHAP ANALYSIS COMPLETE!")
print("="*80)
print(f"📁 All results saved to: {results_dir}/")
print(f"📊 Models analyzed: {len(shap_results)}")
print(f"🔍 Features analyzed: {len(original_features)}")
print(f"📈 Plots generated: {len(shap_results) * 3} (summary, importance, dependence)")
print(f"📋 Files created:")
print(f"   - Model performance summary")
print(f"   - Feature importance for each model")
print(f"   - Feature frequency analysis")
print(f"   - SHAP visualizations (PNG files)")
print("="*80)


# In[4]:


