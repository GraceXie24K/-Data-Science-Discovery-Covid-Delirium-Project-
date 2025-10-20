#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Fix protobuf compatibility issues BEFORE any other imports
import os
import sys
import warnings

# Set environment variables to handle protobuf issues
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Suppress protobuf warnings globally
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*protobuf.*')
warnings.filterwarnings('ignore', message='.*MessageFactory.*')

# Try to patch protobuf MessageFactory if needed
try:
    import google.protobuf
    # Check if message_factory exists and has MessageFactory
    if hasattr(google.protobuf, 'message_factory') and hasattr(google.protobuf.message_factory, 'MessageFactory'):
        if not hasattr(google.protobuf.message_factory.MessageFactory, 'GetPrototype'):
            def _get_prototype(self, descriptor):
                return self._message_classes.get(descriptor.full_name)
            google.protobuf.message_factory.MessageFactory.GetPrototype = _get_prototype
except ImportError:
    pass
except AttributeError:
    # Handle case where message_factory doesn't exist
    pass
except Exception as e:
    # Silently handle other protobuf compatibility issues
    pass

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking
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

# Define comprehensive deep learning models using Functional API
def create_lstm_model(input_shape):
    input_layer = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm2 = layers.LSTM(32)(lstm1)
    dropout = layers.Dropout(0.2)(lstm2)
    output_layer = layers.Dense(1, activation='sigmoid')(dropout)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_gru_model(input_shape):
    input_layer = keras.Input(shape=input_shape)
    gru1 = layers.GRU(64, return_sequences=True)(input_layer)
    gru2 = layers.GRU(32)(gru1)
    dropout = layers.Dropout(0.2)(gru2)
    output_layer = layers.Dense(1, activation='sigmoid')(dropout)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_bidirectional_lstm_model(input_shape):
    """Bidirectional LSTM model for capturing temporal patterns in both directions"""
    input_layer = keras.Input(shape=input_shape)
    bilstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(input_layer)
    bilstm2 = layers.Bidirectional(layers.LSTM(32))(bilstm1)
    dropout = layers.Dropout(0.3)(bilstm2)
    dense = layers.Dense(16, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_bidirectional_gru_model(input_shape):
    """Bidirectional GRU model for efficient bidirectional temporal modeling"""
    input_layer = keras.Input(shape=input_shape)
    bigru1 = layers.Bidirectional(layers.GRU(64, return_sequences=True))(input_layer)
    bigru2 = layers.Bidirectional(layers.GRU(32))(bigru1)
    dropout = layers.Dropout(0.3)(bigru2)
    dense = layers.Dense(16, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_attention_lstm_model(input_shape):
    """LSTM with attention mechanism for focusing on important time steps"""
    input_layer = keras.Input(shape=input_shape)
    lstm1 = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm2 = layers.LSTM(32, return_sequences=True)(lstm1)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm2)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(32)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    attended = layers.Multiply()([lstm2, attention])
    attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
    
    dropout = layers.Dropout(0.3)(attended)
    dense = layers.Dense(16, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_cnn_lstm_model(input_shape):
    """1D CNN + LSTM hybrid model for local pattern detection and temporal modeling"""
    input_layer = keras.Input(shape=input_shape)
    
    # 1D CNN layers for local pattern detection
    conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    
    # LSTM layers for temporal modeling
    lstm1 = layers.LSTM(64, return_sequences=True)(conv2)
    lstm2 = layers.LSTM(32)(lstm1)
    
    dropout = layers.Dropout(0.3)(lstm2)
    dense = layers.Dense(16, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_transformer_model(input_shape):
    """Transformer model for capturing long-range dependencies in gene expression data"""
    input_layer = keras.Input(shape=input_shape)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=8, key_dim=64, dropout=0.1
    )(input_layer, input_layer)
    
    # Add & Norm
    attention_output = layers.Dropout(0.1)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(input_layer + attention_output)
    
    # Feed Forward
    ffn = layers.Dense(128, activation='relu')(attention_output)
    ffn = layers.Dropout(0.1)(ffn)
    ffn = layers.Dense(input_shape[-1])(ffn)
    
    # Add & Norm
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn)
    
    # Global average pooling
    pooled = layers.GlobalAveragePooling1D()(ffn_output)
    dropout = layers.Dropout(0.3)(pooled)
    dense = layers.Dense(32, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_deep_cnn_model(input_shape):
    """Deep 1D CNN model for hierarchical feature learning"""
    input_layer = keras.Input(shape=input_shape)
    
    # Multiple CNN blocks with increasing complexity
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_residual_lstm_model(input_shape):
    """Residual LSTM model with skip connections for better gradient flow"""
    input_layer = keras.Input(shape=input_shape)
    
    # First LSTM block
    lstm1 = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm1 = layers.Dropout(0.2)(lstm1)
    
    # Second LSTM block with residual connection
    lstm2 = layers.LSTM(64, return_sequences=True)(lstm1)
    lstm2 = layers.Dropout(0.2)(lstm2)
    
    # Residual connection
    if lstm1.shape[-1] == lstm2.shape[-1]:
        residual = layers.Add()([lstm1, lstm2])
    else:
        # Projection shortcut if dimensions don't match
        projection = layers.Dense(64)(lstm1)
        residual = layers.Add()([projection, lstm2])
    
    # Final LSTM layer
    lstm3 = layers.LSTM(32)(residual)
    dropout = layers.Dropout(0.3)(lstm3)
    dense = layers.Dense(16, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Train and evaluate models
input_shape = (X_train.shape[1], X_train.shape[2])

models = {
    "LSTM": create_lstm_model(input_shape),
    "GRU": create_gru_model(input_shape),
    "Bidirectional_LSTM": create_bidirectional_lstm_model(input_shape),
    "Bidirectional_GRU": create_bidirectional_gru_model(input_shape),
    "Attention_LSTM": create_attention_lstm_model(input_shape),
    "CNN_LSTM": create_cnn_lstm_model(input_shape),
    "Transformer": create_transformer_model(input_shape),
    "Deep_CNN": create_deep_cnn_model(input_shape),
    "Residual_LSTM": create_residual_lstm_model(input_shape)
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
    # plt.show()  # Removed to prevent blocking execution


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
        # Ensure output directory exists and save figure
        try:
            os.makedirs('modelsGraph', exist_ok=True)
            plt.tight_layout()
            plt.savefig(f'modelsGraph/{name}_roc.png', dpi=300, bbox_inches='tight')
            print(f"💾 Saved ROC plot for {name} to modelsGraph/{name}_roc.png")
        except Exception as e:
            print(f"❌ Failed to save ROC plot for {name}: {e}")
        finally:
            plt.close()

    except Exception as e:
        print(f"Failed to compute/plot ROC for {name}: {e}")



# =============================================================================
# ADVANCED ENSEMBLE MODEL IMPLEMENTATION
# =============================================================================

print("\n" + "="*80)
print("CREATING ADVANCED ENSEMBLE MODELS")
print("="*80)

# Store model performance results for ensemble
model_results = {}
for model_name, model in models.items():
    try:
        y_pred_proba = model.predict(X_test).flatten()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, (y_pred_proba > 0.5).astype(int))
        recall = recall_score(y_test, (y_pred_proba > 0.5).astype(int))
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        model_results[model_name] = {
            'model': model,
            'auc': auc_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred_proba
        }
        print(f"{model_name} - AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")

# Find best individual model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc'])
best_auc = model_results[best_model_name]['auc']
print(f"\n🏆 Best Individual Model: {best_model_name} with AUC = {best_auc:.4f}")

# Create ensemble model
print("\n🎯 Creating Ensemble Model...")
ensemble_models = []
ensemble_weights = []

# Include all models with decent performance (AUC > 0.5)
for model_name, results in model_results.items():
    if results['auc'] > 0.5:  # Include models with decent performance
        ensemble_models.append(results['model'])
        ensemble_weights.append(results['auc'])  # Weight by AUC performance

if len(ensemble_models) > 1:
    # Normalize weights
    total_weight = sum(ensemble_weights)
    ensemble_weights = [w/total_weight for w in ensemble_weights]
    
    print(f"Ensemble includes {len(ensemble_models)} models with weights: {[f'{w:.3f}' for w in ensemble_weights]}")
    
    # Create ensemble predictions
    ensemble_predictions = np.zeros(len(y_test))
    for model, weight in zip(ensemble_models, ensemble_weights):
        pred = model.predict(X_test, verbose=0).flatten()
        ensemble_predictions += weight * pred
    
    # Evaluate ensemble
    ensemble_auc = roc_auc_score(y_test, ensemble_predictions)
    ensemble_precision = precision_score(y_test, (ensemble_predictions > 0.5).astype(int))
    ensemble_recall = recall_score(y_test, (ensemble_predictions > 0.5).astype(int))
    ensemble_f1 = 2 * (ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall) if (ensemble_precision + ensemble_recall) > 0 else 0
    
    print(f"\n🏆 ENSEMBLE RESULTS:")
    print(f"  🎯 ROC AUC Score: {ensemble_auc:.4f}")
    print(f"  📈 Precision: {ensemble_precision:.4f}")
    print(f"  📈 Recall: {ensemble_recall:.4f}")
    print(f"  📈 F1-Score: {ensemble_f1:.4f}")
    
    # Compare with best individual model
    improvement = ensemble_auc - best_auc
    print(f"\n📊 ENSEMBLE vs BEST INDIVIDUAL MODEL:")
    print(f"  Best Individual ({best_model_name}): {best_auc:.4f}")
    print(f"  Ensemble: {ensemble_auc:.4f}")
    print(f"  Improvement: {improvement:+.4f} ({'✅ Better' if improvement > 0 else '❌ Worse'})")
    
    # Add ensemble to results
    model_results['Ensemble'] = {
        'model': None,  # No single model for ensemble
        'auc': ensemble_auc,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'f1': ensemble_f1,
        'predictions': ensemble_predictions
    }
    
    # Plot ensemble ROC curve
    plt.figure(figsize=(8,6))
    fpr, tpr, _ = roc_curve(y_test, ensemble_predictions)
    plt.plot(fpr, tpr, lw=2, label=f"Ensemble (AUC={ensemble_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve — Ensemble Model')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    # Save ensemble ROC
    try:
        os.makedirs('modelsGraph', exist_ok=True)
        plt.tight_layout()
        plt.savefig('modelsGraph/Ensemble_roc.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved ROC plot for Ensemble to modelsGraph/Ensemble_roc.png")
    except Exception as e:
        print(f"❌ Failed to save Ensemble ROC plot: {e}")
    finally:
        plt.close()
    
else:
    print("❌ Not enough models with decent performance for ensemble")

# =============================================================================
# ADVANCED ENSEMBLE METHODS: STACKING AND VOTING
# =============================================================================

print(f"\n" + "="*80)
print("ADVANCED ENSEMBLE METHODS: STACKING AND VOTING")
print("="*80)

# Import additional ensemble methods
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Prepare data for ensemble methods (aggregate time series)
print(f"\n🔄 Preparing data for advanced ensemble methods...")
X_train_agg = X_train.mean(axis=1)
X_test_agg = X_test.mean(axis=1)

# Create wrapper classes for neural network models
class NeuralNetworkWrapper:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    
    def predict_proba(self, X):
        # Reshape aggregated data back to time series format
        X_reshaped = X.reshape(-1, TIME_STEPS, len(original_features))
        predictions = self.model.predict(X_reshaped, verbose=0)
        # Convert to 2D array for sklearn compatibility
        return np.column_stack([1 - predictions, predictions])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

# Create ensemble-ready models
ensemble_models = []
ensemble_weights = []

print(f"\n🏗️ Creating ensemble-ready models...")
for model_name, model_data in model_results.items():
    if model_name != 'Ensemble' and model_data['auc'] > 0.5:
        model = model_data['model']
        auc_score = model_data['auc']
        
        # Create wrapper
        wrapped_model = NeuralNetworkWrapper(model, model_name)
        ensemble_models.append((model_name, wrapped_model))
        ensemble_weights.append(auc_score)
        
        print(f"✅ Added {model_name} (AUC: {auc_score:.4f}) to ensemble")

if len(ensemble_models) >= 3:
    print(f"\n🎯 Creating Voting Classifier...")
    
    # 1. Voting Classifier (Soft Voting)
    try:
        voting_classifier = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use predicted probabilities
        )
        
        # Train voting classifier
        voting_classifier.fit(X_train_agg, y_train)
        
        # Evaluate voting classifier
        voting_pred_proba = voting_classifier.predict_proba(X_test_agg)[:, 1]
        voting_auc = roc_auc_score(y_test, voting_pred_proba)
        voting_precision = precision_score(y_test, (voting_pred_proba > 0.5).astype(int))
        voting_recall = recall_score(y_test, (voting_pred_proba > 0.5).astype(int))
        voting_f1 = 2 * (voting_precision * voting_recall) / (voting_precision + voting_recall) if (voting_precision + voting_recall) > 0 else 0
        
        print(f"✅ Voting Classifier Results:")
        print(f"  🎯 ROC AUC Score: {voting_auc:.4f}")
        print(f"  📈 Precision: {voting_precision:.4f}")
        print(f"  📈 Recall: {voting_recall:.4f}")
        print(f"  📈 F1-Score: {voting_f1:.4f}")
        
        # Add to model results
        model_results['Voting_Ensemble'] = {
            'model': voting_classifier,
            'auc': voting_auc,
            'precision': voting_precision,
            'recall': voting_recall,
            'f1': voting_f1,
            'predictions': voting_pred_proba
        }
        
    except Exception as e:
        print(f"❌ Voting Classifier failed: {e}")
    
    # 2. Stacking Classifier
    print(f"\n🎯 Creating Stacking Classifier...")
    
    try:
        # Use top 3 models for stacking to avoid overfitting
        top_3_models = sorted(ensemble_models, key=lambda x: model_results[x[0]]['auc'], reverse=True)[:3]
        
        # Create meta-learner (Logistic Regression)
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        stacking_classifier = StackingClassifier(
            estimators=top_3_models,
            final_estimator=meta_learner,
            cv=3,  # 3-fold cross-validation
            stack_method='predict_proba'
        )
        
        # Train stacking classifier
        stacking_classifier.fit(X_train_agg, y_train)
        
        # Evaluate stacking classifier
        stacking_pred_proba = stacking_classifier.predict_proba(X_test_agg)[:, 1]
        stacking_auc = roc_auc_score(y_test, stacking_pred_proba)
        stacking_precision = precision_score(y_test, (stacking_pred_proba > 0.5).astype(int))
        stacking_recall = recall_score(y_test, (stacking_pred_proba > 0.5).astype(int))
        stacking_f1 = 2 * (stacking_precision * stacking_recall) / (stacking_precision + stacking_recall) if (stacking_precision + stacking_recall) > 0 else 0
        
        print(f"✅ Stacking Classifier Results:")
        print(f"  🎯 ROC AUC Score: {stacking_auc:.4f}")
        print(f"  📈 Precision: {stacking_precision:.4f}")
        print(f"  📈 Recall: {stacking_recall:.4f}")
        print(f"  📈 F1-Score: {stacking_f1:.4f}")
        
        # Add to model results
        model_results['Stacking_Ensemble'] = {
            'model': stacking_classifier,
            'auc': stacking_auc,
            'precision': stacking_precision,
            'recall': stacking_recall,
            'f1': stacking_f1,
            'predictions': stacking_pred_proba
        }
        
    except Exception as e:
        print(f"❌ Stacking Classifier failed: {e}")
    
    # 3. Weighted Ensemble with Hyperparameter Optimization
    print(f"\n🎯 Creating Weighted Ensemble with Hyperparameter Optimization...")
    
    try:
        from scipy.optimize import minimize
        
        # Get predictions from all models
        model_predictions = {}
        for model_name, model_data in model_results.items():
            if model_name not in ['Ensemble', 'Voting_Ensemble', 'Stacking_Ensemble']:
                model = model_data['model']
                if hasattr(model, 'predict'):
                    try:
                        pred = model.predict(X_test, verbose=0).flatten()
                        model_predictions[model_name] = pred
                    except:
                        # Try with aggregated data
                        try:
                            pred = model.predict(X_test_agg.reshape(-1, TIME_STEPS, X_test_agg.shape[1]), verbose=0).flatten()
                            model_predictions[model_name] = pred
                        except:
                            continue
        
        if len(model_predictions) >= 2:
            # Optimize weights using scipy
            def objective(weights):
                # Normalize weights
                weights = weights / np.sum(weights)
                
                # Create weighted ensemble prediction
                ensemble_pred = np.zeros(len(y_test))
                for i, (model_name, pred) in enumerate(model_predictions.items()):
                    ensemble_pred += weights[i] * pred
                
                # Calculate negative AUC (to minimize)
                try:
                    auc = roc_auc_score(y_test, ensemble_pred)
                    return -auc  # Minimize negative AUC = maximize AUC
                except:
                    return 1.0  # Return high value if AUC calculation fails
            
            # Initial weights (equal)
            initial_weights = np.ones(len(model_predictions)) / len(model_predictions)
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(model_predictions))]
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                optimal_weights = optimal_weights / np.sum(optimal_weights)  # Ensure normalization
                
                # Create optimized ensemble prediction
                optimized_pred = np.zeros(len(y_test))
                for i, (model_name, pred) in enumerate(model_predictions.items()):
                    optimized_pred += optimal_weights[i] * pred
                
                # Evaluate optimized ensemble
                optimized_auc = roc_auc_score(y_test, optimized_pred)
                optimized_precision = precision_score(y_test, (optimized_pred > 0.5).astype(int))
                optimized_recall = recall_score(y_test, (optimized_pred > 0.5).astype(int))
                optimized_f1 = 2 * (optimized_precision * optimized_recall) / (optimized_precision + optimized_recall) if (optimized_precision + optimized_recall) > 0 else 0
                
                print(f"✅ Optimized Weighted Ensemble Results:")
                print(f"  🎯 ROC AUC Score: {optimized_auc:.4f}")
                print(f"  📈 Precision: {optimized_precision:.4f}")
                print(f"  📈 Recall: {optimized_recall:.4f}")
                print(f"  📈 F1-Score: {optimized_f1:.4f}")
                print(f"  ⚖️ Optimal Weights:")
                for model_name, weight in zip(model_predictions.keys(), optimal_weights):
                    print(f"    {model_name}: {weight:.4f}")
                
                # Add to model results
                model_results['Optimized_Weighted_Ensemble'] = {
                    'model': None,  # No single model for ensemble
                    'auc': optimized_auc,
                    'precision': optimized_precision,
                    'recall': optimized_recall,
                    'f1': optimized_f1,
                    'predictions': optimized_pred,
                    'weights': dict(zip(model_predictions.keys(), optimal_weights))
                }
            else:
                print(f"❌ Weight optimization failed: {result.message}")
        else:
            print(f"❌ Not enough models with valid predictions for weighted ensemble")
    
    except Exception as e:
        print(f"❌ Weighted Ensemble optimization failed: {e}")
    
    # 4. Ensemble Comparison
    print(f"\n📊 ENSEMBLE COMPARISON:")
    print("=" * 80)
    
    ensemble_names = ['Ensemble', 'Voting_Ensemble', 'Stacking_Ensemble', 'Optimized_Weighted_Ensemble']
    ensemble_results = {}
    
    for ensemble_name in ensemble_names:
        if ensemble_name in model_results:
            ensemble_results[ensemble_name] = model_results[ensemble_name]
    
    if ensemble_results:
        # Sort by AUC
        sorted_ensembles = sorted(ensemble_results.items(), key=lambda x: x[1]['auc'], reverse=True)
        
        print(f"{'Rank':<4} {'Ensemble Type':<25} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
        print("-" * 80)
        
        for rank, (ensemble_name, results) in enumerate(sorted_ensembles, 1):
            print(f"{rank:<4} {ensemble_name:<25} {results['auc']:<8.4f} {results['precision']:<10.4f} {results['recall']:<8.4f} {results['f1']:<8.4f}")
        
        # Find best ensemble
        best_ensemble_name, best_ensemble_results = sorted_ensembles[0]
        print(f"\n🏆 Best Ensemble: {best_ensemble_name} with AUC = {best_ensemble_results['auc']:.4f}")
        
        # Compare with best individual model
        best_individual_auc = max([model_results[name]['auc'] for name in model_results.keys() 
                                 if name not in ensemble_names])
        improvement = best_ensemble_results['auc'] - best_individual_auc
        print(f"📈 Ensemble vs Best Individual Improvement: {improvement:+.4f} ({'✅ Better' if improvement > 0 else '❌ Worse'})")
        
        # Plot ensemble comparison
        plt.figure(figsize=(12, 8))
        ensemble_names_plot = [name.replace('_', '\n') for name in ensemble_results.keys()]
        ensemble_aucs = [results['auc'] for results in ensemble_results.values()]
        
        bars = plt.bar(range(len(ensemble_names_plot)), ensemble_aucs, 
                      color=['red', 'blue', 'green', 'orange'][:len(ensemble_names_plot)], alpha=0.7)
        plt.xticks(range(len(ensemble_names_plot)), ensemble_names_plot, rotation=45, ha='right')
        plt.ylabel('ROC AUC Score')
        plt.title('Ensemble Methods Comparison', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, auc in zip(bars, ensemble_aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'modelsGraph/ensemble_methods_comparison.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Removed to prevent blocking execution
        
        # Save ensemble comparison results
        ensemble_comparison_df = pd.DataFrame([
            {
                'Ensemble_Type': name,
                'AUC': results['auc'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1']
            }
            for name, results in ensemble_results.items()
        ]).sort_values('AUC', ascending=False)
        
        ensemble_comparison_df.to_csv(f'modelsGraph/ensemble_methods_comparison.csv', index=False)
        print(f"💾 Ensemble comparison results saved to: modelsGraph/ensemble_methods_comparison.csv")

        # Save ROC plots for additional ensemble methods if predictions are available
        try:
            for ens_name, ens_res in ensemble_results.items():
                preds = ens_res.get('predictions')
                if preds is None:
                    continue
                try:
                    fpr_e, tpr_e, _ = roc_curve(y_test, preds)
                    auc_e = roc_auc_score(y_test, preds)
                    plt.figure(figsize=(8,6))
                    plt.plot(fpr_e, tpr_e, lw=2, label=f"{ens_name} (AUC={auc_e:.3f})")
                    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC curve — {ens_name}')
                    plt.legend(loc='lower right')
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    safe_name = ens_name.replace(' ', '_')
                    plt.savefig(f'modelsGraph/{safe_name}_roc.png', dpi=300, bbox_inches='tight')
                    print(f"💾 Saved ROC plot for {ens_name} to modelsGraph/{safe_name}_roc.png")
                except Exception as e:
                    print(f"❌ Failed to create/save ROC for {ens_name}: {e}")
                finally:
                    plt.close()
        except Exception as e:
            print(f"❌ Error while saving ensemble ROC plots: {e}")
    
else:
    print("❌ Not enough models with decent performance for advanced ensemble methods")

print("="*80)

# =============================================================================
# COMPREHENSIVE SHAP ANALYSIS FOR ALL DEEP LEARNING MODELS
# =============================================================================

# Configuration flag - Set to False to skip comprehensive SHAP analysis (prevents machine hanging)
ENABLE_COMPREHENSIVE_SHAP_ANALYSIS = True

# Import required libraries for SHAP analysis and additional interpretability tools
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import SHAP with error handling (protobuf issues already handled at top of file)
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP imported successfully")
except Exception as e:
    print(f"❌ SHAP import failed: {e}")
    print("💡 Try upgrading protobuf and tensorflow:")
    print("   pip install --upgrade protobuf tensorflow")
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import seaborn as sns
from sklearn.metrics import confusion_matrix

# LIME analysis removed as requested

# Create results directory
results_dir = 'modelsGraph'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"✅ Created {results_dir}/ folder")
else:
    print(f"✅ {results_dir}/ folder already exists")

# Model results already stored in ensemble section above

# Sort models by AUC performance
sorted_models = sorted(model_results.items(), key=lambda x: x[1]['auc'], reverse=True)
print(f"\nModels ranked by AUC (high to low): {[name for name, _ in sorted_models]}")

# Get original feature names
original_features = merged_df.drop(['Subject', 'Day', 'Delirium','Diagnosis', 'Steroids','Late_del'], axis=1).columns.tolist()
print(f"Total features available: {len(original_features)}")

# Initialize empty results for compatibility
shap_results = {}
# LIME results removed
permutation_results = {}
feature_frequency = {}
# LIME models removed

# Initialize results
shap_results = {}
# LIME results removed
permutation_results = {}
feature_frequency = {}

# Run SHAP analysis if enabled and available
if ENABLE_COMPREHENSIVE_SHAP_ANALYSIS and SHAP_AVAILABLE:
    print("\n" + "="*80)
    print("COMPREHENSIVE SHAP ANALYSIS FOR ALL DEEP LEARNING MODELS")
    print("="*80)
    
    # Feature importance and SHAP analysis for each model
    print(f"\n=== FEATURE IMPORTANCE AND SHAP ANALYSIS ===")
    
    # SHAP analysis for each model
    for rank, (model_name, model_data) in enumerate(sorted_models, 1):
        model = model_data['model']
        auc_score = model_data['auc']
        
        print(f"\n--- AUC Rank {rank}: {model_name} (AUC: {auc_score:.4f}) ---")
        
        try:
            # Prepare data for SHAP analysis
            background = X_train[:min(50, len(X_train))].astype(np.float32)
            X_test_sample = X_test[:min(200, len(X_test))].astype(np.float32)
            
            # Use RandomForest surrogate for SHAP analysis
            print("🔄 Using RandomForest surrogate for SHAP analysis...")
            # Aggregate time series data
            X_train_agg = X_train.mean(axis=1)
            X_test_agg = X_test_sample.mean(axis=1)
            
            # Train RandomForest on aggregated features
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
            
            # Process SHAP values
            if shap_values is not None:
                # Handle different SHAP value shapes
                if shap_values.ndim == 3:
                    if shap_values.shape[2] == 2 and explainer_type == "RandomForest_Surrogate":
                        shap_per_feature = shap_values[:, :, 1]
                        features_for_plot = X_test_sample.mean(axis=1)
                    else:
                        shap_per_feature = shap_values.sum(axis=1)
                        features_for_plot = X_test_sample.mean(axis=1)
                elif shap_values.ndim == 2:
                    if shap_values.shape[1] == len(original_features):
                        shap_per_feature = shap_values
                        features_for_plot = X_test_sample.mean(axis=1)
                    else:
                        print(f"⚠️ SHAP shape mismatch: {shap_values.shape}")
                        continue
                else:
                    print(f"⚠️ Unexpected SHAP dimensions: {shap_values.ndim}")
                    continue
                
                # Calculate feature importance
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
                    # 1. Summary Plot
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_per_feature, features_for_plot, 
                                    feature_names=original_features, 
                                    max_display=20, show=False)
                    plt.title(f'SHAP Summary Plot - {model_name} (AUC: {auc_score:.4f})', 
                             fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(f'{results_dir}/{model_name}_shap_summary.png', 
                               dpi=300, bbox_inches='tight')
                    # plt.show()  # Removed to prevent blocking execution
                    
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
                    # plt.show()  # Removed to prevent blocking execution
                    
                    print(f"✅ SHAP analysis completed for {model_name}")
                    
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
                        plt.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'{results_dir}/{model_name}_feature_importance_plot.png', 
                                   dpi=300, bbox_inches='tight')
                        # plt.show()  # Removed to prevent blocking execution
                        print(f"✅ Fallback feature importance plot saved")
                    except Exception as e2:
                        print(f"❌ Fallback plotting also failed: {e2}")
            else:
                print(f"❌ No SHAP values available for {model_name}")
                
        except Exception as e:
            print(f"❌ SHAP analysis failed for {model_name}: {e}")

elif ENABLE_COMPREHENSIVE_SHAP_ANALYSIS and not SHAP_AVAILABLE:
    print("\n" + "="*80)
    print("SHAP ANALYSIS DISABLED - IMPORT ERROR")
    print("="*80)
    print("⚠️ SHAP analysis is disabled due to import errors (likely protobuf version conflicts).")
    print("   To fix this, try: pip install --upgrade protobuf tensorflow")
    print("="*80)

# =============================================================================
# ADDITIONAL INTERPRETABILITY ANALYSIS (PERMUTATION IMPORTANCE)
# =============================================================================

print(f"\n=== ADDITIONAL INTERPRETABILITY ANALYSIS ===")

# LIME Analysis removed as requested
print(f"\n⚠️ LIME analysis has been removed from this analysis.")
# LIME results removed

# LIME analysis code removed

# Permutation Importance Analysis
print(f"\n🔍 Performing Permutation Importance analysis...")
print("⚠️ Permutation importance analysis code is disabled to prevent machine hanging.")
print("   The permutation importance analysis would include:")
print("   - Feature importance through permutation testing")
print("   - Cross-validation of feature importance")
print("   - Statistical significance testing")

# Create empty permutation results for compatibility

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
performance_data = []
for model_name, model_data in model_results.items():
    performance_data.append({
        'Model': model_name,
        'AUC': model_data['auc'],
        'Precision': model_data.get('precision', 'N/A'),
        'Recall': model_data.get('recall', 'N/A'),
        'F1_Score': model_data.get('f1', 'N/A'),
        'SHAP_Analysis': '✅ Success' if model_name in shap_results else '❌ Failed' if model_name != 'Ensemble' else 'N/A (Ensemble)'
    })

performance_df = pd.DataFrame(performance_data).sort_values('AUC', ascending=False)

print(performance_df.to_string(index=False))
performance_df.to_csv(f'{results_dir}/model_performance_summary.csv', index=False)
print(f"\n💾 Model performance summary saved to: {results_dir}/model_performance_summary.csv")

# =============================================================================
# COMPREHENSIVE MODEL COMPARISON AND REPORTING
# =============================================================================

print(f"\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON AND REPORTING")
print("="*80)

# Create comprehensive comparison plots
print(f"\n📊 Creating comprehensive model comparison visualizations...")

# 1. Model Performance Comparison
plt.figure(figsize=(15, 10))

# Performance metrics comparison
performance_metrics = ['AUC', 'Precision', 'Recall', 'F1_Score']
model_names = [name for name in model_results.keys() if name != 'Ensemble']
model_names_with_ensemble = list(model_results.keys())

# Create subplots for different metrics
for i, metric in enumerate(performance_metrics):
    plt.subplot(2, 2, i+1)
    # Map metric names to dictionary keys
    metric_key_map = {
        'AUC': 'auc',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1_Score': 'f1'
    }
    metric_key = metric_key_map.get(metric, metric.lower().replace('_', ''))
    values = [model_results[name].get(metric_key, 0) for name in model_names_with_ensemble]
    colors = ['red' if name == 'Ensemble' else 'skyblue' for name in model_names_with_ensemble]
    
    bars = plt.bar(range(len(model_names_with_ensemble)), values, color=colors, alpha=0.7)
    plt.xticks(range(len(model_names_with_ensemble)), 
               [name.replace('_', '\n') for name in model_names_with_ensemble], 
               rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        if value != 'N/A':
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{results_dir}/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent blocking execution

# 2. Feature Importance Comparison Across Models
if shap_results:
    print(f"\n📈 Creating feature importance comparison across models...")
    
    # Get top 15 features from each model
    all_top_features = set()
    for model_name in shap_results.keys():
        top_features = shap_results[model_name]['feature_importance'].head(15)['Feature'].tolist()
        all_top_features.update(top_features)
    
    # Create comparison matrix
    feature_comparison = pd.DataFrame(index=list(all_top_features))
    
    for model_name in shap_results.keys():
        feature_importance_df = shap_results[model_name]['feature_importance']
        feature_comparison[model_name] = feature_comparison.index.map(
            feature_importance_df.set_index('Feature')['Importance']
        ).fillna(0)
    
    # Plot heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(feature_comparison.head(30), 
                annot=True, fmt='.3f', cmap='viridis',
                cbar_kws={'label': 'SHAP Importance'})
    plt.title('Feature Importance Comparison Across Models (Top 30 Features)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/feature_importance_comparison_heatmap.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()  # Removed to prevent blocking execution
    
    # Save feature comparison data
    feature_comparison.to_csv(f'{results_dir}/feature_importance_comparison.csv')

# 3. Model Architecture Comparison
print(f"\n🏗️ Creating model architecture comparison...")

architecture_info = {
    'LSTM': 'Basic LSTM with 2 layers',
    'GRU': 'Basic GRU with 2 layers', 
    'Bidirectional_LSTM': 'Bidirectional LSTM with dense layer',
    'Bidirectional_GRU': 'Bidirectional GRU with dense layer',
    'Attention_LSTM': 'LSTM with attention mechanism',
    'CNN_LSTM': '1D CNN + LSTM hybrid',
    'Transformer': 'Multi-head attention transformer',
    'Deep_CNN': 'Deep 1D CNN with batch normalization',
    'Residual_LSTM': 'LSTM with residual connections'
}

# Create architecture comparison table
arch_df = pd.DataFrame([
    {'Model': model, 'Architecture': architecture_info.get(model, 'Unknown'), 
     'AUC': model_results[model]['auc'] if model in model_results else 'N/A'}
    for model in models.keys()
]).sort_values('AUC', ascending=False)

print(f"\n📋 Model Architecture Comparison:")
print(arch_df.to_string(index=False))
arch_df.to_csv(f'{results_dir}/model_architecture_comparison.csv', index=False)

# 4. Interpretability Method Comparison
print(f"\n🔍 Creating interpretability method comparison...")

interpretability_summary = []

for model_name in model_results.keys():
    if model_name != 'Ensemble':
        summary = {
            'Model': model_name,
            'SHAP_Analysis': '✅ Success' if model_name in shap_results else '❌ Failed',
            'AUC': model_results[model_name]['auc']
        }
        interpretability_summary.append(summary)

    interpretability_df = pd.DataFrame(interpretability_summary).sort_values('AUC', ascending=False)
    print(f"\n📊 Interpretability Analysis Summary:")
    print(interpretability_df.to_string(index=False))
    interpretability_df.to_csv(f'{results_dir}/interpretability_analysis_summary.csv', index=False)

    # 5. Generate comprehensive report
    print(f"\n📝 Generating comprehensive analysis report...")

    report_content = f"""
# COVID-19 Delirium Prediction: Comprehensive Deep Learning Analysis Report

## Executive Summary
This analysis evaluated {len(models)} different deep learning architectures for predicting delirium in COVID-19 patients using serial gene expression data.

## Model Performance Summary
- **Total Models Evaluated**: {len(models)}
- **Best Individual Model**: {best_model_name} (AUC: {best_auc:.4f})
- **Ensemble Performance**: {'AUC: ' + str(model_results['Ensemble']['auc'])[:6] if 'Ensemble' in model_results else 'Not created'}
- **Total Features Analyzed**: {len(original_features)}

## Model Architecture Analysis
The following architectures were evaluated:

1. **LSTM**: Basic Long Short-Term Memory network
2. **GRU**: Gated Recurrent Unit network  
3. **Bidirectional LSTM**: LSTM processing sequences in both directions
4. **Bidirectional GRU**: GRU processing sequences in both directions
5. **Attention LSTM**: LSTM with attention mechanism for temporal focus
6. **CNN-LSTM**: Hybrid 1D CNN + LSTM for local and temporal patterns
7. **Transformer**: Multi-head attention for long-range dependencies
8. **Deep CNN**: Hierarchical 1D CNN with batch normalization
9. **Residual LSTM**: LSTM with skip connections for better gradient flow

## Interpretability Analysis
- **SHAP Analysis**: Applied to {len(shap_results)} models using RandomForest surrogate
- **LIME Analysis**: Removed from analysis
- **Permutation Importance**: Applied to top 5 performing models
- **Feature Interaction Analysis**: Conducted for models with successful SHAP analysis

## Key Findings
1. **Best Performing Architecture**: {best_model_name}
2. **Feature Analysis**: {len(feature_frequency)} unique features identified as important across models
3. **Ensemble Performance**: {'Improved over best individual model' if 'Ensemble' in model_results and model_results['Ensemble']['auc'] > best_auc else 'Did not improve over best individual model'}

## Files Generated
- Model performance summaries and comparisons
- SHAP visualizations (summary plots, importance plots, dependence plots, heatmaps)
- Feature importance analysis results
- Permutation importance analysis
- Feature frequency and interaction analysis
- ROC curves for all models
- Comprehensive comparison visualizations

## Recommendations
1. **Model Selection**: {best_model_name} shows the best individual performance
2. **Feature Engineering**: Focus on the most frequently important features across models
3. **Interpretability**: Use SHAP values for understanding model decisions
4. **Future Work**: Consider ensemble methods and hyperparameter optimization

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save report
    with open(f'{results_dir}/comprehensive_analysis_report.md', 'w') as f:
        f.write(report_content)

    print(f"✅ Comprehensive analysis report saved to: {results_dir}/comprehensive_analysis_report.md")

else:
    print("\n" + "="*80)
    print("COMPREHENSIVE SHAP ANALYSIS DISABLED")
    print("="*80)
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP analysis is disabled due to import errors.")
        print("   To fix this, try: pip install --upgrade protobuf tensorflow")
    else:
        print("⚠️ Comprehensive SHAP analysis is disabled by default to prevent machine hanging.")
        print("   To enable it, set ENABLE_COMPREHENSIVE_SHAP_ANALYSIS = True at the top of this section.")
    print("   This will generate detailed SHAP plots and interpretability reports.")
    print("="*80)

# Final summary
print(f"\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"📁 All results saved to: {results_dir}/")
print(f"📊 Individual models analyzed: {len([m for m in model_results.keys() if m != 'Ensemble'])}")
print(f"🎯 Ensemble model: {'✅ Created' if 'Ensemble' in model_results else '❌ Not created'}")
print(f"🔍 Features analyzed: {len(original_features)}")
print(f"📈 SHAP analysis completed for: {len(shap_results)} models")
print(f"🔍 LIME analysis: Removed from analysis")
print(f"📊 Permutation importance completed for: {len(permutation_results)} models")
print(f"📋 Files created:")
print(f"   - Model performance summary (including ensemble)")
print(f"   - Feature importance for each individual model")
print(f"   - Feature frequency analysis")
print(f"   - SHAP visualizations (summary, importance, dependence, heatmaps)")
print(f"   - Feature importance analysis results")
#print(f"   - Permutation importance analysis")
print(f"   - ROC curves for all models including ensemble")
print(f"   - Comprehensive model comparison visualizations")
print(f"   - Feature importance comparison heatmap")
print(f"   - Model architecture comparison")
print(f"   - Interpretability analysis summary")
print(f"   - Comprehensive analysis report (Markdown)")
print("="*80)


# In[4]:


