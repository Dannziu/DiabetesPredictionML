import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint

# ============= LOAD & ANALYZE ORIGINAL DATA =============
df = pd.read_csv('diabetes.csv')

print("="*60)
print("=== ORIGINAL DATASET ANALYSIS ===")
print("="*60)
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nValue counts (Outcome):\n{df['Outcome'].value_counts()}")

# ============= VISUALIZE BEFORE PREPROCESSING =============
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = (df[cols_with_zeros] == 0).sum()

# Create a copy to preserve original for visualization
df_original = df.copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Zero values before preprocessing (from original data)
zero_counts_original = (df_original[cols_with_zeros] == 0).sum()
axes[0, 0].bar(zero_counts_original.index, zero_counts_original.values, color='#FF6384')
axes[0, 0].set_title('Zero Values Before Preprocessing', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Class distribution
outcome_counts = df_original['Outcome'].value_counts()
axes[0, 1].pie(outcome_counts.values, labels=['Negative', 'Positive'], autopct='%1.1f%%', colors=['#36A2EB', '#FF6384'])
axes[0, 1].set_title('Class Distribution (Original)', fontweight='bold', fontsize=12)

# Plot 3: Box plot before preprocessing
df_plot = df_original[cols_with_zeros].copy()
axes[1, 0].boxplot(df_plot.values, labels=cols_with_zeros)
axes[1, 0].set_title('Feature Distribution Before Preprocessing', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Value')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Heatmap of zero values
missing_df = df_original[cols_with_zeros].copy()
missing_df[missing_df == 0] = np.nan
sns.heatmap(missing_df.isnull(), cbar=True, cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Zero Values (Treated as Missing)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('data_preprocessing_before.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: data_preprocessing_before.png")

# ============= PREPROCESSING =============
print("\n" + "="*60)
print("=== DATA PREPROCESSING ===")
print("="*60)

# Treat zeros as missing
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Impute with median
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

print("\nAfter treating zeros as missing and imputing with median:")
print(f"Missing values:\n{df.isnull().sum()}")

# ============= TRANSFORMED DATA (FIRST 5 ROWS) =============
print("\n" + "="*60)
print("=== TRANSFORMED DATA (First 5 Rows) ===")
print("="*60)
print(df.head(5))

# ============= FEATURE SCALING VISUALIZATION =============
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\n" + "="*60)
print("=== FEATURE SCALING (First 5 Rows) ===")
print("="*60)
print("Before Scaling:")
print(X_train.head(5))
print("\n(Note: Scaling will be applied inside the pipeline during training)")

# ============= VISUALIZE AFTER PREPROCESSING =============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Zero values after preprocessing
zero_counts_after = (df[cols_with_zeros] == 0).sum()
axes[0, 0].bar(zero_counts_after.index, zero_counts_after.values, color='#36A2EB')
axes[0, 0].set_title('Zero Values After Preprocessing', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Before/After comparison
x_pos = np.arange(len(cols_with_zeros))
width = 0.35
before_zeros = [(df[col] == 0).sum() for col in cols_with_zeros]
after_zeros = [0] * len(cols_with_zeros)
axes[0, 1].bar(x_pos - width/2, before_zeros, width, label='Before', color='#FF6384')
axes[0, 1].bar(x_pos + width/2, after_zeros, width, label='After', color='#36A2EB')
axes[0, 1].set_ylabel('Zero Count')
axes[0, 1].set_title('Before vs After Preprocessing', fontweight='bold', fontsize=12)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(cols_with_zeros, rotation=45)
axes[0, 1].legend()

# Plot 3: Box plot after preprocessing
df_plot_after = df[cols_with_zeros].copy()
axes[1, 0].boxplot(df_plot_after.values, labels=cols_with_zeros)
axes[1, 0].set_title('Feature Distribution After Preprocessing', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Value')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Statistical summary
stats_summary = df[cols_with_zeros].describe().loc[['mean', 'std', 'min', 'max']]
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_summary.round(2).values,
                         rowLabels=stats_summary.index,
                         colLabels=stats_summary.columns,
                         cellLoc='center',
                         loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
axes[1, 1].set_title('Statistical Summary After Preprocessing', fontweight='bold', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('data_preprocessing_after.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: data_preprocessing_after.png")

# ============= MODEL TRAINING WITH HYPERPARAMETER TUNING =============
print("\n" + "="*60)
print("=== MODEL TRAINING ===")
print("="*60)

pipeline = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

param_dist = {
    'clf__n_estimators': randint(100, 500),
    'clf__max_depth': randint(3, 30),
    'clf__min_samples_split': randint(2, 10),
    'clf__min_samples_leaf': randint(1, 6),
    'clf__max_features': ['sqrt', 'log2', 0.2, 0.5, None],
    'clf__class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=40,
                            scoring='roc_auc', n_jobs=-1, cv=cv, random_state=42, verbose=1)

search.fit(X_train, y_train)
best = search.best_estimator_
print(f"\nBest parameters: {search.best_params_}")

# ============= PREDICTIONS & EVALUATION =============
print("\n" + "="*60)
print("=== MODEL EVALUATION ===")
print("="*60)

y_pred_train = best.predict(X_train)
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\nTraining Accuracy: {accuracy_train:.4f}")
print(f"Testing Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation scores
cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"\nCross-Validation ROC AUC Scores: {cv_scores}")
print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============= LOSS & ACCURACY GRAPHS =============
print("\n" + "="*60)
print("=== GENERATING PERFORMANCE GRAPHS ===")
print("="*60)

# Plot 1: Training vs Testing Accuracy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

accuracies = [accuracy_train, accuracy_test]
labels_acc = ['Training', 'Testing']
colors_acc = ['#1976d2', '#FF6384']
axes[0, 0].bar(labels_acc, accuracies, color=colors_acc, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Training vs Testing Accuracy', fontweight='bold', fontsize=12)
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.03, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 2: All Performance Metrics
metrics_vals = [accuracy_test, precision, recall, f1, roc_auc]
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']
colors_metrics = ['#1976d2', '#64b5f6', '#FF6384', '#FFB6C1', '#90EE90']
axes[0, 1].bar(metrics_labels, metrics_vals, color=colors_metrics, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Score', fontweight='bold')
axes[0, 1].set_title('Model Performance Metrics', fontweight='bold', fontsize=12)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(metrics_vals):
    axes[0, 1].text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

# Plot 3: Cross-Validation Scores
axes[1, 0].plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linewidth=2, markersize=8, color='#1976d2')
axes[1, 0].axhline(y=cv_scores.mean(), color='#FF6384', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
axes[1, 0].fill_between(range(1, len(cv_scores) + 1), cv_scores - cv_scores.std(), cv_scores + cv_scores.std(), alpha=0.2, color='#1976d2')
axes[1, 0].set_xlabel('Fold', fontweight='bold')
axes[1, 0].set_ylabel('ROC AUC Score', fontweight='bold')
axes[1, 0].set_title('Cross-Validation Scores', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1, 1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}', color='#1976d2', linewidth=2)
axes[1, 1].plot([0,1],[0,1],'--', color='gray', linewidth=1.5)
axes[1, 1].set_xlabel('False Positive Rate', fontweight='bold')
axes[1, 1].set_ylabel('True Positive Rate', fontweight='bold')
axes[1, 1].set_title('ROC Curve', fontweight='bold', fontsize=12)
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: model_evaluation.png")

# ============= SAVE ARTIFACTS =============
print("\n" + "="*60)
print("=== SAVING MODEL ARTIFACTS ===")
print("="*60)

joblib.dump(best, 'diabetes_pipeline.joblib')
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(best.named_steps['clf'], f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(best.named_steps['scaler'], f)

print("✓ Saved: diabetes_pipeline.joblib")
print("✓ Saved: diabetes_model.pkl")
print("✓ Saved: scaler.pkl")

print("\n" + "="*60)
print("=== PREPROCESSING & MODEL TRAINING COMPLETE ===")
print("="*60)
print("\nGenerated Visualizations:")
print("  1. data_preprocessing_before.png - Original data analysis")
print("  2. data_preprocessing_after.png - Cleaned data analysis")
print("  3. model_evaluation.png - Loss, accuracy, and performance metrics")
print("\n" + "="*60)