import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

train_datasets = [
    "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/ad_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/dlb_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/mci_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/nph_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/vad_vs_nc_train.csv",
]

test_datasets = [
    "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/ad_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/dlb_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/mci_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/nph_vs_nc_train.csv",
    # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch2_overall/vad_vs_nc_train.csv",
]

# Initialize an empty DataFrame for results
results = pd.DataFrame()

# Process each dataset pair
for train_path, test_path in zip(train_datasets, test_datasets):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Align the columns in test dataset to match the order in train dataset
    test_data = test_data[train_data.columns]

    X_train = train_data.drop(['Diagnosis', 'ID_1'], axis=1)
    y_train = train_data['Diagnosis']
    X_test = test_data.drop(['Diagnosis', 'ID_1'], axis=1)
    y_test = test_data['Diagnosis']
    
    # Ensure labels are encoded if they're categorical
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    classifiers = {
        'SVM': SVC(kernel='linear', C=1, probability=True, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42)
    }
    
    for name, clf in classifiers.items():
        rfe = RFE(estimator=clf, n_features_to_select=30)
        rfe.fit(X_train, y_train)
        selected_features = X_train.columns[rfe.support_]
        
        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)
        
        clf.fit(X_train_rfe, y_train)
        y_train_pred = clf.predict(X_train_rfe)  # Predict on training data
        y_pred = clf.predict(X_test_rfe)
        y_prob = clf.predict_proba(X_test_rfe)[:, 1]
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate training accuracy
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        aucroc = roc_auc_score(y_test, y_prob)
        
        # Creating a DataFrame for this iteration and concatenating it
        iter_results = pd.DataFrame({
            'Model': [name],
            'Dataset': [train_path.split('/')[-1]],
            'Train Accuracy': [train_accuracy],
            'Test Accuracy': [accuracy],
            'F1 Score': [f1],
            'AUC-ROC': [aucroc]
        })
        results = pd.concat([results, iter_results], ignore_index=True)

        # Feature importances or coefficients
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        else:
            importance = clf.coef_[0]
        
        importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importance})
        plt.figure(figsize=(10, 8))
        sns.heatmap(importance_df.set_index('Feature').T, cmap='viridis', annot=True)
        plt.title(f'{name} Feature Importances for {train_path.split("/")[-1]}')
        plt.show()

# Save results to CSV
results.to_csv('/home/aghasemi/CompBio481/ML_classifiers/ML_classifiers_results/ad_vs_nc_model_performance_metrics.csv', index=False)