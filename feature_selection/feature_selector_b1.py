import pandas as pd
import numpy as np
import os
import logging
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from scipy.stats import f_oneway
import warnings

warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, data, target_column, out_folder):
        """
        Initialize the feature selector.

        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.
        - out_folder: Directory where output files will be saved.
        """
        self.data = data
        self.target_column = target_column
        self.out_folder = out_folder

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        logging.basicConfig(filename=os.path.join(out_folder, 'feature_selector.log'), level=logging.DEBUG)

    def relieff(self, data, target_column):
        """
        Apply the ReliefF algorithm to rank features based on their importance.
    
        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.
    
        Returns:
        - results: DataFrame containing features ranked by their ReliefF score.
        """
        features = data.drop(columns=[target_column, 'ID_1']).values
        target = data[target_column].values
    
        fs = ReliefF(n_neighbors=100)
        fs.fit(features, target)
        scores = fs.feature_importances_
    
        feature_names = data.drop(columns=[target_column, 'ID_1']).columns
        results = pd.DataFrame({'Feature': feature_names, 'ReliefF Score': scores})
    
        return results

    def welch_anova(self, data, target_column):
        """
        Apply the Welch's ANOVA algorithm to rank features based on their importance.
    
        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.
    
        Returns:
        - results: DataFrame containing features ranked by their Welch's ANOVA F-value.
        """
        features = data.select_dtypes(include=[np.number]).drop(columns=[target_column])
        groups = [features[data[target_column] == label] for label in data[target_column].unique()]
        F, p_values = f_oneway(*groups)
    
        results = pd.DataFrame({
            'Feature': features.columns,
            'F-Value': F,
            'P-Value': p_values
        }).sort_values(by='P-Value')
    
        return results
        
    def bootstrap_feature_selection(self, n_bootstraps, n_samples=None, stratify=True, tests=['relieff', 'welch_anova']):
        for i in range(n_bootstraps):
            if stratify:
                bootstrap_sample = resample(self.data, n_samples=n_samples, stratify=self.data[self.target_column], replace=True)
            else:
                bootstrap_sample = resample(self.data, n_samples=n_samples, replace=True)
            for test in tests:
                if test == 'relieff':
                    test_results = self.relieff(bootstrap_sample, self.target_column)
                elif test == 'welch_anova':
                    test_results = self.welch_anova(bootstrap_sample, self.target_column)
                
                test_results.to_csv(os.path.join(self.out_folder, f'{os.path.basename(self.data_file).replace(".csv", "")}_bootstrap_{i+1}_{test}_results.csv'), index=False)

def main():
    dataset_paths = [
        "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/ad_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/dlb_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/mci_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/nph_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/vad_vs_nc.csv"
    ]

    dataset_paths = [
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/ad_nc_female.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/ad_nc_male.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/dlb_nc_female.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/dlb_nc_male.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/mci_nc_female.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/mci_nc_male.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/nph_nc_female.csv", 
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/nph_nc_male.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/vad_nc_female.csv",
        # "/home/aghasemi/CompBio481/datasets/processed_datasets/usable_datasets_branch1/vad_nc_male.csv"
    ]

    for dataset_path in dataset_paths:
        dataset = pd.read_csv(dataset_path)
        dataset = dataset.drop(columns=['Age', 'Sex', 'APOE4'], errors='ignore')  # Drop columns if present

        out_folder = os.path.join('/home/aghasemi/CompBio481/feature_selection/feat_select_res_branch1_overall/', os.path.basename(dataset_path).replace(".csv", ""))
        fs = FeatureSelector(dataset, 'Diagnosis', out_folder)
        fs.data_file = dataset_path  # Keep track of the current dataset file path
        
        fs.bootstrap_feature_selection(n_bootstraps=3, n_samples=len(dataset), stratify=True, tests=['relieff'])

if __name__ == "__main__":
    main()
