import pandas as pd
import os
import logging
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, data, target_column, out_folder):
        self.data = data
        self.target_column = target_column
        self.out_folder = out_folder
        self.data_file = ''

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        logging.basicConfig(filename=os.path.join(out_folder, 'feature_selector.log'), level=logging.DEBUG, force=True)

    def find_smallest_group_size(self):
        if self.target_column in self.data.columns:
            return self.data[self.target_column].value_counts().min()
        else:
            raise ValueError(f"The specified target column '{self.target_column}' does not exist in the dataset.")
    
    def rfe(self, max_features=10):
        X = self.data.drop([self.target_column, 'ID_1', 'Age', 'Sex', 'APOE4'], axis=1, errors='ignore')
        y = self.data[self.target_column]
    
        smallest_group_size = self.find_smallest_group_size()
        
        # Set ratio as 2:1, meaning for every 2 samples, 1 feature can be supported
        features_by_ratio = int(smallest_group_size / 2)
        
        n_features_to_select = min(features_by_ratio, max_features)
        n_features_to_select = max(n_features_to_select, 1)  # Ensure at least 1 feature is selected
    
        estimator = LogisticRegression(max_iter=500, solver='liblinear')
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        selector = selector.fit(X, y)
    
        rfe_results = pd.DataFrame({
            'Feature': X.columns,
            'Selected': selector.support_,
            'Ranking': selector.ranking_
        })
    
        return rfe_results[rfe_results['Selected'] == True]



def process_dataset(dataset_path, target_column):
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.drop(columns=['Age', 'Sex', 'APOE4'], errors='ignore')  # Drop columns if present

    out_folder = os.path.join('/home/aghasemi/CompBio481/feature_selection/feat_select_res_branch1_sex_specific_p2_refinement', os.path.basename(dataset_path).replace(".csv", ""))
    fs = FeatureSelector(dataset, target_column, out_folder)
    
    rfe_results = fs.rfe()
    rfe_results.to_csv(os.path.join(out_folder, 'rfe_results.csv'), index=False)
    print("RFE results saved to:", os.path.join(out_folder, 'rfe_results.csv'))

def main():

    dataset_paths = [
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_overall/ad_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_overall/dlb_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_overall/mci_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_overall/nph_vs_nc.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_overall/vad_vs_nc.csv",
    ] 
    dataset_paths = [
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/ad_nc_male.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/dlb_nc_male.csv",
        "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/mci_nc_male.csv",
        "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/nph_nc_male.csv",
        "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/vad_nc_male.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/ad_nc_female.csv",
        # "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/dlb_nc_female.csv",
        "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/mci_nc_female.csv",
        "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/nph_nc_female.csv",
        "/home/aghasemi/CompBio481/datasets/filtered_datasets_after_rank_feat_select_branch1_sex_specific/vad_nc_female.csv",
    ]
    
    for dataset_path in dataset_paths:
        process_dataset(dataset_path, 'Diagnosis')

if __name__ == "__main__":
    main()