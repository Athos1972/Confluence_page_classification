import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from Confluence_page_classification.util import global_config, Util, logger


class Prediction:
    def __init__(self):
        self.model = None
        self.data = None
        self.features = None
        self.scaler = None
        self.features_scaled = None

    def load_model_data(self):
        model_file = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            "trained_model.pkl")
        data_file = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            "feature_engineered_data.pkl")
        with open(model_file, 'rb') as model_f, open(data_file, 'rb') as data_f:
            self.model = pickle.load(model_f)
            self.data = pickle.load(data_f)
        logger.info(f"Model and data loaded.")

    def prepare_features(self):
        self.features = pd.concat([self.data[['h1_count', 'h2_count', 'h3_count', 'tasks_open', 'tasks_closed',
                                              'page_properties', 'page_properties_report', 'user_mentions',
                                              'dates_in_text', 'incoming_links_count', 'pure_length', 'age_in_days',
                                              'last_updated_in_days', 'version_count', 'ancestors_count',
                                              'children_count', 'table_count', 'rows_count',
                                              'jira_links_count', 'confluence_links_count', 'image_in_page_count',
                                              'quality_points', 'plantuml_macros_count']],
                                   # the LDA-Fieldnames:
                                   self.data[[col for col in self.data.columns if 'LDA' in col]],
                                   pd.get_dummies(self.data['sentiment']),
                                   self.data[['readability']],
                                   # The columns from bert_embeddings should be named "bert_0", "bert_1", ...
                                   pd.DataFrame(self.data['bert_embeddings'].tolist(),
                                                columns=[f"bert_{i}" for i in range(768)])],
                                  axis=1)

        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)

        logger.info("Features prepared.")

    def predict(self):
        self.data['predicted_label'] = self.model.predict(self.features_scaled)

        reverse_label_mapping = {1: 'golden_ml', 0: 'archiv_ml', -1: 'unsicher_ml'}
        self.data['predicted_label'] = self.data['predicted_label'].map(reverse_label_mapping)

        file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            f"predictions_{Util.get_current_date_formatted_for_filename()}.xlsx")
        Util.write_dataframe_to_excel(dfs=self.data, filename=str(file_name), sheetname="Predictions")
        logger.info(f"Predictions saved in {file_name}")

    def run(self):
        self.load_model_data()
        self.prepare_features()
        self.predict()


if __name__ == '__main__':
    prediction = Prediction()
    prediction.run()
