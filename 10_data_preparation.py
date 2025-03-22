import pandas as pd
from pathlib import Path
import pickle
from Confluence_page_classification.util import global_config, logger
from Confluence_page_classification.UtilFileHandling import UtilFilehandling
from Confluence_page_classification.UtilHtmlHandling import UtilHtmlHandling
from Confluence_page_classification.UtilML import UtilML
from Confluence_page_classification.MlProcessing import MlProcessing
from transformers import BertTokenizer


class DataPreparation(MlProcessing):
    def __init__(self):
        super().__init__()
        self.set_step(1)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

    def load_downloaded_data(self):
        path_to_pickle = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data'))
        latest_pickle = UtilFilehandling.get_latest_file_in_folder(
            str(path_to_pickle), extension="pkl",
            mask=f"confluence_pages_{global_config.get_config('confluence_space_name')}*.")

        with open(latest_pickle, 'rb') as file:
            data_pickled = pickle.load(file)
            self.data = pd.DataFrame([vars(page) for page in data_pickled.values()])
        logger.info(f"Downloaded data loaded from pickle file {latest_pickle}.")

    def preprocess_text(self):
        self.data.rename(columns={"plain_text_content": "text"}, inplace=True)
        self.data['text'] = self.data['text'].apply(lambda x: UtilML.truncate_text(x, self.tokenizer))
        logger.info("Text truncation completed.")

    def save_preprocessed_data(self):
        file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            f"{global_config.get_config('confluence_space_name')}_preprocessed_data.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(self.data, file)
        logger.info(f"Preprocessed data saved to {file_name}")

    def run(self):
        self.load_downloaded_data()
        self.preprocess_text()
        self.save_preprocessed_data()


if __name__ == '__main__':
    data_preparation = DataPreparation()
    data_preparation.run()
