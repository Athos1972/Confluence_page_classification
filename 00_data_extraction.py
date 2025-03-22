"""
This is a helper program.

At this point we've downloaded the data and stored in a pickle file.
The pickl file contains a list of ConfluencePage objects.

In the excel sheet from config-file with key "ml_training_file_keys" we find columns "space" and "id".
The pickl-files have the space in their name (e.g. confluence_pages_<space>_<timestamp>.pkl).

We need to find the right pick-file, extract all ConfluencePage objects and store them in a new pickle file.
the new pickl file will be named "training_pages_<space>_<timestamp>.pkl".

The resulting file will be used for training the model.

"""

import pandas as pd
from pathlib import Path
import pickle
from Confluence_page_classification.util import global_config, logger, Util
from Confluence_page_classification.UtilFileHandling import UtilFilehandling
from Confluence_page_classification.MlProcessing import MlProcessing


class DataPreparation:
    """
    This class is responsible for preparing the data for the training process.
    """
    def __init__(self):
        self.page_ids = None

    def __load_excel_file_of_page_ids(self):
        """
        Load the excel file with the page ids.
        """
        excel_file = Path.cwd().joinpath(global_config.get_config('ml_training_file_keys'))
        self.page_ids = pd.read_excel(excel_file)
        logger.info(f"Excel file with page ids loaded from {excel_file}")

    def __load_training_data(self):
        """
        Load the training data from the pickle file.
        """
        path_to_pickle = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data'))
        latest_pickle = UtilFilehandling.get_latest_file_in_folder(str(path_to_pickle), "pkl")
        with open(latest_pickle, 'rb') as file:
            data_pickled = pickle.load(file)
            self.data = pd.DataFrame([vars(page) for page in data_pickled.values()])
        logger.info(f"Training data loaded from pickle file {latest_pickle}. We have {len(self.data)} pages.")

    def __filter_data(self):
        """
        Filter the data based on the page ids from the excel file.
        """
        self.data = self.data[self.data['id'].isin(self.page_ids['id'])]
        logger.info(f"Data filtered. We have now {len(self.data)} pages.")

    def __save_data(self):
        """
        Save the data to a new pickle file.
        """
        file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            f"training_pages_{self.page_ids['space'].iloc[0]}_{Util.get_current_date_formatted_for_filename()}.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(self.data, file)
        logger.info(f"Training data saved to {file_name}")

    def run(self):
        """
        Run the data preparation process.
        """
        self.__load_excel_file_of_page_ids()
        self.__load_training_data()
        self.__filter_data()
        self.__save_data()


if __name__ == '__main__':
    data_preparation = DataPreparation()
    data_preparation.run()
