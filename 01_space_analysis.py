"""
This is a helper program.

At this point we've downloaded the data and stored in a pickle file.
The pickl file contains a list of ConfluencePage objects.

The pickl-files have the space in their name (e.g. confluence_pages_<space>_<timestamp>.pkl).

We want to provide some analysis of the contents based on the ConfluencePage-Object (count, sum, etc) various
attributes

Labels to keep are in config-file under key labels_to_keep (in upper case)
Labels to remove are in config-file under key labels_to_remove (in upper case)

unwanted page titles are in config-file under key page_title_ignore (in upper case)

"""

import pandas as pd
import numpy as np

from pathlib import Path
import pickle
from Confluence_page_classification.util import global_config, Util, logger, timeit
from Confluence_page_classification.UtilHtmlHandling import UtilHtmlHandling


class SpaceAnalysis:

    labels_to_keep = global_config.get_config('labels_to_keep')
    labels_to_remove = global_config.get_config('labels_to_remove')
    page_title_ignore = global_config.get_config('page_title_ignore')

    def __init__(self):
        self.data = None
        self.raw_data_unchanged = None
        self.dfs = []    # Export Dataframes
        self.dfs_names = []
        self.analysis_df = None
        self.analysis_results = {}

    @timeit
    def load_data(self):
        file_name = (Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).
                     joinpath(f"{global_config.get_config('confluence_space_name')}_preprocessed_data.pkl"))
        logger.info(f"Loading preprocessed data from {file_name}")
        with open(file_name, 'rb') as file:
            self.data = pickle.load(file)

        logger.info(f"Data loaded. We have {len(self.data)} entries. Starting text processing.")
        # plain_text_content is the plain text of the HTML-Content
        self.data['plain_text_content'] = self.data['content'].apply(
            lambda x: UtilHtmlHandling.extract_plain_text_from_html(x))

        logger.info("Text processing completed.")

        # if the data in "url" doesn't start with "http" we will add the base_url to it
        # Otherwise it's always a mess to find the page as you'd have to manually add the base-url.
        if not self.data['url'].str.startswith("https://").all():
            self.data['url'] = self.data['url'].apply(lambda x: global_config.get_config('confluence_base_url') + x)

        self.raw_data_unchanged = self.data.copy()
        self.dfs.append(self.raw_data_unchanged)
        self.dfs_names.append('all_data')
        logger.info(f"Preprocessed data loaded from {file_name}")

    def analyze_space(self):
        """
        We will provide additional overall analysis of the data in a separate tab in the excel file

        :return:
        """
        # self.analysis_results has already various entries from previous methods
        # We want to bring some general statistics first into the dict:

        analysis_results_top = {
            'total_pages': len(self.raw_data_unchanged),
            'average_length': self.raw_data_unchanged['plain_text_content'].apply(lambda x: len(x)).mean(),
            'total_tasks_open': self.raw_data_unchanged['tasks_open'].sum(),
            'total_tasks_closed': self.raw_data_unchanged['tasks_closed'].sum(),
            'total_incoming_links': self.raw_data_unchanged['incoming_links_count'].sum(),
            'average_age_in_days': self.raw_data_unchanged['age_in_days'].mean(),
            'average_last_updated_in_days': self.raw_data_unchanged['last_updated_in_days'].mean(),
            'average_version_count': self.raw_data_unchanged['version_count'].mean(),
            'total_version_count': self.raw_data_unchanged['version_count'].sum(),
            'total_links_count': self.raw_data_unchanged['incoming_links_count'].sum(),
            'total_word_count': self.raw_data_unchanged['plain_text_content'].apply(lambda x: len(x.split())).sum(),
        }
        # We want to round all results to integers:
        analysis_results_top = {k: np.round(v, 0)
                                if isinstance(v, (int, float)) else v for k, v in analysis_results_top.items()}

        # Add statistics about the pages that are excluded from the analysis
        for df_name in self.dfs_names:
            if df_name not in ['all_data', 'space_analysis']:
                analysis_results_top[f"Pages in tab {df_name}"] = len(self.dfs[self.dfs_names.index(df_name)])

        # Add one line for each year that pages have been created and the number of pages created in that year
        years = self.raw_data_unchanged['created'].apply(lambda x: x.year)
        years = years[years > 2000]
        years = years.value_counts().sort_index()
        analysis_results_top['Years with pages created'] = ""
        for year, count in years.items():
            analysis_results_top[f"Pages created in {year}"] = count

        # for the top 100 autors we want to know how many pages they have created
        top_authors = self.raw_data_unchanged['created_by'].value_counts().head(100)
        analysis_results_top['Top 100 Authors'] = ""
        for author, count in top_authors.items():
            analysis_results_top[f"Pages created by {author}"] = count

        self.analysis_results = {**analysis_results_top, **self.analysis_results, **{"Analysis Values": ""}}

        # Convert the analysis results dictionary to a DataFrame
        analysis_df = pd.DataFrame(list(self.analysis_results.items()), columns=['Parameter', 'Value'])

        # Append the describe DataFrame for additional statistics
        describe_df = self.raw_data_unchanged.describe().T.reset_index()
        # We want to round all results to 1 decimal places:
        describe_df = describe_df.apply(lambda x: np.round(x, 1) if x.dtype == 'float64' else x)
        describe_df.columns = ['Parameter'] + list(describe_df.columns[1:])

        # Concatenate the analysis results with the describe DataFrame
        self.analysis_df = pd.concat([analysis_df, describe_df], axis=0, ignore_index=True)

    def __collect_label_entries(self, label_name):
        """
        Labels are in self.data['labels'] as string delimited by ",".
        Creates a new dataframe for all pages that have label_name in their labels.
        Removes those pages from self.data.
        :param label_name: Label name to filter by
        :return: DataFrame containing entries with the specified label
        """
        # Ensure labels are in uppercase for comparison
        label_name_upper = label_name.upper()

        # Filter pages with the specified label
        new_dfs = self.data[self.data['labels'].apply(lambda x: label_name_upper in x.upper())].copy()
        logger.info(f"For label {label_name} we found {len(new_dfs)} entries.")

        # Remove these pages from self.data
        self.data = self.data[~self.data['labels'].apply(lambda x: label_name_upper in x.upper())]
        return new_dfs

    def process_labels_to_keep(self):
        """
        Process labels to keep.
        :return:
        """
        for label in self.labels_to_keep:
            df = self.__collect_label_entries(label)
            # Only append if we have at least one entry in the dataframe
            if not df.empty:
                self.dfs_names.append(label)
                self.dfs.append(df)

    def process_labels_to_remove(self):
        """
        Process labels to remove.
        :return:
        """
        for label in self.labels_to_remove:
            df = self.__collect_label_entries(label)
            # Only append if we have at least one entry in the dataframe
            if not df.empty:
                self.dfs_names.append(label)
                self.dfs.append(df)

    @timeit
    def collect_unwanted_page_titles(self):
        """
        Collect entries that have unwanted page titles
        unwanted page titles are a list of upper-case strings in page_title_ignore.
        We want one DFS for all those entries, but we need the title-criteria as column.
        :return:
        """
        for title in self.page_title_ignore:
            df = self.data[self.data['title'].apply(lambda x: title.upper() in x.upper())].copy()
            if not df.empty:
                self.dfs_names.append(title)
                self.dfs.append(df)
                # Remove from self.data:
                self.data = self.data[~self.data['title'].apply(lambda x: title in x)]

    @timeit
    def collect_all_other_pages_with_labels(self):
        """
        Collect all pages that have labels, that are not in labels_to_keep or labels_to_remove
        As those pages have been already removed from data we can simply search for pages with labels.
        :return:
        """

        def sort_by_value(item):
            try:
                return int(item[1])
            except ValueError:
                return item[1]

        self.dfs_names.append('all_other_pages_with_labels')
        df = self.data[self.data['labels'].apply(lambda x: x != '')]
        self.dfs.append(df)

        # Also make a list of other lables and their occurences
        other_labels = {}
        for labels in df['labels']:
            for label in labels.split(","):
                label = label.strip()
                if other_labels.get(label) is None:
                    other_labels[label] = 1
                else:
                    other_labels[label] += 1
        # Kill all labels that are assigned to less than 2 pages as insignificant
        other_labels = {k: v for k, v in other_labels.items() if v > 2}
        # Put the labels with many assignments first in the list:
        other_labels = dict(sorted(other_labels.items(), key=lambda item: item[1], reverse=True))

        # Add those to analysis sheet:
        self.analysis_results['Important Labels'] = ""
        self.analysis_results.update(other_labels)

    @timeit
    def collect_pages_with_dates_in_title(self):
        """
        pages that start with yyyy-mm-dd are usually easily outdated and irrelevant
        the page title is usually like that "2024-12-31: - sometitle" where the ":" behind the date
        might or might not be there. We will collect all pages that have a date in the title
        :return:
        """
        def is_date(date_str):
            try:
                pd.to_datetime(date_str, errors='raise', format='%Y-%m-%d')
                return True
            except ValueError:
                return False

        self.dfs_names.append('pages_with_dates_in_title')
        self.dfs.append(self.data[self.data['title'].apply(lambda x: is_date(x[:10]))].copy())
        # Remove from self.data:
        self.data = self.data[~self.data['title'].apply(lambda x: is_date(x[:10]))]

    def collect_pages_too_short(self):
        """
        pages, that are less than 500 characters long are considered too short
        :return:
        """
        self.dfs_names.append('too_short')
        # this must be a new Dataframe, as we will remove the entries from self.data
        self.dfs.append(self.data[self.data['plain_text_content'].apply(lambda x: len(x) < 500)].copy())
        # Remove from self.data:
        self.data = self.data[~self.data['plain_text_content'].apply(lambda x: len(x) < 500)]

    def save_to_excel(self):
        """
        take the DFS and DFS-Names and use Util.save_dfs_to_excel to save them to an excel file
        :return:
        """
        file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')).joinpath(
            f"01_space_analysis_{global_config.get_config('confluence_space_name')}_"
            f"{Util.get_current_date_formatted_for_filename()}.xlsx")
        Util.write_dataframe_to_excel(filename=str(file_name), dfs=self.dfs, sheetname=self.dfs_names)

    def preparations_before_save(self):
        """
        This method is called before saving the data to excel.
        We will remove "content" and "text" from all dataframes, that do have these columns
        :return:
        """
        for df in self.dfs:
            if 'content' in df.columns:
                df.drop(columns=['content'], inplace=True)
            if 'text' in df.columns:
                df.drop(columns=['text'], inplace=True)
            if 'plain_text_content' in df.columns:
                df.drop(columns=['plain_text_content'], inplace=True)

    def collect_remaining_pages(self):
        """
        Collect all remaining pages that have not been processed yet.
        :return:
        """
        # We want those to be the first sheet (Analysis will be inserted even before that)
        self.dfs_names.insert(0, 'remaining_pages')
        self.dfs.insert(0, self.data.copy())

    def run(self):
        self.load_data()
        self.process_labels_to_keep()
        self.process_labels_to_remove()
        self.collect_all_other_pages_with_labels()
        self.collect_unwanted_page_titles()
        self.collect_pages_too_short()
        self.collect_pages_with_dates_in_title()
        self.collect_remaining_pages()

        # Analysis-DF is not yet in self.dfs. We want it as the first DFS (index 0)
        self.analyze_space()
        self.dfs.insert(0, self.analysis_df)
        self.dfs_names.insert(0, 'space_analysis')

        self.preparations_before_save()


if __name__ == '__main__':
    space_analysis = SpaceAnalysis()
    space_analysis.run()
    space_analysis.save_to_excel()
    logger.info("Space analysis completed.")
    logger.info(f"Dataframe names: {space_analysis.dfs_names}")
    logger.info(f"Dataframe lengths: {[len(df) for df in space_analysis.dfs]}")
    logger.info(f"Remaining entries in self.data: {len(space_analysis.data)}")

