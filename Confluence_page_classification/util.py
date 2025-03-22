import sys
import time
from datetime import datetime
from datetime import date
from os import environ
import pandas as pd
import numpy as np
import json
from functools import wraps
import re
from dotenv import load_dotenv
from Confluence_page_classification import global_config, logger
from pathlib import Path


def custom_cache(func):
    """
    Eigener Cache-Decorator, der auch DICTs cachen kann und die Cache-Statistik zurückgibt.
    Einfach mit @custom_cache über die Funktion drüber und schon wird gecached.
    Aufruf der Cache-Statistik mit dekorierter <Funktion>.cache_info(), z.B.
    cache_statistik = fetch_issue_details.cache_info()
    print(f"Cache Hits: {cache_statistik['hits']}, Cache Misses: {cache_statistik['misses']}")
    :param func:
    :return:
    """
    cache_dict = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize cache statistics if not already present
        if not hasattr(wrapper, 'hits'):
            wrapper.hits = 0
        if not hasattr(wrapper, 'misses'):
            wrapper.misses = 0

        # Create a unique key from arguments by serializing them
        key = json.dumps({'args': [dict(a) if isinstance(a, dict) else a for a in args],
                          'kwargs': {k: dict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}},
                         sort_keys=True)

        if key in cache_dict:
            wrapper.hits += 1
            return cache_dict[key]

        wrapper.misses += 1
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result

    def cache_info():
        """Return cache statistics"""
        return {'hits': getattr(wrapper, 'hits', 0), 'misses': getattr(wrapper, 'misses', 0)}

    wrapper.cache_info = cache_info
    return wrapper


def timeit(f):
    """
    timeit-decorator
    :param f:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        logger.info(f'func: {f.__name__} took: {te - ts:.5} sec')
        return result

    return timed


class Util:
    """
    Diverse Utility-Methoden.
    """

    @staticmethod
    def remove_between_tags(begin_tag, end_tag, string) -> str:
        """
        Sucht nach Start-Tag und ab dort nach end_tag. Der Text dazwischen wird entfernt und so lange
        wiederholt bis Start-Tag nicht mehr vorhanden ist.

        :param begin_tag: Start-Tag, ab wo entfernt werden soll
        :param end_tag: End-Tag, bis wohin entfernt werden soll
        :param string: Der zu befummelnde String
        :return: Der befummelte String
        """
        l_iterator = 0
        l_len = len(string)
        while string.find(begin_tag) > 0:
            if len(string) > l_len:
                breakpoint("Stringlänge wurde größer statt kleiner. Hier ist was faul.")
            l_len = len(string)
            l_iterator += 1
            if string.find(begin_tag) > 0:
                start_pos = string.find(begin_tag)
                end_pos = string.find(end_tag, start_pos) + len(end_tag)
                string = string[:start_pos] + string[end_pos:]

            if l_iterator > 200:
                breakpoint()  # Was ist hier los? Gibt's echt 200x etwas zu ersetzen oder simma in Endlos-Schleife?

        return string

    @staticmethod
    def get_logger():
        """
        Logger wird im __init__ initialisiert. Hier geben wir ihn zurück
        :return: Den Logger.
        """
        return logger

    @staticmethod
    def get_confluence_space_name() -> str:
        return global_config.get_config("confluence_space_name", optional=False)

    @staticmethod
    def get_base_url() -> str:
        return global_config.get_config("base_url", optional=False)

    @staticmethod
    def get_base_url_for_pages() -> str:
        return f"{Util.get_base_url()}/pages/viewpage.action?pageId="

    @staticmethod
    def get_base_url_for_issues() -> str:
        return f"{global_config.get_config('JIRA_base_url')}/browse/"

    @staticmethod
    def get_current_date_formatted():
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def get_date_from_confluence_short_date(confluence_date: str) -> date:
        # Date as it is used in Tasks and generally in HTML-Output (dd.mm.yyyy)
        if not confluence_date:
            return None
        return date(datetime.strptime(confluence_date, "%d.%m.%Y"))

    @staticmethod
    def get_date_from_confluence_api_date(confluence_api_date: str) -> date:
        # '2024-01-26T19:55:30.000+01:00'
        if not confluence_api_date:
            return None
        parsed_date = datetime.strptime(confluence_api_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        date_only = parsed_date.date()
        return date_only

    @staticmethod
    def get_current_date_formatted_for_filename():
        return datetime.now().strftime("%Y%m%d_%H%M")

    @staticmethod
    def load_env_file():
        """
        Ließt das .env-File und schreibt die beiden Parameter conf_user und conf_pwd in das Environment
        Die Parameter werden von der JIRA und CONFLUENCE-Instanz für die Anmeldung verwendet.
        """
        try:
            load_dotenv()
        except FileNotFoundError:
            logger.critical(f"Du hast kein .env-File. Abbruch. Check README.md")
            sys.exit()

        if not environ.get("CONF_USER"):
            logger.critical(f"Im .env-File fehlt die Variable 'CONF_USER'")
            sys.exit("Check log und README.MD (CONF_USER")
        if not environ.get("CONF_PWD"):
            logger.critical(f"Im .env-File fehlt die Variable 'CONF_PWD'")
            sys.exit("Check log und README.MD (CONF_PWD)")

    @staticmethod
    def sort_dictionary_by_key(input_dict: dict) -> dict:

        sorted_keys = sorted([key for key in input_dict.keys() if key is not None])
        sorted_dict = {key: input_dict[key] for key in sorted_keys}

        # Add entries with None keys at the end
        none_keys = [key for key in input_dict.keys() if key is None]
        for key in none_keys:
            sorted_dict[key] = input_dict[key]

        return sorted_dict

    @staticmethod
    def write_delta_dataframe_to_excel(current_df: pd.DataFrame, prev_df: pd.DataFrame, step_number):
        """
        Searches in two dataframes for differences and writes them to an Excel file
        :param current_df:
        :param prev_df:
        :param step_number:
        :return:
        """
        if not global_config.get_config('save_intermediate_ml_results', optional=True, default_value=False):
            logger.debug("save_intermediate_ml_results is set to False. No delta-file written. Check config.toml")
            return

        base_path = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data'))
        # Find new columns
        new_columns = set(current_df.columns) - set(prev_df.columns)

        # Find changed values
        changed_data = pd.DataFrame()
        for column in current_df.columns:
            if column in prev_df.columns:
                # Identify rows where values have changed
                changed_rows = current_df[column] != prev_df[column]
                if changed_rows.any():
                    if changed_data.empty:
                        changed_data = current_df.loc[changed_rows, [column]]
                    else:
                        changed_data[column] = current_df.loc[changed_rows, column]
            else:
                # Add new column data
                if changed_data.empty:
                    changed_data = current_df[[column]]
                else:
                    changed_data[column] = current_df[column]

        # Combine new columns and changed values
        if not changed_data.empty:
            file_name = Path(base_path).joinpath(
                f"ml_step_{step_number}_changes_{Util.get_current_date_formatted_for_filename()}.xlsx")
            Util.write_dataframe_to_excel(dfs=changed_data, filename=str(file_name),
                                          sheetname=f"Step_{step_number}_Changes")
            logger.info(f"Changes for step {step_number} saved in {file_name}")
        else:
            logger.info(f"No changes detected for step {step_number}. No file written.")

    @staticmethod
    def write_dataframe_to_excel(filename: str, sheetname, dfs, index=False):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_numbers': True}})
        if isinstance(dfs, list):
            # This is a list of dataframes - each should be treated separately:
            for sheet_number, df in enumerate(dfs):
                Util._transfer_dfs_to_excel_and_format(df, Util.make_valid_excel_sheet_name(sheetname[sheet_number]),
                                                       writer, index=index)
        else:
            Util._transfer_dfs_to_excel_and_format(dfs, Util.make_valid_excel_sheet_name(sheetname), writer,
                                                   index=index)
        writer.close()
        logger.info(f"File {filename} wurde erzeugt")

    @staticmethod
    def make_unique_list_entries(lst: list) -> list:
        """
        Makes all entries in a list unique
        :param lst: The list
        :return: The list with unique entries
        """
        unique_items = set()
        result = []

        for item in lst:
            if isinstance(item, dict):
                # Convert dictionary to a sorted tuple of key-value pairs
                item_tuple = tuple(sorted(item.items()))
            else:
                item_tuple = item

            # Check if the item_tuple is already in the set of unique items
            if item_tuple not in unique_items:
                unique_items.add(item_tuple)
                result.append(item)

        return result

    @staticmethod
    def make_valid_excel_sheet_name(name, existing_names=None):
        if not existing_names:
            existing_names = []

        # Trim or pad the name to ensure it is at most 31 characters
        valid_name = name[:31]

        # Replace invalid characters with underscore
        valid_name = re.sub(r'[\[\]*?/:\\]', '_', valid_name)

        # Remove leading and trailing single quotes
        valid_name = valid_name.strip("'")

        # Ensure the name is not blank after removing invalid characters
        if not valid_name:
            valid_name = "UnnamedSheet"

        # Ensure uniqueness
        original_valid_name = valid_name
        count = 1
        while valid_name.lower() in (existing.lower() for existing in existing_names):
            valid_name = f"{original_valid_name[:30 - len(str(count))]}_{count}"
            count += 1

        return valid_name

    @staticmethod
    def _transfer_dfs_to_excel_and_format(dfs, sheetname, writer, index=False):
        dfs.to_excel(writer, sheet_name=sheetname, index=index)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object

        if index:
            workbook = writer.book
            # Define a format for the index: left-align text.
            index_format = workbook.add_format({'align': 'left'})

            # Apply format to the index column (typically column A in Excel).
            worksheet.set_column('A:A', None, index_format)

        for spalten_nummer, spalte in enumerate(dfs):  # loop through all columns
            try:
                series = dfs[spalte].infer_objects(copy=False)
                series = series.astype(str)

                lengths = series.map(len)

                # Use a percentile to determine the max length, e.g., 95th percentile
                percentile = 95

                try:
                    threshold_len = np.percentile(lengths, percentile)
                except IndexError:
                    # Leeres Feld
                    threshold_len = 2

                # You may want to set a reasonable minimum width as well
                min_width = 2  # for example, adjust as needed

                # Calculate the desired column width, not exceeding the threshold
                # and considering a minimum width
                try:
                    max_len = max(min(threshold_len, lengths.max()), min_width)
                except ValueError:
                    max_len = 15

                if max_len > 50:
                    max_len = 50
                worksheet.set_column(spalten_nummer, spalten_nummer, max_len)  # set column width
            except KeyError as ex:
                logger.warning(f"Irgendwas komisch beim aufbereiten XLS. Fehler war: {ex}")
        # Setzen Autofilter
        worksheet.autofilter(0, 0, dfs.shape[0], dfs.shape[1] - 1)


