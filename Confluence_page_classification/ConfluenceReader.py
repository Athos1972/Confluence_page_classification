from Confluence_page_classification.util import Util, global_config, logger, timeit
from Confluence_page_classification.UtilFileHandling import UtilFilehandling
from os import environ
from time import sleep
from requests import session
from pathlib import Path
from atlassian import Confluence
from datetime import datetime, timedelta
import pickle
import urllib.parse


class ConfluenceReader:
    def __init__(self, page_limit=13000):

        self.url = global_config.get_config("confluence_base_url")
        self.username = environ.get("CONF_USER")
        self.password = environ.get("CONF_PWD")
        self.page_data = {}
        self.spaces = [global_config.get_config("confluence_space_name")]
        self.all_pages = 0
        self.page_fetch_limit = 100
        self.page_limit = page_limit
        self.session = session()
        self.session.headers = self._get_http_headers()
        self.session.auth = (self.username, self.password)

        self.read_from_disc_if_exists = global_config.get_config("read_from_disc_if_exists", optional=False)
        self.save_to_disc_when_read = global_config.get_config("save_to_disc_when_read", optional=False)

        # Create a directory for storing page buffer data
        Path.cwd().joinpath(global_config.get_config(
            "path_for_page_buffer_data")).mkdir(parents=True, exist_ok=True)
        self.path_for_page_buffer_data = Path.cwd().joinpath(global_config.get_config(
            "path_for_page_buffer_data"))

        # Create a confluence-API-Instance
        # Verbindung zu Confluence herstellen
        self.confluence = Confluence(
            url=global_config.get_config("confluence_base_url"),
            username=environ.get("CONF_USER"),
            password=environ.get("CONF_PWD")
        )

    @staticmethod
    def _get_http_headers():
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            "User-Agend": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 +'
                          '(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15'
        }
        return headers

    def _return_file_if_exists(self, **kwargs):
        """
        generate the hashed filename from the kwargs and return the file if it exists
        :param kwargs:
        :return:
        """
        filename = f"{UtilFilehandling.generate_filename_hash_from_parameters(**kwargs)}.pkl"
        file = self.path_for_page_buffer_data.joinpath(filename)
        if file.exists():
            logger.warning(f"For request {kwargs} the file {file} is loaded. Not read from Confluence-Instance!")
            with open(file, "rb") as f:
                return pickle.load(f)
        else:
            return None

    def _save_results_to_file(self, result_to_pickle, **kwargs):
        """
        Save the results to a file
        :param result_to_pickle:
        :param kwargs:
        :return:
        """
        filename = f"{UtilFilehandling.generate_filename_hash_from_parameters(**kwargs)}.pkl"
        file = self.path_for_page_buffer_data.joinpath(filename)
        with open(file, "wb") as f:
            pickle.dump(result_to_pickle, f)
        logger.info(f"Results for {kwargs} saved to {file}")

    @timeit
    def fetch_all_pages_in_space(self, space_key, page_limit=999999):

        if self.read_from_disc_if_exists:
            result = self._return_file_if_exists(space_key=space_key, page_limit=page_limit)
            if result:
                return result

        cql = f"space in ('{space_key}') AND type in ('page')"
        encoded_cql = urllib.parse.quote(cql)
        start = 0
        limit = 100
        pages = []

        while len(pages) < self.page_limit:
            data = None
            url = f"{self.url}/rest/api/search?cql={encoded_cql}&start={start}&limit={limit}"
            # check, if we had read this URL before - if so, return the result:
            if self.read_from_disc_if_exists:
                data = self._return_file_if_exists(url=url, page_limit=page_limit)
            if not data:
                logger.info(f"Sending request to {url}")
                data = self._execute_with_retry(url)
                if data and self.save_to_disc_when_read:
                    self._save_results_to_file(data, url=url, page_limit=page_limit)

            if not data:
                logger.error(f"Failed to retrieve data for {url}")
                break

            results = data.get('results', [])
            logger.info(f"Received response with {len(results)} pages.")

            if not results:
                logger.warning(f"No more results found at {url}. Stopping.")
                break

            pages.extend(results)
            start += limit

            if len(results) < limit:
                logger.info("No more pages to fetch.")
                break

        if self.save_to_disc_when_read:
            self._save_results_to_file(pages, space_key=space_key, page_limit=page_limit)

        return pages

    def get_page_watchers(self, page_id):
        """
        Get the watchers of a page by its ID
        :param page_id:
        :return:
        """
        endpoint = f"/json/listwatchers.action?pageId={page_id}"
        try:
            response = self.get_any_confluence_url_via_requests(self.url + endpoint)
            logger.debug(f"page_id {page_id}, Watcher count: {len(response.get('results', []))}")
            if response.get('pageWatchers'):
                page_watchers = [w.get('fullName', 'Unknown') for w in response.get('pageWatchers', [])]
                space_watchers = [w.get('fullName', 'Unknown') for w in response.get('spaceWatchers', [])]
                return {
                    "page_watchers": page_watchers,
                    "space_watchers": space_watchers
                }
            else:
                logger.warning(f"No watchers found or unexpected response for page {page_id}: {response}")
                return {
                    "page_watchers": [],
                    "space_watchers": []
                }
        except Exception as e:
            logger.warning(f"Error reading watchers for page_id '{page_id}': {e}")
            return {
                "page_watchers": [],
                "space_watchers": []
            }

    def get_any_confluence_url_via_requests(self, url):
        """
        Get any confluence url (or read from disc if exists)
        :param url:
        :return:
        """
        if self.read_from_disc_if_exists:
            result = self._return_file_if_exists(url=url)
            if result:
                return result
        result = self._execute_with_retry(url)
        if self.save_to_disc_when_read and result:
            if result.get("status-code", 0) > 399:
                return result
            self._save_results_to_file(result, url=url)
        return result

    def get_page_by_id(self, page_id, expand=None):
        """
        Get a page by its ID
        :param page_id:
        :param expand:
        :return:
        """
        url = f"{self.url}/rest/api/content/{page_id}"
        if expand:
            url += f"?expand={expand}"
        return self.get_any_confluence_url_via_requests(url)

    def get_attachments_by_page_id(self, page_id):
        """
        Get all attachments of a page by its ID
        :param page_id:
        :return:
        """
        url = f"{self.url}/rest/api/content/{page_id}/child/attachment"
        return self.get_any_confluence_url_via_requests(url)

    def get_page_statistics(self, page_id):
        """
        Get the statistics information of a page by its ID
        endpoints are /views and /viewers
        Return as dict "statistics" with dicts "views" and "viewers"
        :param page_id:
        :return:
        """
        views = {}
        if global_config.get_config("read_monthly_page_access_statistics", default_value=False):
            # The content analysis is available "only" for 364 days into the past.
            from_date = datetime.now() - timedelta(days=364)
            # URL-Encode the date and for whatever reason the API needs an uppercase "Z" at the end
            from_date = urllib.parse.quote(from_date.strftime("%Y-%m-%dT%H:%M:%S.000%z")) + "Z"

            to_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000%z") + "Z"
            to_date = urllib.parse.quote(to_date)

            url = (f"{self.url}/rest/confanalytics/1.0/content/viewsByDate?"
                   f"contentId={page_id}&fromDate={from_date}&toDate={to_date}"
                   f"&contentType=page&type=total&period=month&timezone=GMT%2B02%3A00")
            views = self.get_any_confluence_url_via_requests(url)

        url = f"{self.url}/rest/confanalytics/1.0/content/viewsByUser?contentId={page_id}&contentType=page"
        viewers = self.get_any_confluence_url_via_requests(url)

        return {"statistics": {"viewsByDate": views.get("viewsByDate"), "viewers": viewers.get("viewsByUser")}}

    def _execute_with_retry(self, url, retries=3):
        for i in range(retries):
            try:
                response = self.session.get(url)
                logger.debug(f"Status: {response.status_code} Headers: {response.headers} for url {url}")
                sleep(0.05)
                return response.json()
            except Exception as e:
                logger.error(f"Error while fetching {url}: {e}")
                if i == retries - 1:
                    raise e
                sleep(1)
