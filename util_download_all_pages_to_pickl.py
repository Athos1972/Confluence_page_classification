from html import unescape
from Confluence_page_classification.util import Util, global_config, logger, timeit
from Confluence_page_classification.UtilHtmlHandling import UtilHtmlHandling
from Confluence_page_classification.ConfluenceReader import ConfluenceReader
from Confluence_page_classification.ConfluencePage import ConfluencePage
from collections import Counter
from difflib import SequenceMatcher
import pickle
import pandas as pd
from pathlib import Path
from pprint import pprint
import re


@timeit
def main_loop():
    confluence_reader = ConfluenceReader(page_limit=global_config.get_config("page_limit"))

    page_data = {}
    pages = []

    spaces = [global_config.get_config("confluence_space_name")]

    # Für jeden Raum alle Seiten abrufen
    for space in spaces:
        pages = confluence_reader.fetch_all_pages_in_space(
            space_key=space,
            page_limit=global_config.get_config("page_limit", default_value=99999))
        if not pages:
            break
        # It looks like the pages include duplicates
        pages = _check_for_duplicate_pages(pages)

        logger.info(f"I found {len(pages)} pages in space {space}.")
        for page_index, page in enumerate(pages):
            page_id = page['content']['id']
            logger.info(f"Getting details for page {page_id}. Page {page_index} of {len(pages)}.")
            page_content = confluence_reader.get_page_by_id(
                page_id,
                expand='body.storage,version,history,metadata.labels,'
                       'metadata.incominglinks,ancestors,children.page,attachments')
            if not page_content:
                logger.error(f"Page {page_id} could not be fetched despite retry. Skipped.")
                continue

            # Neues: Watcher-Infos abrufen
            watchers = confluence_reader.get_page_watchers(page_id)

            page_data[page_id] = ConfluencePage(
                space=space,
                page_content=page_content)
            page_data[page_id].set_watchers(watchers)

            # Check, if we need to read more children pages
            while page_data[page_id].get_next_children_url():
                logger.info(f"Fetching next children for page {page_id}.")
                url = confluence_reader.url + page_data[page_id].get_next_children_url()
                children = confluence_reader.get_any_confluence_url_via_requests(url)
                page_data[page_id].set_next_children(children)
            page_data[page_id].set_attachments(confluence_reader.get_attachments_by_page_id(page_id))
            page_data[page_id].set_statistics(confluence_reader.get_page_statistics(page_id))


    # Before we can set the incoming links we need to resolve all internal links (those that don't have page_id yet
    # either need to be ignored (because they are external links or at least not in this space or web-links).
    resolve_incoming_links(page_data)

    # Change incoming links now that we have all page_ids resolved
    set_incoming_links(page_data)

    # Remove common text from pages (this is for pages that come from the same template. they share the same label
    # and we don't want the template content to be part of the training data)
    remove_common_text(page_data)

    # Calculate the page statistics based on the max values of all pages
    calculate_page_statistics_based_on_page_statistics(page_data)

    # Dictionary in PICKL-Datei speichern
    file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')). \
        joinpath(f"confluence_pages_{spaces[0]}_"
                 f"{Util.get_current_date_formatted_for_filename()}.pkl")
    with open(file_name, 'wb') as file:
        pickle.dump(page_data, file)
    logger.info(f"{len(pages)} Seiten wurden erfolgreich abgerufen und in {file_name} gespeichert.")

    save_as_xls(page_data, spaces[0])


def _check_for_duplicate_pages(pages):
    len_pages_before_duplicates = len(pages)
    pages = list({page['content']['id']: page for page in pages}.values())
    if len(pages) != len_pages_before_duplicates:
        logger.critical(f"Found duplicates in pages. {len_pages_before_duplicates} pages before, "
                        f"{len(pages)} after removing duplicates. That's not good!")
    return pages


def __make_proper_link_title(link_title: str) -> str:
    """
    Make the link_title searchable by removing special characters and converting to lower case
    :param link_title:
    :return:
    """
    link_title = link_title.split("#")[0]  # Remove anchors in the link.
    link_title = link_title.split("?")[0]  # Remove parameters in the link.
    link_title = unescape(link_title)
    return link_title


def resolve_incoming_links(page_data: dict):
    """
    internal links have the page_title as the link. We need to resolve this to the page_id.
    We find candidates to resolve in page.confluence_links.
    If we find non-numeric value in there we can search in all page_data for this title.
    If found: replace the line in page.confluence_links with the page_id.
    If not found: remove the line from page.confluence_links and make a warning in the log
    :param page_data:
    :return:
    """
    for page in page_data.values():
        if not page.outgoing_confluence_links:
            continue
        index_to_pop = []
        for link_index, link in enumerate(page.outgoing_confluence_links):
            if isinstance(link, int):
                # This is already a page_id but should be converted to string:
                page.outgoing_confluence_links[link_index] = str(link)
                continue
            if isinstance(link, dict):
                # The dictionary consists of content_title and space.
                # if space = empty it is our current space. We can search for the title in our page_data
                # otherwise we can't find this page and will ignore it
                if not link.get('space', "no space in dict"):
                    link_title = __make_proper_link_title(link['content_title'])

                    found_page = [page_id for page_id, page in page_data.items() if page.title == link_title]
                    if found_page:
                        page.outgoing_confluence_links[link_index] = found_page[0]
                    else:
                        logger.info(f"Weird. Could not resolve internal link {link['content_title']} in page {page.id}."
                                    f" Most probably broken link in the page. Removing it.")
                        page.quality_assessment_details.append(f"Broken Link {link}")
                        index_to_pop.append(link_index)
                    continue
                elif link.get("space") and \
                        link.get("space", "").upper() != global_config.get_config("confluence_space_name").upper():
                    # Other space - we can't resolve this link. We need to remove it.
                    logger.debug(f"Could not resolve remote link {link['content_title']} in page {page.id}. "
                                 f"It has space: '{link['space']}'. Removing it.")
                    index_to_pop.append(link_index)
                    continue
                elif link.get("space") and \
                        link.get("space", "").upper() == global_config.get_config("confluence_space_name").upper():
                    # This is a link in our space.
                    link_title = __make_proper_link_title(link['content_title'])
                    # Confluence-Specific URL-Encoding replacement. In the URL they encode space as "+". In one
                    # case we see "TEST + TEST" as "TEST+++TEST". We need to replace this with "TEST + TEST
                    found_page = [page_id for page_id, page in page_data.items() if page.title == link_title]
                    if not found_page:
                        # Try with this weird "+"-replacement that sometimes works.
                        link_title = link_title.replace("+", " ")
                        found_page = [page_id for page_id, page in page_data.items() if page.title == link_title]
                    if found_page:
                        page.outgoing_confluence_links[link_index] = found_page[0]
                    else:
                        logger.info(f"Weird. Link in our space with space-name but can't be resolved to page"
                                    f"{link} in page {page.id}. Removing it.")
                        index_to_pop.append(link_index)
                elif link.get('type') == "a":
                    # <a href-Linktype. Most probably we won't find anything from that.
                    if "mailto:" in link['href']:
                        # This is an email link. We can't resolve this. Remove it.
                        index_to_pop.append(link_index)
                        continue
                    if "?pageId=" in link['href']:
                        page_id = link['href'].split("?pageId=")[1]
                        page_id = page_id.split("&")[0]
                        page_id = page_id.split("#")[0]
                        page.outgoing_confluence_links[link_index] = page_id
                        continue
                    elif "confluence" in link['href'] and "/display/" in link['href']:
                        # https://confluence.example.com/display/SPACEKEY/PAGETITLE
                        # We need to see if SPACEKEY is us. If so, we can extract pagetitle to search within our pages
                        # If not, we can't resolve this link.
                        space_key = link['href'].split("/display/")[1].split("/")[0]
                        if space_key.upper() == global_config.get_config("confluence_space_name").upper():
                            page_title = __make_proper_link_title(link['href'].split("/display/")[1].split("/")[1])
                            found_page = [page_id for page_id, page in page_data.items() if page.title == page_title]
                            if found_page:
                                page.outgoing_confluence_links[link_index] = found_page[0]
                            else:
                                logger.info(f"Could not resolve internal link {page_title} in page {page.id}. "
                                            f"Removing it.")
                                index_to_pop.append(link_index)
                            continue
                        else:
                            logger.debug(f"Could not resolve internal link {link['href']} in page {page.id}. as from "
                                         f"another space. Removing it.")
                            index_to_pop.append(link_index)
                        continue
                    elif "/x/" in link['href']:
                        # There's a good chance this is a shortlink. We try to find it in ConfluencePage.tinyui
                        # (including the leading /x/
                        short_link = '/x/' + link['href'].split("/x/")[1]
                        found_page = [page_id for page_id, page in page_data.items() if page.tinyui == short_link]
                        # If this bothers us in the future we could also query Confluence
                        # using /rest/api/shortcuts/<tinyui> to get page_id and space from the shortlink
                        if found_page:
                            page.outgoing_confluence_links[link_index] = found_page[0]
                        else:
                            logger.info(f"Could not resolve short link {link['href']} in page {page.id}. "
                                        f"Removing it.")
                            index_to_pop.append(link_index)
                        continue
                    elif "jira" in link['href']:
                        # This is a Jira-Link. We can't resolve this. Remove it.
                        logger.debug(f"JIRA-Link ignored {link['href']} in page {page.id}.")
                        index_to_pop.append(link_index)
                        continue

                    index_to_pop.append(link_index)
                    logger.info(f"{link['href']} could not be resolved to internal page of this space. Ignoring it.")
                    continue
                else:
                    logger.info(f"No Idea what link {link} this is. Ignoring it. Page {page.id}")
                    index_to_pop.append(link_index)
                    continue
            else:
                logger.warning(f"Received unknown type of link: {link}. Ignoring it.")

        # Remove all links that we couldn't resolve. Start from the highest index and work our way to the top.
        for index in index_to_pop[::-1]:
            page.outgoing_confluence_links.pop(index)


def remove_common_text(page_data: dict):
    """
    When pages share the same label there's a good chance, that the page was created using a template. This template
    content is irrelevant for ML-Classification. We need to identify those common parts of the pages and remove them.
    :param page_data:
    :return:
    """

    from sklearn.feature_extraction.text import CountVectorizer

    def preprocess_text(text):
        # Convert to lower case
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[\W_]+', ' ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_ngrams(text, n=3):
        cleaned_text = preprocess_text(text)
        # Generate n-grams from the text
        vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern=r'\b\w+\b', min_df=1)
        analyzer = vectorizer.build_analyzer()
        return analyzer(cleaned_text)

    def find_common_segments(pages, n=3, threshold=0.4):
        """
        Identify common text segments in a list of pages using n-grams
        if more than threshold percent of the pages contain this common text identify it as common_segment.
        Those will be removed during the next step of the cleaning process
        :param pages: List of pages to analyze
        :param n: Number of n-grams to split into. Value 2 provided no useful results. 3 seems to be as good as it gets.
        :param threshold: float percentage of pages that must contain the n-gram to be considered common
        :return: List of common segments
        """
        # Split each page content into n-grams
        all_ngrams = []
        for page in pages:
            ngrams = generate_ngrams(page.plain_text_content.lower(), n)
            all_ngrams.extend(ngrams)

        # Count occurrences of each n-gram
        ngram_counter = Counter(all_ngrams)

        # Calculate the minimum number of pages an n-gram must appear in to be considered common
        min_occurrences = int(len(pages) * threshold) + 1

        # Identify common n-grams (appear in more than 50% of pages)
        common_segments = [ngram for ngram, count in ngram_counter.items() if count >= min_occurrences]

        return common_segments

    def remove_common_segments_from_pages(pages, common_segments):
        cleaned_pages = []

        for page in pages:
            text_content = preprocess_text(page.plain_text_content.lower())
            for segment in common_segments:
                text_content = text_content.replace(segment, '').strip()
            cleaned_page = page.copy()
            cleaned_page.plain_text_content = preprocess_text(text_content)
            cleaned_pages.append(cleaned_page)

        return cleaned_pages

    labels_to_remove_common_text = global_config.get_config("labels_to_remove_common_text", default_value=[])

    if not labels_to_remove_common_text:
        logger.info("No labels to remove common text from. Config 'labels_to_remove_common_text' is empty.")
        return

    for label in labels_to_remove_common_text:
        # Find all pages, that have this label:
        pages_with_label = [page for page in page_data.values() if label in page.labels.upper()]
        if not pages_with_label:
            logger.info(f"No pages with label {label} found. Skipping text process for this label.")
            continue
        logger.info(f"For label {label} found {len(pages_with_label)} pages. Start identifying common text to remove")
        # run Analysis on those pages
        common_texts = find_common_segments(pages_with_label)
        logger.info(f"Found {len(common_texts)} common texts in pages with label {label}. {pprint(common_texts)}")
        cleaned_pages = remove_common_segments_from_pages(pages_with_label, common_texts)

        # Update the page_data with the cleaned pages
        for page in cleaned_pages:
            page_data[page.id] = page
        logger.info(f"Removed common text from {len(cleaned_pages)} pages with label {label}.")


def set_incoming_links(page_data: dict):
    """
    In each entry in page_data we may have attribute confluence_links. For each entry in confluence_links we need to
    update the destination page's incoming_links attribute.
    :param page_data:
    :return:
    """
    for page in page_data.values():
        if not page.outgoing_confluence_links:
            continue
        for link in page.outgoing_confluence_links:
            if isinstance(link, str):
                link = link.strip()
            elif isinstance(link, int):
                pass
            else:
                x = 123
            if link in page_data:
                page_data[link].set_incoming_link(page.id)
            else:
                logger.debug(f"Page {link} not found in page_data. Incoming link not set.")


def calculate_page_statistics_based_on_page_statistics(page_data: dict):
    """
    We will find out the max number of viewers of a page and the max number of views of a page.
    then we'll provide this info the the ConfluencePage-Instances to calculate their own statistics
    :param page_data:
    :return:
    """
    max_views = 0
    max_viewers = 0
    for page in page_data.values():
        if page.overall_page_views > max_views:
            max_views = page.overall_page_views
        if page.overall_page_viewers > max_viewers:
            max_viewers = page.overall_page_viewers

    # Now that we have each max-value we can set it in the ConfluencePage-Instances
    for page in page_data.values():
        page.set_space_statistics(max_views=max_views, max_viewers=max_viewers)


def save_as_xls(list_of_dataclasses: dict, space_name: str):
    """
    Save the list of ConfluencePage-Dataclasses as an XLSX-File
    :param list_of_dataclasses:
    :param space_name: Name-addition for the filename generation
    :return:
    """
    df = pd.DataFrame([vars(page) for page in list_of_dataclasses.values()])
    # Delete rows content, ancestors
    df = df.drop(columns=["content", "ancestors", "plain_text_content"])
    # the url is only in parts there. We need to concatenate with the base-url
    df['url'] = global_config.get_config('confluence_base_url') + df['url']
    # the column "outgoing_confluence_links" and "incoming_links is a list. We need to convert it to a string
    df['outgoing_confluence_links'] = df['outgoing_confluence_links'].apply(lambda x: ", ".join(x))
    df['incoming_links'] = df['incoming_links'].apply(lambda x: ", ".join(x))
    # the column "quality_assessment_details" is a list. We want it joined as string with \n between entries
    df['quality_assessment_details'] = df['quality_assessment_details'].apply(lambda x: "\n".join(x))
    # Watchers is a list of strings. We need to convert it to a string
    df['watchers'] = df['watchers'].apply(lambda x: ", ".join(x))
    # Column next_children_url is not needed - remove
    df = df.drop(columns=["next_children_url"])
    file_name = Path.cwd().joinpath(global_config.get_config('path_for_page_storage_data')). \
        joinpath(f"confluence_pages_{space_name}_{Util.get_current_date_formatted_for_filename()}.xlsx")
    Util.write_dataframe_to_excel(dfs=df, filename=str(file_name), sheetname="Metadata")


if __name__ == '__main__':
    Util.load_env_file()
    logger = Util.get_logger()
    main_loop()
