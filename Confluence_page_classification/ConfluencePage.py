from dataclasses import dataclass, field
from Confluence_page_classification.util import Util, logger, global_config
from Confluence_page_classification.UtilHtmlHandling import UtilHtmlHandling
from datetime import datetime
import re
import copy

# Import some global configuration values
created_by_important_names = global_config.get_config("created_by_important_names", default_value=[])
labels_to_keep = global_config.get_config("labels_to_keep", default_value=[])
labels_to_remove = global_config.get_config("labels_to_remove", default_value=[])


@dataclass
class ConfluencePage:
    # General data:
    id: int
    title: str
    url: str
    space: str

    content: str
    labels: str

    parent: str = None
    children: list[str] = field(default_factory=list)
    next_children_url: str = None
    attachments: list[str] = field(default_factory=list)
    version_count: int = None
    incoming_links: list[str] = field(default_factory=list)
    tinyui: str = None   # This is the tinyui-URL of the page. It's sometimes use for links...
    ancestors: dict = field(default_factory=dict)
    ancectors_page_ids: list[str] = field(default_factory=list)

    # datetime fields
    created: datetime.date = None
    last_changed: datetime.date = None
    age_in_days: int = None
    last_updated_in_days: int = None
    created_by: str = None

    # transient fields
    pure_length: int = None
    incoming_links_count: int = None
    ancestors_count: int = None
    children_count: int = None
    attachment_count: int = None
    quality_points_total: int = None
    quality_points_plus: int = None
    quality_points_minus: int = None
    quality_assessment_details: list[str] = field(default_factory=list)
    statistics_data: dict = None
    overall_page_views: int = None
    overall_page_viewers: int = None

    # HTML-Analysen:
    table_count: int = None
    rows_count: int = None
    columns_count: int = None
    jira_links_count: int = None
    h1_count: int = None
    h2_count: int = None
    h3_count: int = None
    h4_count: int = None
    outgoing_confluence_links: list[any] = field(default_factory=list)
    outgoing_confluence_links_count: int = None
    image_in_page_count: int = None
    # Contains the text without HTML-Tags
    plain_text_content: str = None
    # Contains the length of the text within tables
    text_in_tables_len: int = None
    dates_in_text: int = None
    tasks_open: int = None
    tasks_closed: int = None
    page_properties: int = None
    page_properties_report: int = None
    plantuml_macros_count: int = None
    plantuml_macros_textlen: int = None
    status_macros_count: int = None
    taskreport_macros_count: int = None
    user_mentions: int = None
    text_in_tables_percentage: float = None
    watchers: list[str] = field(default_factory=list)
    watchers_count: int = None

    def __str__(self):
        return f"{self.id}: {self.title} ({self.url})"

    def __init__(self, space, page_content):
        self.space = space
        self.id = page_content['id']
        self.title = page_content['title']
        self.url = page_content['_links']['webui']
        self.content = page_content['body']['storage']['value']
        self.plain_text_content = UtilHtmlHandling.extract_plain_text_from_html(self.content)
        self.labels = self.__format_labels(page_content['metadata']['labels']['results'])
        self.children = ([child['title'] for child in page_content['children']['page']['results']]
                         if 'results' in page_content['children']['page'] else [])
        self.next_children_url = page_content['children']['page']['_links'].get('next')
        self.children_count = len(self.children)
        self.pure_length = len(self.plain_text_content)
        self.lastChanged = Util.get_date_from_confluence_api_date(page_content['version']['when'])
        self.created = Util.get_date_from_confluence_api_date(page_content['history']['createdDate'])
        self.created_by = page_content['history']['createdBy']['displayName']
        self.version_count = page_content['version']['number']
        self.incoming_links = page_content.get('metadata', {}).get('incomingLinks', {}).get('results') or []
        self.incoming_links_count = len(self.incoming_links)
        self.ancestors = page_content['ancestors']
        self.ancestors_count = len(self.ancestors)
        self.parent = self.create_parent_string()
        self.tinyui = page_content['_links']['tinyui']
        self.age_in_days = (datetime.date(datetime.now()) - self.created).days
        self.last_updated_in_days = (datetime.date(datetime.now()) - self.lastChanged).days
        self.overall_page_viewers = 0

        self.__html_analyses()

        analysis = ConfluencePageQualityAnalysis(self)
        self.quality_points_total, self.quality_points_plus, self.quality_points_minus = analysis.get_points()
        self.quality_assessment_details = analysis.get_assessment_details()

    def create_parent_string(self):
        """
        Ancecstors contains the whole tree. Last item in the list is the direct ancestor,
        second ancestor the great-parent, and so on.
        :return:
        """
        if not self.ancestors:
            return ""
        # Prepare the parent string
        try:
            parent = " > ".join([ancestor['title'] for ancestor in self.ancestors])
        except KeyError:
            return ""
        # Also fill in the ancestor_page_ids for later when we need to set plus-values or minus-values
        # for pages that should be kept or removed as children under a specific page_id.
        self.ancectors_page_ids = [ancestor['id'] for ancestor in self.ancestors]
        return parent

    def __html_analyses(self):
        """
        Analysiert den HTML-Inhalt der Seite
        :return:
        """
        self.table_count = UtilHtmlHandling.count_html_elements(self.content, 'table')
        self.rows_count = UtilHtmlHandling.count_html_elements(self.content, 'tr')
        self.columns_count = UtilHtmlHandling.count_html_elements(self.content, 'td')

        self.__count_embedded_images()
        self.__count_text_within_tables()
        self.__count_date_macros()
        self.__count_tasks()
        self.__count_page_properties_macros()
        self.__count_page_properties_report_macros()
        self.__count_user_mentions()
        self.__count_plantuml_macros()
        self.__list_links_to_confluence_pages()
        self.__count_header_tags()
        self.__count_status_macros()
        self.__count_taskreport_macros()
        self.__count_jira_links()

        self.text_in_tables_percentage = self.text_in_tables_len / self.pure_length if self.pure_length > 0 else 0

    def __count_header_tags(self):
        """
        Count the number of H1, H2, H3 and H4 tags in the content
        :return:
        """
        self.h1_count = UtilHtmlHandling.count_html_elements(self.content, 'h1')
        self.h2_count = UtilHtmlHandling.count_html_elements(self.content, 'h2')
        self.h3_count = UtilHtmlHandling.count_html_elements(self.content, 'h3')
        self.h4_count = UtilHtmlHandling.count_html_elements(self.content, 'h4')

    def __count_jira_links(self):
        self.jira_links_count = UtilHtmlHandling.count_jira_links(self.content)

    def __count_status_macros(self):
        """
        Count the number of status macros in the content
        :return:
        """
        status_macro_pattern = re.compile(r'<ac:structured-macro[^>]*ac:name="status"[^>]*>', re.IGNORECASE)

        status_macros = status_macro_pattern.findall(self.content)

        self.status_macros_count = len(status_macros)

    def __count_taskreport_macros(self):
        """
        Count the number of taskreport macros in the content
        :return:
        """
        taskreport_macro_pattern = re.compile(r'<ac:structured-macro[^>]*ac:name="tasks-report-macro"[^>]*>',
                                              re.IGNORECASE)

        taskreport_macros = taskreport_macro_pattern.findall(self.content)

        self.taskreport_macros_count = len(taskreport_macros)

    def __count_embedded_images(self):
        """
        Embedded image tags are like <ac:image><ri:attachment ri:filename="image.png" /></ac:image>
        We want to know how many of those images are embedded in the page
        :return:
        """
        # RE to detect <ac:image>-Tags
        image_tag_pattern = re.compile(r'<ac:image[^>]*>.*?</ac:image>', re.DOTALL)
        # RE to detect <ri:attachment>-Tags within <ac:image>
        attachment_tag_pattern = re.compile(r'<ri:attachment[^>]*>', re.DOTALL)

        # Find all <ac:image>-Tags
        image_tags = image_tag_pattern.findall(self.content)

        # Count <ri:attachment>-Tags within <ac:image>-Tags
        self.image_in_page_count = sum(len(attachment_tag_pattern.findall(image_tag)) for image_tag in image_tags)

    def __count_text_within_tables(self):
        # Regex zur Identifizierung von <table>-Tags und ihrem Inhalt
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)

        # Finde alle <table>-Tags im Inhalt
        tables = table_pattern.findall(self.content)

        self.text_in_tables_len = 0

        # Prozessieren jedes gefundenen <table>-Tags
        for table in tables:
            plain_text = UtilHtmlHandling.extract_plain_text_from_html(table)
            self.text_in_tables_len += len(plain_text)

    def __count_date_macros(self):
        # Regex zur Identifizierung von Confluence Date Macros
        date_macro_pattern = re.compile(r'<time[^>]*\/>', re.IGNORECASE)

        # Finde alle Vorkommen der Date Macros im Inhalt
        date_macros = date_macro_pattern.findall(self.content)

        self.dates_in_text = len(date_macros)

    def __count_tasks(self):
        # Regex to identify Confluence task macros
        open_task_pattern = re.compile(r'<ac:task>(.*?)<ac:task-status>incomplete</ac:task-status>(.*?)</ac:task>',
                                       re.IGNORECASE | re.DOTALL)
        closed_task_pattern = re.compile(r'<ac:task>(.*?)<ac:task-status>complete</ac:task-status>(.*?)</ac:task>',
                                         re.IGNORECASE | re.DOTALL)

        # Find all occurrences of open tasks
        open_tasks = open_task_pattern.findall(self.content)
        # Find all occurrences of closed tasks
        closed_tasks = closed_task_pattern.findall(self.content)

        self.tasks_open = len(open_tasks)
        self.tasks_closed = len(closed_tasks)

    def __count_plantuml_macros(self):
        # Regex to identify PlantUML macros
        plantuml_pattern = re.compile(
            r'(<ac:structured-macro[^>]*ac:name="plantuml"[^>]*>.*?</ac:structured-macro>)', re.IGNORECASE | re.DOTALL)

        # Find all occurrences of PlantUML macros
        plantuml_macros = plantuml_pattern.findall(self.content)

        self.plantuml_macros_count = len(plantuml_macros)

        # Calculate the number of characters between each PlantUML start and end macro
        self.plantuml_macros_textlen = sum([len(macro) for macro in plantuml_macros])
        self.pure_length -= self.plantuml_macros_textlen

    def __count_page_properties_macros(self):
        # Regex to identify page_properties macros
        page_properties_pattern = re.compile(r'<ac:structured-macro[^>]*ac:name="details"[^>]*>', re.IGNORECASE)

        # Find all occurrences of page_properties macros
        page_properties_macros = page_properties_pattern.findall(self.content)

        self.page_properties = len(page_properties_macros)

    def __count_page_properties_report_macros(self):
        # Regex to identify page_properties_report macros
        page_properties_report_pattern = re.compile(r'<ac:structured-macro[^>]*ac:name="detailssummary"[^>]*>',
                                                    re.IGNORECASE)

        # Find all occurrences of page_properties_report macros
        page_properties_report_macros = page_properties_report_pattern.findall(self.content)

        self.page_properties_report = len(page_properties_report_macros)

    def __count_user_mentions(self):
        # Regex to identify @-mentions (ri:user tags)
        user_mention_pattern = re.compile(r'<ri:user[^>]*>', re.IGNORECASE)

        # Find all occurrences of @-mentions
        user_mentions = user_mention_pattern.findall(self.content)

        self.user_mentions = len(user_mentions)

    def __list_links_to_confluence_pages(self):
        links = []
        text = self.content

        # Define regex patterns
        ac_link_pattern = re.compile(r'<ac:link>(.*?)</ac:link>')
        ri_page_pattern = re.compile(r'<ri:page(.*?)/>')
        href_pattern = re.compile(r'<a.*?href="(.*?)".*?>')

        # Extract <ac:link> elements
        for ac_link_match in ac_link_pattern.finditer(text):
            ac_link_content = ac_link_match.group(1)
            ri_page_match = ri_page_pattern.search(ac_link_content)
            if ri_page_match:
                attributes = ri_page_match.group(1)
                space_key_match = re.search(r'ri:space-key="(.*?)"', attributes)
                content_title_match = re.search(r'ri:content-title="(.*?)"', attributes)
                anchor_match = re.search(r'ac:anchor="(.*?)"', attributes)
                links.append({
                    'type': 'ac:link',
                    'space': space_key_match.group(1) if space_key_match else None,
                    'content_title': content_title_match.group(1) if content_title_match else None,
                    'anchor': anchor_match.group(1) if anchor_match else None
                })
            elif len(ac_link_content) == len("4028986c781e3120017a096888b70083"):
                # Damaged ri:userkey - ignore. This is a special case
                continue
            elif "<![CDATA[" in ac_link_content:
                # User-Mention- here not relevant
                continue
            elif "ri:space" in ac_link_content:
                # Link to a different space - not a single page. Irrelevant
                continue
            elif "ri:userkey" in ac_link_content:
                # Userkeys are not interesting for the links.
                continue
            elif "ri:attachment" in ac_link_content:
                # Attachment-Links are not interesting for the links.
                continue
            else:
                logger.warning(f"Unknown ac:link content: {ac_link_content}. Ignoring this link")
                x = 123   # What is this?
                continue

        # Extract <a> elements
        for href_match in href_pattern.finditer(text):
            href = href_match.group(1)
            page_id_match = re.search(r'pageId=(\d+)', href)
            if page_id_match:
                page_id = page_id_match.group(1)
                links.append({
                    'type': 'a',
                    'href': href,
                    'page_id': page_id
                })
            else:
                links.append({
                    'type': 'a',
                    'href': href
                })

        # Links might include dupliates. Let's have only unique links
        links = Util.make_unique_list_entries(links)

        self.outgoing_confluence_links = links
        self.outgoing_confluence_links_count = len(links)

    def get_next_children_url(self):
        """
        if we haven't read all children yet we want to get the next page of children. We'll return an URL to send
        us the next children.
        :return:
        """
        if self.next_children_url:
            return self.next_children_url
        return None

    def set_next_children(self, children):
        """
        After we answered with a URL in get_next_children_url the URL was queried. We receive the result here
        :param children:
        :return:
        """
        self.children.extend([child['title'] for child in children['results']])
        self.children_count = len(self.children)
        self.next_children_url = children['_links'].get('next')

    def set_attachments(self, attachments):
        self.attachments = [attachment['title'] for attachment in attachments['results']]
        self.attachment_count = len(self.attachments)

    def set_watchers(self, watchers: dict):
        self.watchers = []
        for k,v in watchers.items():
            for watcher_name in v:
                self.watchers.append(watcher_name)
        # Remove duplicates
        self.watchers = list(set(self.watchers))
        self.watchers_count = len(self.watchers)

    def set_incoming_link(self, incoming_link):
        """
        Externally driven method to set incoming links
        :param incoming_link:
        :return:
        """
        self.incoming_links.append(str(incoming_link))
        self.incoming_links_count = len(self.incoming_links)

    def set_statistics(self, statistics_data: dict):
        """
        Statistics-data comes as dict with keys views and viewers
        We will store this data in the page object. Later in the analysis we will use this data to assign points
        :param statistics_data:
        :return:
        """
        self.statistics_data = statistics_data["statistics"]
        if self.statistics_data.get("viewsByDate"):
            logger.warning(f"Statistics data for page {self.title} contains viewsByDate. You've activated "
                           f"read_monthly_page_access_statistics in config. There is no logic implemented to deal"
                           f"with those numbers.")

        # Count statistics data->viewers->views into self.overall_page_views
        self.overall_page_views = 0
        for viewer in statistics_data["statistics"]["viewers"]:
            self.overall_page_views += viewer["views"]
            self.overall_page_viewers += 1

    def set_space_statistics(self, max_viewers: int, max_views: int):
        """
        We receive the maximum number of viewers and views for the space. Combined with our own statistics we can
        determine how important we are. We will assign points based on this information.
        :param max_viewers:
        :param max_views:
        :return:
        """
        if not self.overall_page_viewers:
            self.quality_points_minus -= 1000
            self.quality_assessment_details.append("Page has never been viewed in this space")
            # recalculate the overall quality points.
            self.quality_points_total = self.quality_points_plus + self.quality_points_minus
            return

        viewers_ratio = self.overall_page_viewers / max_viewers
        views_ratio = self.overall_page_views / max_views

        # FIXME: These rates are most probably wrong. Both the high and the low values.
        # 0.5 as upper value was way to high
        # Let's try like this:
        # for every percent above 10% viewers we get 10 points
        # same for views
        if viewers_ratio > 0.1:
            self.quality_points_plus += 10 * (viewers_ratio * 100)
            self.quality_assessment_details.append(f"Page has {viewers_ratio*100}% of maximum viewers")

        if views_ratio > 0.1:
            self.quality_points_plus += 10 * (views_ratio * 100)
            self.quality_assessment_details.append(f"Page has {views_ratio*100}% of maximum views")

        if views_ratio < 0.01:
            self.quality_points_minus -= 10
            self.quality_assessment_details.append("Page has less than 1 percent of the maximum views")

        if self.age_in_days < 30:
            # Page is pretty new - we can't expect to see much traffic here.
            pass
        else:
            if viewers_ratio < 0.05:
                self.quality_points_minus -= 10
                self.quality_assessment_details.append("Page has less than 5 percent of the maximum viewers")

            if self.overall_page_viewers <= 2:
                self.quality_points_minus -= 20
                self.quality_assessment_details.append("Page has 2 viewers or less")

        # recalculate the overall quality points.
        self.quality_points_total = self.quality_points_plus + self.quality_points_minus

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def __format_labels(labels_list_of_dicts: list[dict]) -> str:
        """
        Konvertiert eine Liste von Label-Dictionaries in eine komma-separierte Zeichenfolge
        :param labels_list_of_dicts: Liste von Label-Dictionaries
        :return: Komma-separierte Zeichenfolge
        """
        return ', '.join([label['name'] for label in labels_list_of_dicts])


class ConfluencePageQualityAnalysis:
    """
    Analysis various aspects of the page and created a numeric quality indicator. This is an example implementation only
    YMMV.
    """

    def __init__(self, page: ConfluencePage):
        self.page = page
        self.quality_points_minus = 0
        self.quality_points_plus = 0
        self.assessment_details = []
        self.__analyze()

    def get_points(self):
        quality_points_total = self.quality_points_plus + self.quality_points_minus
        return quality_points_total, self.quality_points_plus, self.quality_points_minus

    def get_assessment_details(self):
        return self.assessment_details

    def __analyze(self):
        self.__check_created_by_important_names()
        self.__check_labels_to_keep()
        self.__check_labels_to_remove()
        self.__check_h1_h2_h3__h4_tags()
        self.__check_jira_tasks()
        self.__check_length_of_tables()
        self.__check_plantuml_macros()
        self.__check_date_in_title()
        self.__check_page_title()
        self.__check_page_versions()
        self.__check_length()
        self.__check_status_macros()
        self.__check_taskreport_macros()

    def __check_created_by_important_names(self):
        if self.page.created_by in created_by_important_names:
            self.quality_points_plus += 20
            self.assessment_details.append(f"Important creator {self.page.created_by}")
            logger.info(f"Important creator {self.page.created_by} found on page {self.page.title}")

    def __check_labels_to_keep(self):
        if not self.page.labels:
            return

        message = ""
        for label in labels_to_keep:
            if label in self.page.labels.upper():
                self.quality_points_plus += 1000
                self.assessment_details.append(f"Important label {label}")
                message += f"Important label {label}"

        if message:
            logger.info(message)

    def __check_status_macros(self):
        """
        If there are too many status macros in the page (relative to page content) it's most probably
        not good content
        :return:
        """
        if not self.page.pure_length:
            return

        # FIXME: Value 0.1 is not verified!
        if self.page.status_macros_count / self.page.pure_length > 0.1:
            self.quality_points_minus -= 10
            self.assessment_details.append(f"Too many status macros")
            logger.info(f"Too many status macros on page {self.page.title}")

    def __check_taskreport_macros(self):
        """
        if there is one or more taskreport macros on the page it's probably not good content
        :return:
        """
        if self.page.taskreport_macros_count > 0:
            self.quality_points_minus -= 50
            self.assessment_details.append(f"Taskreport macros found")
            logger.info(f"Taskreport macros found on page {self.page.title}")

    def __check_labels_to_remove(self):
        if not self.page.labels:
            return

        message = ""
        for label in labels_to_remove:
            if label in self.page.labels.upper():
                self.quality_points_minus -= 1000
                self.assessment_details.append(f"Label {label} - page not relevant")
                message += f"Label {label} - page not relevant"

    def __check_date_in_title(self):
        """
        If the title starts with a date yyyy-mm-dd it's usually not golden content.
        :return:
        """
        if re.match(r'^\d{4}-\d{2}-\d{2}', self.page.title):
            self.quality_points_minus -= 10
            self.assessment_details.append(f"Date in title {self.page.title}")
            logger.info(f"Date in title {self.page.title}")

    def __check_h1_h2_h3__h4_tags(self):
        """
        a good page has a structure. The longer the page the more important the structure
        We want to have a mininum of H1, H2, H3 and H4 tags per page length, for instance
        if the length is between 2000 and 4000 we should have at least 3 tags,
        if the length is between 4001 and 6000 we should have at least 4 tags, etc.
        :param self:
        :return:
        """
        textlen_without_tables = int(self.page.pure_length - self.page.text_in_tables_len)

        criteria = {"len_to": [800, 1000, 4000, 6000, 8000, 10000],
                    "min_tags": [0, 1, 3, 5, 6, 7]}

        for i, len_to in enumerate(criteria["len_to"]):
            if textlen_without_tables < len_to:
                min_tags = criteria["min_tags"][i]
                break
        else:
            min_tags = 7

        sum_header_counts = self.page.h1_count + self.page.h2_count + self.page.h3_count + self.page.h4_count
        if sum_header_counts < min_tags:
            self.quality_points_minus -= 10
            self.assessment_details.append(f"Too few header tags. Only {sum_header_counts} "
                                           f"for {textlen_without_tables} characters")
            logger.info(f"Too few header tags on page {self.page.title}")

    def __check_plantuml_macros(self):
        """
        PlantUML are usually good for documentation. Let's assume they are valuable
        :return:
        """
        if not self.page.plantuml_macros_count:
            return

        self.quality_points_plus += self.page.plantuml_macros_count * 10
        self.assessment_details.append(f"PlantUML macros found {self.page.plantuml_macros_count}")

    def __check_length(self):
        """
        if the length of the page is less than 500 characters and it doesn't have children
        then it's very likely not a good page
        :return:
        """
        if self.page.pure_length < 500 and self.page.children_count == 0:
            self.quality_points_minus -= 80
            logger.info(f"Page {self.page.title} not relevant")
            self.assessment_details.append(f"Page not relevant from __check_length")

    def __check_length_of_tables(self):
        """
        If the page has a lot of tables compared to the page length it might be just a table list and irrelevant
        :param self:
        :return:
        """
        if not self.page.pure_length:
            return

        if self.page.table_count > 0:
            if self.page.table_count / self.page.pure_length > 0.1:
                self.quality_points_minus -= 10
                logger.info(f"Too many tables on page {self.page.title}")
                self.assessment_details.append(f"Too many tables")
            if self.page.rows_count > 100:
                self.quality_points_minus -= 50
                logger.info(f"Too many rows {self.page.rows_count} in tables on page {self.page.title}")
                self.assessment_details.append(f"Too many rows in tables")

    def __check_page_title(self):
        """
        When there are certain words in the page title we might want to ignore the page
        :return:
        """
        words_to_ignore = global_config.get_config("page_title_ignore", default_value=[])
        for word in words_to_ignore:
            if word in self.page.title:
                self.quality_points_minus -= 1000
                logger.info(f"Page title contains {word} - page not relevant")
                self.assessment_details.append(f"Page title contains {word} - page not relevant")

    def __check_page_versions(self):
        """
        If a page has only very few versions, was not updated, has no tasks, no incoming links, no embedded images,
        no attachments and no children it's probably not relevant
        :return:
        """
        if self.page.version_count < 5 and self.page.last_updated_in_days > 365 and \
                self.page.tasks_open == 0 and self.page.tasks_closed == 0 and \
                self.page.incoming_links_count == 0 and self.page.image_in_page_count == 0 and \
                self.page.attachment_count == 0 and self.page.children_count == 0:
            self.quality_points_minus -= 1000
            logger.info(f"Page {self.page.title} not relevant")
            self.assessment_details.append(f"Page not relevant from __check_page_versions")

    def __check_jira_tasks(self):
        """
        If the page has a lot of jira_tasks compared to the page length it might be just a link list and irrelevant
        :param self:
        :return:
        """
        # FIXME: The numbers have not been verified! Update 30.5.: Not looking too bad. with 0.01 and 0.0015
        # We might actually have to lower the threholds
        if self.page.jira_links_count > 0:
            if self.page.jira_links_count / self.page.pure_length > 0.01:
                self.quality_points_minus -= 1000
                self.assessment_details.append(f"Too many JIRA links")
                logger.info(f"Too many JIRA links on page {self.page.title}. "
                            f"Relation: {self.page.jira_links_count / self.page.pure_length}")
            elif self.page.jira_links_count / self.page.pure_length > 0.0015:
                self.quality_points_minus -= 20
                self.assessment_details.append(f"Many JIRA links")
                logger.info(f"Many JIRA links on page {self.page.title}. Relation: "
                            f"{self.page.jira_links_count / self.page.pure_length}")
