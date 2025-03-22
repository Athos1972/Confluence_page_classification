import re
from bs4 import BeautifulSoup


class UtilHtmlHandling:
    html_tags = {
        "ü": "&uuml;",
        "Ü": "&Uuml;",
        "Ä": "&Auml;",
        "ß": "&szlig;",
        "ä": "&auml;",
        "ö": "&ouml;",
        "Ö": "&Ouml;",
        "&": "&amp;",
        "<": "&lt;",
        '"': "&quot;",
        ">": "&gt;",
        "  ": "&nbsp;",
    }

    @staticmethod
    def count_html_elements(html_text: str, element: str) -> int:
        """
        Count the number of HTML elements in a given HTML text.
        :param html_text: The HTML text
        :param element: The HTML element to count
        :return: The number of elements
        """
        pattern = re.compile(rf'<{element}[^>]*>')
        return len(re.findall(pattern, html_text))

    @staticmethod
    def extract_plain_text_from_html(html_text: str) -> str:
        """
        Extract plain text from HTML text using BeautifulSoup
        :param html_text: HTML content as a string
        :return: Plain text extracted from the HTML
        """
        soup = BeautifulSoup(html_text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Extract plain text
        plain_text = soup.get_text(separator=" ", strip=True)

        # The text contains Newlines. Those new lines or "NBSP".
        # This should be replaced by ". " to make the text more readable
        plain_text = plain_text.replace("\n", ". ")
        plain_text = plain_text.replace("\xa0", ". ")
        # Remove extra whitespace and newlines
        plain_text = ' '.join(plain_text.split())

        return plain_text

    @staticmethod
    def html_operation(operation="escape", string="") -> str:
        """
        Escape/Unescape von HTML Entities

        :param operation: "escape" (ü->&uuml;) oder "unescape" (&uuml;->ü)
        :param string: irgendein STring
        :return: behandelter STring
        """
        # & (wie in Susi&Strolch) muss vorab übersetzt werden, weil er sonst bereits übersetzte &uuml auch nochmal
        # Escaped
        if operation == "escape":
            if "&" in string:
                string = string.replace("&", "&amp;")

            for (k, v) in UtilHtmlHandling.html_tags.items():
                if k == "&":
                    continue
                string = string.replace(k, v)

        if operation == "unescape":
            for (k, v) in UtilHtmlHandling.html_tags.items():
                string = string.replace(v, k)

        return string

    @staticmethod
    def count_jira_links(html_text: str) -> int:
        """
        extracts the JIRA-Links from the storage format of Confluence and counts them
        Each JIRA-Link is like this in storage format: <ac:structured-macro ac:name="jira">...</ac:structured-macro>
        :param html_text:
        :return:
        """
        jira_pattern = re.compile(r'<ac:structured-macro[^>]*ac:name="jira"[^>]*>.*?</ac:structured-macro>',
                                  re.IGNORECASE | re.DOTALL)
        jira_count = len(re.findall(jira_pattern, html_text))

        # Second way to link an issue:
        jira_pattern = re.compile(r'<ri:issue[^>]*ri:issue-key="[^"]+"[^>]*>', re.IGNORECASE)
        jira_count = jira_count + len(jira_pattern.findall(html_text))

        # Another pattern is:
        # https://*jira*/browse/ISSUE-1234
        jira_pattern = re.compile(r'https://[^/]*jira[^/]*/browse/[^"]+', re.IGNORECASE)
        jira_count = jira_count + len(jira_pattern.findall(html_text))

        return jira_count
