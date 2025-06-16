from bs4 import BeautifulSoup
from typing import Dict

def parse_html_content(content: str) -> Dict[str, str]:
    soup = BeautifulSoup(content, "html.parser")
    return {
        "subject": "HTML Email",
        "from": "unknown@example.com",
        "to": "recipient@example.com",
        "date": "",
        "body": soup.get_text(separator="\n"),
    }