import xml.etree.ElementTree as ET
from typing import Dict

def parse_xml_content(content: str) -> Dict[str, str]:
    root = ET.fromstring(content)
    return {
        "subject": root.findtext("subject", "XML Email"),
        "from": root.findtext("from", "unknown@example.com"),
        "to": root.findtext("to", "recipient@example.com"),
        "date": root.findtext("date", ""),
        "body": root.findtext("body", "No body found."),
    }