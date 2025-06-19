from backend.app.core.graph_client import GraphClient

class EmailListener:
    def __init__(self):
        self.client = GraphClient()

    def fetch_emails(self, top=10):
        return self.client.get_messages(top=top)
