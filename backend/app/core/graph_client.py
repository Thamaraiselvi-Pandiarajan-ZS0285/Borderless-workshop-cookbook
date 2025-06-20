import msal
import requests
from backend.config.dev_config import *

class GraphClient:
    def __init__(self):
        self.app = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_SECRET
        )
        self.token = self._get_token()

    def _get_token(self):
        result = self.app.acquire_token_for_client(scopes=SCOPE)

        if "access_token" in result:
            return result["access_token"]
        else:
            raise RuntimeError(f"Failed to acquire token: {result.get('error_description', result)}")

    def get_headers(self):
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def get_messages(self, folder=MAIL_FOLDER, top=10):
        url = f"{GRAPH_ENDPOINT}/users/{USER_ID}/mailFolders/{folder}/messages?$top={top}"
        res = requests.get(url, headers=self.get_headers())
        res.raise_for_status()
        return res.json()["value"]

    def move_message(self, message_id, destination_folder):
        url = f"{GRAPH_ENDPOINT}/users/{USER_ID}/messages/{message_id}/move"
        res = requests.post(url, headers=self.get_headers(), json={"destinationId": destination_folder})
        res.raise_for_status()
        return res.json()
