import requests
from typing import Dict

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"

class EmailMover:
    def __init__(self, access_token: str, user_id: str):
        self.access_token = access_token
        self.user_id = user_id
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        self.folder_cache: Dict[str, str] = {}  # name -> id mapping

    def _get_folder_id(self, folder_name: str) -> str:
        """
        Retrieve the ID of a folder by display name.
        Caches results to avoid repeated API calls.
        """
        #if folder_name in self.folder_cache:
        #    return self.folder_cache[folder_name]

        url = f"{GRAPH_API_BASE}/users/{self.user_id}/mailFolders"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Failed to list mail folders: {response.text}")

        folders = response.json().get("value", [])
        for folder in folders:
            if folder["displayName"].lower() == folder_name.lower():
                #self.folder_cache[folder_name] = folder["id"]
                return folder["id"]

        # Folder not found — create it
        print(f"Folder '{folder_name}' not found. Creating it...")
        return self._create_folder(folder_name)
        #raise ValueError(f"Folder '{folder_name}' not found for user {self.user_id}.")

    def move_email_to_folder(self, message_id: str, destination_folder_name: str) -> Dict:
        """
        Move a specific email message to a given folder by name.
        """
        folder_id = self._get_folder_id(destination_folder_name)

        url = f"{GRAPH_API_BASE}/users/{self.user_id}/messages/{message_id}/move"
        body = {"destinationId": folder_id}
        response = requests.post(url, headers=self.headers, json=body)

        if response.status_code != 201:
            raise RuntimeError(f"Failed to move email: {response.text}")

        return response.json()

    def _create_folder(self, folder_name: str) -> str:
        """
        Create a new mail folder with the given name under root.
        """
        #url = f"{self.base_url}/mailFolders"
        url = f"{GRAPH_API_BASE}/users/{self.user_id}/mailFolders"
        data = {
            "displayName": folder_name
        }
        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code != 201:
            raise Exception(f"Failed to create folder '{folder_name}': {response.text}")

        return response.json()["id"]