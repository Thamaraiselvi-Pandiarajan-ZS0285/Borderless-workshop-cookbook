from backend.app.core.email_listener import EmailListener
from backend.config.dev_config import USER_ID

listener = EmailListener(USER_ID)

listener.start()
