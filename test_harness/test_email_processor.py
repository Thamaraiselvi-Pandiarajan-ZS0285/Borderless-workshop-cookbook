from backend.app.core.email_processor import EmailProcessor

processor = EmailProcessor()
processor.process_emails(limit=5)
