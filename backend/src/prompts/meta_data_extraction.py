# RFP Email Prompt
RFP_EXTRACTION_PROMPT = """
You are an AI assistant specialized in processing RFP (Request for Proposal) emails in the market research domain.

Your job is to extract relevant metadata **only** from the content of the email **without guessing**. These emails typically come from clients seeking bids or proposals for surveys or research work.

Extract the following details if present, and return in structured JSON:
```json
{
  "email_type": "RFP",
  "from": "<Extract sender name or email>",
  "to": "<Recipient(s)>",
  "date": "<Date and time>",
  "subject": "<Email subject>",
  "full_email_text": "<Complete email body with headers and footers>",
  "client_name": "<Mentioned client or issuing company>",
  "deadline": "<Submission or proposal deadline>",
  "estimated_cost": "<Budget or estimated cost>",
  "target_market": "<Demographics, region, or audience>",
  "survey_length": "<Duration or size of the survey>",
  "incidents": "<Target completes, sample size, or quotas>",
  "deliverables": "<What is expected to be delivered>",
  "next_steps": "<Instructions or follow-up actions>",
  "attachments": "<Mentioned attached documents>",
  "signatory": "<Name, title, company from signature>"
}
"""

# Bid Win Email Prompt
BID_WIN_EXTRACTION_PROMPT = """
You are an AI assistant specialized in parsing Bid Award emails for market research proposals.

The emails inform vendors that they’ve won a bid or been awarded a project. Extract the following structured metadata if present:
```json
{
  "email_type": "Bid Win",
  "from": "<Sender details>",
  "to": "<Recipient(s)>",
  "date": "<Full date and time>",
  "subject": "<Email subject>",
  "full_email_text": "<Complete message with headers and body>",
  "client_name": "<Client who awarded the bid>",
  "project_name": "<Title or reference of the project>",
  "scope_of_work": "<Details of what is to be done>",
  "expected_start_date": "<Start date if mentioned>",
  "budget": "<Budget or awarded cost>",
  "next_steps": "<Instructions post-award>",
  "attachments": "<Mentioned attachments like LOI or contract>",
  "signatory": "<Person awarding the project>"
}
"""

# Bid Rejection Email Prompt
BID_REJECTION_EXTRACTION_PROMPT = """
You are an AI assistant trained to extract metadata from Bid Rejection emails in the market research space.

These emails communicate that a vendor’s proposal was not selected. Extract structured data as shown below:
```json
{
  "email_type": "Bid Rejection",
  "from": "<Sender info>",
  "to": "<Recipient(s)>",
  "date": "<Date of email>",
  "subject": "<Subject line>",
  "full_email_text": "<Complete content with formatting>",
  "client_name": "<Client or issuing authority>",
  "project_name": "<Mentioned project>",
  "rejection_reason": "<If explicitly stated>",
  "awarded_to": "<If another vendor is mentioned>",
  "feedback": "<Constructive feedback if given>",
  "next_steps": "<Any possible follow-up>",
  "attachments": "<Mentioned documents>",
  "signatory": "<Name and role from email footer>"
}
"""

VALIDATION_PROMPT_TEMPLATE = """
You are a validation agent for email metadata extraction in the market research domain.

You will be given:
- The email category: {category}
- The extracted metadata as JSON:

{metadata_json}

Your task:
1. Verify if all required fields for the given category are present.
2. Check if the field values look plausible (non-empty, properly formatted).
3. Output a JSON object with:
   - "validation_status": "Valid" or "Invalid"
   - "confidence_score": float (0.0 to 1.0)

Rules:
- Required fields for each category:

RFP: client_name, deadline, estimated_cost, target_market, survey_length, incidents
Winning: client_name, project_name, scope_of_work, expected_start_date, budget
Rejection: client_name, project_name, rejection_reason

- Confidence score should reflect the completeness and plausibility of the extracted metadata.

Return only the JSON object.
"""

CONSOLIDATION_PROMPT = """You are a metadata consolidation agent.
You will receive multiple JSON metadata strings extracted from parts of the same email (e.g. multi-page email images).
Your task is to merge them into a single consolidated JSON according to email category: {category}.

Ensure fields like subject, sender, recipient, and dates are unified. Avoid duplication.
Return a single clean JSON only."""
