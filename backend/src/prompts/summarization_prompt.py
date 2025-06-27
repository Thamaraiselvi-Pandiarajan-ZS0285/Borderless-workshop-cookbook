SUMMARIZATION_PROMPT = """
You are an expert document summarization agent.
Your task is to read the document content and produce a clear and structured summary.

Instructions:
- Ensure that your summary is no longer than 2000 tokens. Be concise and avoid repetition.
- Keep limited to 3–4 lines.
- The entire summary must not exceed **75 tokens**.
- Exclude signatures, headers, footers, and formatting content.
- Focus only on main ideas and relevant details. Be concise and avoid repetition.
"""
TASK_VARIANTS: dict[str, str] = {
    "Project Description": """Describe the project in detail by answering the following:
What is the type and objective of the project?
Where is the project located (city, state, facility name)?
What is the scale and scope (capacity in kW, area in sq. ft., technology involved)?
What are the expected deliverables (e.g., technical specifications, timeline, monitoring capabilities, etc.)?""",

    "Pricing Requirements": """Summarize the financial expectations mentioned in the email, including:
Whether a commercial offer is requested
If the offer must include applicable taxes
If warranty and after-sales service details are required
Any other pricing-specific instructions or documents""",

    "Vendor Description": """Describe the required vendor profile, including:
Who the vendor is expected to be (company details such as name, location, and contact information if mentioned)
Vendor qualifications or prior experience (e.g., similar projects, technical expertise)
Required documentation (e.g., safety certifications, quality standards)
Any compliance requirements (e.g., MNRE guidelines, government standards)
Submission and clarification deadlines (date and time)"""
    }
