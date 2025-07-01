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

PERSPECTIVE_PROMPT_VARIANTS: dict[str, str] = {
    "interview_type" :"""
            Analyze the RFP content to identify interview methodologies mentioned:
            - Quantitative research (surveys, polls, structured interviews)
            - Qualitative research (focus groups, in-depth interviews, ethnography)
            - Mixed methods approaches
            - Online vs offline data collection
            - Sample size requirements and methodology
            Summarize in 2-3 concise sentences focusing on research approach.""",

    "demographics" : """
            Extract demographic targeting information from the RFP:
            - Age groups, gender specifications
            - Income levels, education requirements
            - Occupation or professional categories
            - Household composition or family status
            - Any specific demographic exclusions
        """,
    "geographics" : """
            Identify geographical scope and requirements:
            - Target regions, countries, states, or cities
            - Urban vs rural specifications
            - Market penetration areas
            - Regional compliance or regulatory considerations
            - Language or cultural requirements by geography
           """,
    "sector": """ 
            Determine the industry sector and market focus:
            - Primary industry vertical (healthcare, finance, retail, etc.)
            - Sub-sectors or niche markets
            - B2B vs B2C orientation
            - Market maturity stage (emerging, established, declining)
            - Competitive landscape considerations
            Provide sector classification and key market characteristics.""",
    "methodology": """
            Extract research methodology and technical requirements:
            - Data collection methods and tools
            - Statistical analysis requirements
            - Quality assurance protocols
            - Technology platforms or software requirements
            - Validation and verification processes
            Summarize methodological approach and technical specifications.""",
    "timeline": """
            Identify project timeline and milestones:
            - Project duration and key phases
            - Submission deadlines and decision timelines
            - Data collection periods
            - Reporting and deliverable schedules
            - Critical path dependencies
            """,
    "budget_scope": """
            Analyze budget and financial scope indicators:
            - Project scale indicators (without specific amounts)
            - Cost structure preferences (fixed vs variable)
            - Payment terms and milestone-based payments
            - Value-added services expectations
            - ROI or cost-effectiveness criteria
            """,
    "deliverables": """
            Extract expected deliverables and outcomes:
            - Report formats and presentation requirements
            - Data visualization and dashboard needs
            - Raw data delivery specifications
            - Presentation and stakeholder communication requirements
            - Follow-up services or ongoing support
            Summarize deliverable expectations and format requirements.
            """,
    "vendor_requirements": """
            Identify vendor qualification and capability requirements:
            - Industry experience and track record
            - Team composition and expertise requirements
            - Certification and compliance needs
            - Technology infrastructure capabilities
            - Geographic presence or partnership requirements
            Summarize vendor profile requirements"""

}
