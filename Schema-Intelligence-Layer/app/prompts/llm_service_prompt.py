"""
Prompt templates for the LLM service.
Each function returns a fully-rendered prompt string ready to be sent to the LLM.
"""


def get_column_descriptions_prompt(context: str) -> str:
    """
    Prompt for generating short column descriptions.

    Args:
        context: Dataset metadata context built by _build_metadata_context().

    Returns:
        Rendered prompt string.
    """
    return f"""You are a data analyst. Given the following dataset metadata, generate a short, \
clear description (one sentence, max ~20 words) for each column. Base the description strictly on \
the column's name, data type, and sample values provided below — do not invent meaning that isn't \
supported by this evidence.

{context}

Rules:
- Include EVERY column listed in the metadata above, and ONLY those columns. Do not skip any, and \
do not add columns that are not present in the metadata.
- Each description must be a single sentence.
- If a column's purpose is unclear from the metadata, describe it based on its name and type and \
say so plainly (e.g. "Numeric identifier, likely a foreign key; purpose unclear from sample data.") \
rather than guessing a specific business meaning.

Return your response as a single valid JSON object, and nothing else — no markdown code fences, \
no ```json wrapper, no preamble, no trailing commentary. The response must start with {{ and end \
with }}.

Format:
{{
    "Column1": "Description of what Column1 represents",
    "Column2": "Description of what Column2 represents"
}}"""


def get_classify_dataset_prompt(
    context: str,
    desc_section: str,
) -> str:
    """
    Prompt for classifying a dataset into a business domain and sub-domain.

    Args:
        context: Dataset metadata context built by _build_metadata_context().
        desc_section: Optional block of generated column descriptions.

    Returns:
        Rendered prompt string.
    """
    return f"""You are a business data classification expert. Given the following dataset metadata, \
classify the dataset into the most appropriate business domain AND sub-domain, and provide a brief summary.

{context}
{desc_section}

Analyze the column names, types, descriptions, and sample values to determine:
1. The general business domain — prefer one of: Finance, Healthcare, E-commerce, Sales, Human \
Resources, Logistics, Marketing, Operations, Education, Real Estate, Insurance, Technology, Legal, \
Government, Manufacturing. Only use a different domain name if none of these reasonably fit.

Domains can overlap in vocabulary — many datasets contain currency amounts, budgets, or \
"actual vs. forecast" columns regardless of their actual subject. When that happens, prefer \
the most SPECIFIC domain whose distinctive terminology is present, rather than defaulting to \
a broader, more generic-sounding fit. Domain-specific signals to watch for:
- Finance: general ledger, revenue, cost center, P&L, treasury, chart of accounts — banking/\
corporate accounting itself, not another domain's financial activity
- Insurance: policy, premium, underwriting, claims, coverage, reinsurance, loss ratio, \
actuarial — insurance data is inherently financial but belongs here, not Finance
- Real Estate: property, listing, tenant, lease, square footage, mortgage, appraisal, occupancy
- Healthcare: patient, diagnosis, treatment, provider, clinical, prescription, admission
- Human Resources: employee, payroll, headcount, hire date, performance review, compensation, tenure
- Sales: deal, opportunity, pipeline, quota, lead, sales rep, CRM stage
- Marketing: campaign, ad spend, impressions, click-through, channel, conversion, audience segment
- Logistics: shipment, carrier, warehouse, freight, delivery route, fleet
- E-commerce: SKU, cart, order, product listing, marketplace, checkout
- Education: student, course, enrollment, grade, curriculum, semester
- Technology: API, deployment, server, repository, ticket, incident
- Legal: case, contract, litigation, compliance, clause, counsel
- Government: citizen, permit, agency, jurisdiction, public record, regulation
- Manufacturing: production line, defect rate, machine, batch, yield, work order
- Operations: process throughput, capacity, SLA, uptime, workflow — use only when no more \
specific domain's terminology is present

2. The specific sub-domain — a short, specific phrase (2-4 words) describing the purpose of the \
data within that domain (e.g. "Payments", "Clinical Trials", "Inventory Management", "Payroll", \
"Digital Ad Campaigns").

Base your classification only on evidence in the metadata above. If the data is ambiguous or \
generic, choose the closest reasonable domain and lower the confidence score accordingly rather \
than forcing a specific-sounding answer.

Return your response as a single valid JSON object, and nothing else — no markdown code fences, \
no ```json wrapper, no preamble, no trailing commentary. The response must start with {{ and end \
with }}. Fields:
- "business_domain": string, the general business domain name
- "sub_domain": string, the specific sub-domain within that domain
- "dataset_summary": string, 1-2 sentences describing what this dataset contains
- "confidence": a number (not a string) between 0.0 and 1.0 indicating classification confidence
- "reason": string, a brief explanation of why this domain and sub-domain were chosen, referencing \
specific column names as evidence

Example:
{{
    "business_domain": "Finance",
    "sub_domain": "Payments",
    "dataset_summary": "Contains transaction records, payment methods, and settlement data for a payments platform.",
    "confidence": 0.95,
    "reason": "Columns such as transaction_id, amount, and payment_method are typical of payments data."
}}"""