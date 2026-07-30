"""Prompt templates for LLM interactions."""

SCHEMA_INTELLIGENCE_SYSTEM = """You are a data schema intelligence system. Your job is to confirm or override 
semantic type and column role candidates for dataset columns.

Rules:
- You MUST NOT change the physical data type.
- You may confirm or override the candidate semantic type and column role.
- If you override, provide a more specific semantic type from the domain context.
- Return confidence between 0.0 and 1.0.
- Decision must be one of: confirmed, overridden, unresolved.
- Respond ONLY with valid JSON matching the schema. No extra text."""

SCHEMA_INTELLIGENCE_PROMPT_V1 = """Analyze these columns and confirm or override their semantic types and roles.

Primary Domain: {primary_domain}
Dataset Context: {row_count} rows, {column_count} columns

Columns to analyze:
{columns_json}

For each column, decide:
1. Is the candidate semantic type correct? If so, decision="confirmed".
2. Should it be overridden with a more specific type? If so, decision="overridden" and provide the new type.
3. If uncertain, decision="unresolved".

Also recommend whether each column should be:
- mandatory (true/false/null if unsure)
- expected_unique (true/false/null if unsure)

Return JSON with this structure:
{{
  "columns": [
    {{
      "column_name": "col_name",
      "decision": "confirmed|overridden|unresolved",
      "confirmed_semantic_type": "type_or_null",
      "confirmed_column_role": "role_or_null",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation",
      "recommended_mandatory": true/false/null,
      "recommended_expected_unique": true/false/null
    }}
  ],
  "model_name": "model_identifier",
  "prompt_version": "si-v1"
}}"""

SECONDARY_DOMAIN_SYSTEM = """You are a domain classification system. You classify datasets into secondary domains.

Rules:
- You MUST select from the provided allowed secondary domains ONLY.
- Do NOT invent new domains.
- Return confidence between 0.0 and 1.0.
- If no domain fits well, set selected_domain to null.
- Respond ONLY with valid JSON. No extra text."""

SECONDARY_DOMAIN_PROMPT_V1 = """Classify this dataset into one secondary domain.

Primary Domain: {primary_domain}
Allowed Secondary Domains: {allowed_domains}

Dataset evidence:
- Column names: {column_names}
- Semantic types detected: {semantic_types}
- Column roles: {column_roles}
- Representative values sample: {sample_values}

Select the most appropriate secondary domain from the allowed list.

Return JSON:
{{
  "selected_domain": "domain_name_or_null",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "evidence": ["signal1", "signal2"]
}}"""

RULE_SUGGESTION_SYSTEM = """You are a business rule discovery system. Propose candidate business rules
based on the dataset profile.

Rules:
- Only suggest rules using one of these exact supported types — these are the only
  types the evaluation engine can execute:
  - non_null: requires target_column
  - expected_unique: requires target_column
  - regex_match: requires target_column, pattern
  - allowed_values: requires target_column, values (list of strings)
  - numeric_range: requires target_column, and at least one of min_value/max_value
    (inclusive_min/inclusive_max default to true)
  - column_comparison: requires left_column, operator (one of >=, >, <=, <, ==, !=), right_column
  - conditional_required: requires condition_column, condition_value, required_column
- target_column / left_column / right_column / condition_column / required_column MUST be
  exact column names copied from the column profiles given to you. Never invent a column name.
- rule_key must be a short, unique, snake_case identifier (e.g. "amount_non_negative").
- Only populate the fields relevant to the chosen type; leave the rest at their defaults.
- Provide confidence between 0.0 and 1.0, reflecting how confident you are the rule holds
  given the evidence (nulls, distinct counts, sample values) in the profile.
- Respond ONLY with valid JSON matching the schema. No extra text."""

RULE_SUGGESTION_PROMPT_V1 = """Analyze this dataset profile and suggest business rules.

Primary Domain: {primary_domain}
Secondary Domain: {secondary_domain}

Column profiles:
{columns_summary}

Suggest up to 5 business rules that should hold for this data, using only the supported
types and their required fields described above.

Return JSON:
{{
  "suggestions": [
    {{
      "rule_key": "short_unique_snake_case_key",
      "type": "one_of_the_supported_types",
      "description": "human readable rule",
      "reasoning": "why this rule should hold, citing evidence from the profile",
      "confidence": 0.0-1.0,
      "severity": "low|medium|high",
      "target_columns": ["col1"],
      "target_column": "col_name_or_null",
      "pattern": "regex_or_null",
      "values": ["allowed", "values"],
      "min_value": 0.0,
      "max_value": null,
      "inclusive_min": true,
      "inclusive_max": true,
      "left_column": "col_name_or_null",
      "operator": ">=",
      "right_column": "col_name_or_null",
      "condition_column": "col_name_or_null",
      "condition_value": "value_or_null",
      "required_column": "col_name_or_null"
    }}
  ]
}}

Omit or null out any field not relevant to the chosen type — only include the fields
that type actually needs, plus rule_key, type, description, reasoning, confidence,
severity, and target_columns which are always required."""
