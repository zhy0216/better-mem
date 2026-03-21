ASSEMBLE_PROMPT = """Given the user's query and a set of retrieved propositions (each with a belief confidence score), do three things:
1. Select only the propositions that are genuinely relevant to the query
2. Organize them into a coherent, chronological context paragraph
3. Note any information gaps
4. Weight higher-confidence propositions more heavily; mention uncertainty for low-confidence ones

Query: {query}

Retrieved propositions (with confidence scores):
{propositions}

User profile summary:
{profile_summary}

Return JSON:
{{
    "context": "A coherent paragraph summarizing relevant information...",
    "selected_proposition_ids": ["id1", "id2"],
    "confidence": 0.85,
    "information_gaps": ["No information about accommodation preferences"]
}}"""
