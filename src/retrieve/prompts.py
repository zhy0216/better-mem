ASSEMBLE_PROMPT = """Given the user's query and a set of retrieved facts, do three things:
1. Select only the facts that are genuinely relevant to the query
2. Organize them into a coherent, chronological context paragraph
3. Note any information gaps

Query: {query}

Retrieved facts:
{facts}

User profile summary:
{profile_summary}

Return JSON:
{{
    "context": "A coherent paragraph summarizing relevant information...",
    "selected_fact_ids": ["id1", "id2"],
    "confidence": 0.85,
    "information_gaps": ["No information about accommodation preferences"]
}}"""
