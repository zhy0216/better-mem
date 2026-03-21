FACT_EXTRACTION_PROMPT = """You are a fact extraction engine. Extract atomic facts from the conversation below.

Rules:
1. Each fact must be a single, self-contained proposition
2. Each fact must include WHO did/said WHAT and WHEN
3. Resolve pronouns to specific names
4. Resolve relative times to absolute dates based on the conversation timestamp
5. Classify each fact into one of: observation, declaration, plan, preference, relation
6. Estimate importance (0.0 - 1.0): routine small talk = 0.1, specific plans = 0.7, life events = 0.9
7. For plans/events with a future time scope, include valid_from and valid_until
8. Filter out greetings, acknowledgments, and low-information content
9. For each fact, add relevant tags (2-5 tags)
10. Set speaker_id to the ID of the participant whose information the fact is primarily about.
    Use the participant list below to look up IDs by name. If uncertain, use null.
11. Set occurred_at to the ISO-8601 timestamp of the message that produced this fact.
    Use the message timestamps from the conversation. If a message has no timestamp, use the conversation timestamp.

Conversation timestamp: {timestamp}
Participants (name -> id): {participants}

Conversation:
{conversation}

Return a JSON object:
{{
    "facts": [
        {{
            "content": "Zhang San plans to visit Tokyo in April 2024.",
            "fact_type": "plan",
            "occurred_at": "2024-03-14T10:30:00Z",
            "importance": 0.7,
            "valid_from": "2024-04-01",
            "valid_until": "2024-04-30",
            "tags": ["travel", "tokyo", "plan"],
            "speaker_id": "user_001"
        }}
    ]
}}"""


CONTRADICTION_CHECK_PROMPT = """You are checking whether new facts contradict existing facts about a user.

New fact: {new_fact}

Existing facts (possibly related):
{existing_facts}

For each existing fact, determine the relationship:
- "contradicts": the new fact directly contradicts the existing fact (e.g., old: "prefers Python", new: "hates Python")
- "updates": the new fact is an update or refinement of the existing fact (e.g., old: "lives in Beijing", new: "moved to Shanghai")
- "unrelated": the facts are about different things
- "consistent": the facts are compatible and non-contradictory

Return JSON:
{{
    "results": [
        {{
            "existing_fact_id": "...",
            "relationship": "contradicts|updates|consistent|unrelated",
            "reason": "brief explanation"
        }}
    ]
}}"""


PROFILE_SYNTHESIS_PROMPT = """Analyze the following facts about a user and update their profile.

Existing profile:
{existing_profile}

New facts (since last update):
{new_facts}

Instructions:
1. Merge new information with the existing profile
2. If new facts contradict existing profile entries, prefer the newer information
3. Each profile attribute must reference the fact IDs that support it
4. Profile attributes:
   - skills: [{{name, level (beginner/intermediate/expert), evidence_fact_ids}}]
   - personality: [{{trait, evidence_fact_ids}}]
   - preferences: [{{key, value, evidence_fact_ids}}]
   - goals: [{{description, status (active/completed/abandoned), evidence_fact_ids}}]
   - relations: [{{target_user_id, target_name, relation, evidence_fact_ids}}]
   - summary: One paragraph summary of the user

Return the complete updated profile as JSON:
{{
    "skills": [],
    "personality": [],
    "preferences": [],
    "goals": [],
    "relations": [],
    "summary": ""
}}"""
