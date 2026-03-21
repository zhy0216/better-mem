PROPOSITION_EXTRACTION_PROMPT = """You are a proposition extraction engine. Extract atomic propositions from the conversation below.

Rules:
1. Each proposition must be a single, self-contained, canonicalized statement
2. Each proposition must include WHO did/said WHAT and WHEN
3. Resolve pronouns to specific names
4. Resolve relative times to absolute dates based on the conversation timestamp
5. Classify each proposition into one of: observation, declaration, plan, preference, relation
6. Estimate importance (0.0 - 1.0): routine small talk = 0.1, specific plans = 0.7, life events = 0.9
7. For plans/events with a future time scope, include valid_from and valid_until
8. Filter out greetings, acknowledgments, and low-information content
9. For each proposition, add relevant tags (2-5 tags)
10. Set speaker_id to the ID of the participant whose information the proposition is primarily about.
    Use the participant list below to look up IDs by name. If uncertain, use null.
11. Set observed_at to the ISO-8601 timestamp of the message that produced this proposition.
12. For each proposition, generate a semantic_key that represents the semantic slot
    this proposition is answering. Rules:
    a. Use dot-separated lowercase English, e.g. "residence", "favorite_editor", "trip_tokyo_2024_04.budget"
    b. The key should describe WHAT QUESTION this proposition answers, not the answer itself.
       "residence" is correct, "lives_in_shanghai" is wrong.
    c. For time-scoped topics, include a time qualifier: "trip_tokyo_2024_04", not just "trip"
    d. For stable attributes (preferences, skills, relations), use simple keys: "preferred_language", "employer"
    e. If you cannot confidently determine the semantic slot, output null

Conversation timestamp: {timestamp}
Participants (name -> id): {participants}

Conversation:
{conversation}

Return a JSON object:
{{
    "propositions": [
        {{
            "canonical_text": "Zhang San plans to visit Tokyo in April 2024.",
            "proposition_type": "plan",
            "semantic_key": "trip_tokyo_2024_04",
            "observed_at": "2024-03-14T10:30:00Z",
            "importance": 0.7,
            "valid_from": "2024-04-01",
            "valid_until": "2024-04-30",
            "tags": ["travel", "tokyo", "plan"],
            "speaker_id": "user_001",
            "evidence_type": "utterance",
            "quoted_text": "I'm planning a trip to Tokyo next month."
        }}
    ]
}}"""


BELIEF_UPDATE_PROMPT = """You are evaluating how a new proposition relates to existing propositions about a user.

New proposition: {new_proposition}

Existing propositions (possibly related):
{existing_propositions}

For each existing proposition, determine:
- "supports": the new proposition reinforces or is consistent with the existing one
- "contradicts": the new proposition directly contradicts the existing one
- "updates": the new proposition is a newer version of the same information (e.g., old: "lives in Beijing", new: "moved to Shanghai")
- "unrelated": the propositions are about different things

For "contradicts" and "updates", these are treated the same way: the new proposition adds contradicting evidence to the old one, and supporting evidence to itself.

Return JSON:
{{
    "results": [
        {{
            "existing_proposition_id": "...",
            "relationship": "supports|contradicts|updates|unrelated",
            "reason": "brief explanation"
        }}
    ]
}}"""


PROFILE_SYNTHESIS_PROMPT = """Analyze the following propositions about a user and update their profile.
Only include propositions that the system has high confidence in.

Existing profile:
{existing_profile}

New propositions (since last update):
{new_propositions}

Instructions:
1. Merge new information with the existing profile
2. If new propositions contradict existing profile entries, prefer the newer information
3. Each profile attribute must reference the proposition IDs that support it
4. Profile attributes:
   - skills: [{{name, level (beginner/intermediate/expert), evidence_proposition_ids}}]
   - personality: [{{trait, evidence_proposition_ids}}]
   - preferences: [{{key, value, evidence_proposition_ids}}]
   - goals: [{{description, status (active/completed/abandoned), evidence_proposition_ids}}]
   - relations: [{{target_user_id, target_name, relation, evidence_proposition_ids}}]
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
