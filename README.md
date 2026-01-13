# Zoning Compliance Agent

This project is a minimal Python MVP that checks whether a resident’s building or usage plan may be allowed under Dutch zoning regulations for a specific address
It works in two steps:
1. Build a chunked knowledge base (KB) from raw zoning JSON
2. Run an agent pipeline using an LLM to answer zoning questions

# Requirements
Python 3.10+

OpenAI API key


## Step 1 — Build Knowledge Base

Convert raw zoning JSON into a chunked KB.

python build_kb.py \
  --inp zoning_plan_Dennenweg_4__8172AJ_Vaassen.json \
  --out zp_kb_vaassen.json


Output:
zp_kb_vaassen.json (used for all queries at that address)
Repeat for each address.

## Step 2 — Ask Zoning Questions

Run the agent using the chunked KB:

python zoning_agent.py \
  --kb kb_vaassen.json \
  --query "I want to build a 20 m² outbuilding, 3 m high, used as living space. Can this be permit-free?"


The output is JSON with:
answer: YES / NO / CONDITIONAL 

permit_status: PERMIT_FREE or PERMIT_REQUIRED_OR_UNKNOWN 

reasoning_summary: short explanation 

minor_changes: small changes that could help 

assumptions: missing info or assumptions 

citations: chunk IDs used as evidence
