# zoning_agent.py
"""
Query a chunked KB with an English resident request.

Pipeline:
1) LLM intent extraction (English -> structured intent + Dutch retrieval keywords)
2) Retrieval over Artikel chunks (boosted keyword scoring)
3) LLM decision constrained to evidence chunks only
4) Guardrails:
   - Permit-free only if evidence explicitly contains vergunningvrije / zonder omgevingsvergunning
   - No evidence => CONDITIONAL
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json, re

from rank_bm25 import BM25Okapi
from openai import OpenAI

PERMIT_FREE_RE = re.compile(
    r"\b(vergunningvrije|mag\s+zonder\s+omgevingsvergunning|toegestaan\s+zonder\s+omgevingsvergunning)\b",
    re.IGNORECASE
)

# -----------------------
# OpenAI client wrapper
# -----------------------

class OpenAILLMClient:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.output_text

# -----------------------
# Intent (LLM)
# -----------------------

INTENT_SYSTEM = """You extract structured intent from an English resident request about zoning/building rules in the Netherlands.
Return ONLY valid JSON (no markdown).

Schema:
{
  "plan_type": one of ["outbuilding","extension","dormer","garage","roof_raise","home_business","unit_split","use_change","other"],
  "permit_free_requested": boolean,   // true only if user explicitly asks permit-free / without a permit
  "area_m2": number|null,
  "height_m": number|null,
  "intended_use_en": string|null,     // e.g., "living space", "storage", "office"
  "retrieval_keywords_nl": string[],  // Dutch keywords likely in zoning rules
  "needs": string[]                  // missing facts required to decide permissibility/permit-free
}

Guidance:
- Always include Dutch keywords relevant to the plan: e.g. ["bijgebouw","bijbehorend bouwwerk","berging","tuinhuis","bouwregels","gebruiksregels","omgevingsvergunning","vergunningvrije","bouwhoogte","goothoogte","bouwvlak","achtererf","wonen","tuin"].
- Populate "needs" with any missing details typically required: placement (front/back/side), inside/outside bouwvlak, distance to boundaries, etc.
"""

def parse_intent(query_en: str, llm: OpenAILLMClient) -> Dict[str, Any]:
    raw = llm.complete(INTENT_SYSTEM, json.dumps({"query_en": query_en}, ensure_ascii=False))
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {}

    # minimal normalization defaults
    obj.setdefault("plan_type", "other")
    obj.setdefault("permit_free_requested", bool(re.search(r"\b(permit[- ]free|without a permit)\b", query_en.lower())))
    obj.setdefault("area_m2", None)
    obj.setdefault("height_m", None)
    obj.setdefault("intended_use_en", None)
    obj.setdefault("retrieval_keywords_nl", [])
    obj.setdefault("needs", [])
    if not isinstance(obj["retrieval_keywords_nl"], list):
        obj["retrieval_keywords_nl"] = []
    if not isinstance(obj["needs"], list):
        obj["needs"] = []
    return obj

# -----------------------
# Retrieval (boosted keyword scoring)
# -----------------------

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","is","are","be","by","at","it","this","that","we","our"
}

def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9à-ÿ]+", " ", s)
    return [t for t in s.split() if t and t not in STOPWORDS and len(t) > 2]

DUTCH_STOP = {
    "de","het","een","en","van","voor","op","in","met","te","dat","die","dit","als","bij",
    "aan","door","tot","uit","of","is","zijn","wordt","worden","mag","mogen","kan","kunnen",
    "niet","wel","geen","onder","boven","binnen","buiten","dan","maar","ook","meer","minder",
    "deze","daar","daarin","daarvoor","hier","hierin","hiermee","waar","waarbij","waaronder",
    "lid","artikel","hoofdstuk"
}

def tokenize_nl(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9à-ÿ]+", " ", s)
    toks = [t for t in s.split() if t and t not in DUTCH_STOP and len(t) > 2]
    return toks


def build_bm25_index(kb: dict) -> tuple[BM25Okapi, list[dict]]:
    chunks = []
    corpus = []
    for doc in kb.get("documents", []):
        for c in doc.get("chunks", []):
            chunks.append(c)
            corpus.append(tokenize_nl(c.get("text", "") or ""))
    bm25 = BM25Okapi(corpus)
    return bm25, chunks

PLAN_DEFAULT_KWS = {
    "outbuilding": ["bijgebouw","bijbehorend","bouwwerk","berging","tuinhuis","schuur","aanbouw","uitbouw"],
    "extension": ["aanbouw","uitbouw","bouwen","bouwvlak","bouwhoogte","goothoogte"],
    "dormer": ["dakkapel","dakopbouw","bouwhoogte","goothoogte"],
    "home_business": ["aan huis","beroep","bedrijf","praktijk","gebruiksregels","strijdig"],
}

def build_dutch_query_terms(intent: dict, kb: dict) -> list[str]:
    terms = []
    # strongest: intent keywords
    terms += intent.get("retrieval_keywords_nl", []) or []
    # fallback by plan type
    terms += PLAN_DEFAULT_KWS.get(intent.get("plan_type",""), [])
    # zoning area hints
    terms += kb.get("bestemmingsvlakken", []) or []
    # permit focus
    if intent.get("permit_free_requested"):
        terms += ["vergunningvrij","omgevingsvergunning","zonder","vergunning"]
    # normalize
    joined = " ".join(terms)
    return tokenize_nl(joined)

def score_chunk(query_toks: List[str], chunk_text: str, boosts: List[str]) -> float:
    ctoks = set(tokenize(chunk_text))
    overlap = sum(1.0 for t in query_toks if t in ctoks)

    lower = chunk_text.lower()
    boost = 0.0
    for b in boosts:
        if b and b.lower() in lower:
            boost += 2.0

    if PERMIT_FREE_RE.search(chunk_text):
        boost += 0.5

    return overlap + boost

def retrieve(kb: dict, query_en: str, intent: dict, top_k: int = 10):
    # Build (or cache) bm25 index
    bm25, chunks = build_bm25_index(kb)

    q_terms = build_dutch_query_terms(intent, kb)
    if not q_terms:
        # absolute fallback: use a tiny Dutch seed set
        q_terms = tokenize_nl("bouwregels gebruiksregels omgevingsvergunning wonen tuin")

    scores = bm25.get_scores(q_terms)

    # Optional: add simple boosts/penalties using headings
    results = []
    for s, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True):
        bonus = 0.0
        hp = " ".join(c.get("heading_path", [])).lower()

        # penalize transitional/definitions (usually irrelevant to new build)
        if any(w in hp for w in ["overgangsrecht", "begrippen", "slotregels"]):
            bonus -= 3.0

        # boost if chunk mentions a relevant bestemmingsvlak
        text_low = (c.get("text","") or "").lower()
        for z in kb.get("bestemmingsvlakken", []):
            if z and z.lower() in text_low:
                bonus += 1.5

        # boost if permit-related language is present and user asked about it
        if intent.get("permit_free_requested") and ("omgevingsvergunning" in text_low or "vergunningvrij" in text_low):
            bonus += 1.0

        final = float(s) + bonus
        if final > 0:
            results.append((final, c))

        if len(results) >= top_k * 5:
            # enough candidates before slicing
            break

    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:top_k]]


# -----------------------
# Decision (LLM) + guardrails
# -----------------------

DECIDE_SYSTEM = """You are a cautious zoning-compliance assistant.
Hard rules:
- Use ONLY the provided evidence chunks. Do not invent rules.
- If critical info is missing, answer CONDITIONAL and list assumptions.
- Permit-free: ONLY if evidence explicitly states 'vergunningvrije' or 'zonder omgevingsvergunning/vergunning'.

Output ONLY valid JSON with keys:
{
  "answer": "YES"|"NO"|"CONDITIONAL",
  "permit_status": "PERMIT_FREE"|"PERMIT_REQUIRED_OR_UNKNOWN",
  "reasoning_summary": string[],    // 3-6 bullets
  "minor_changes": string[],        // small changes that could flip outcome
  "assumptions": string[],
  "citations": string[]             // chunk_id list
}
“Return raw JSON only. Do NOT wrap the output in ``` fences.”
"""

def decide(kb: Dict[str, Any], query_en: str, intent: Dict[str, Any], evidence: List[Dict[str, Any]], llm: OpenAILLMClient) -> Dict[str, Any]:
    def fmt(c: Dict[str, Any]) -> str:
        hp = " > ".join(c.get("heading_path", []))
        t = c.get("text", "")
        t = t if len(t) <= 2200 else (t[:2200] + "\n...[truncated]")
        return f"chunk_id={c.get('chunk_id')} | doc={c.get('doc_title')} | path={hp}\n{t}\n"

    payload = {
        "address": kb.get("address", {}),
        "bestemmingsvlakken": kb.get("bestemmingsvlakken", []),
        "maatvoeringen": kb.get("maatvoeringen", []),
        "request_en": query_en,
        "intent": intent,
        "evidence": "\n\n".join(fmt(c) for c in evidence) if evidence else "(no evidence retrieved)",
    }
    raw = llm.complete(DECIDE_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        out = json.loads(raw)
    except Exception:
        out = {
            "answer": "CONDITIONAL",
            "permit_status": "PERMIT_REQUIRED_OR_UNKNOWN",
            "reasoning_summary": ["Model did not return valid JSON."],
            "minor_changes": [],
            "assumptions": ["Could not parse model output."],
            "citations": [c.get("chunk_id") for c in evidence],
        }
    return out

def guardrails(out: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Evidence missing => conditional
    if not evidence:
        out["answer"] = "CONDITIONAL"
        out["permit_status"] = "PERMIT_REQUIRED_OR_UNKNOWN"
        out.setdefault("assumptions", []).append("No relevant evidence chunks were retrieved.")
        out.setdefault("citations", [])
        return out

    # Permit-free gate
    if out.get("permit_status") == "PERMIT_FREE":
        supported = any(PERMIT_FREE_RE.search((c.get("text","") or "")) for c in evidence)
        if not supported:
            out["permit_status"] = "PERMIT_REQUIRED_OR_UNKNOWN"
            out.setdefault("assumptions", []).append(
                "Permit-free downgraded: no evidence explicitly states vergunningvrije / without permit."
            )

    # Ensure citations exist
    if not out.get("citations"):
        out["answer"] = "CONDITIONAL"
        out.setdefault("assumptions", []).append("No citations provided; treating as inconclusive.")
        out["citations"] = [c.get("chunk_id") for c in evidence if c.get("chunk_id")]

    return out

def answer_query(kb: Dict[str, Any], query_en: str, llm: OpenAILLMClient) -> Dict[str, Any]:
    intent = parse_intent(query_en, llm)
    evidence = retrieve(kb, query_en, intent, top_k=10)
    out = decide(kb, query_en, intent, evidence, llm)
    out = guardrails(out, evidence)
    return out

# -----------------------
# CLI
# -----------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--kb", required=True, help="chunked KB json")
    p.add_argument("--query", required=True, help="English resident request")
    p.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    p.add_argument("--api_key", required=True, help="OpenAI API key", default=None)
    args = p.parse_args()

    kb = json.loads(Path(args.kb).read_text(encoding="utf-8"))
    llm = OpenAILLMClient(model=args.model, api_key=args.api_key)
    out = answer_query(kb, args.query, llm)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
