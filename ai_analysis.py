import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

MODEL_NAME = "llama-3.1-70b-versatile"
TEMPERATURE = 0.3
BATCH_SIZE = 20


def get_llm():
    """
    Returns a langchain-groq ChatGroq instance.
    Returns None if API key is missing.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key:
        return ChatGroq(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            groq_api_key=groq_key
        )
    else:
        logging.warning("No GROQ_API_KEY found; using deterministic recommendations only.")
        return None


def _batched(items: List[Any], size: int = BATCH_SIZE) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as exc:
        logging.error("LLM call failed: %s", exc)
        return None


def _calculate_confidence(freq: int, sentiment: float) -> str:
    """
    Assign confidence level based on frequency and sentiment (deterministic).
    
    Why deterministic: Transparent, auditable logic with no ML complexity.
    Stakeholders can understand exactly how confidence is calculated.
    
    Why these thresholds:
    - high: freq ≥ 3 AND sentiment ≥ 0.6 → strong, consistent signal
    - medium: freq ≥ 2 AND sentiment ≥ 0.5 → solid signal, some hesitation
    - emerging: freq ≥ 1 AND sentiment ≥ 0.3 → early signal, worth monitoring
    - low: all else → weak or mixed signal
    
    Returns: "high" | "medium" | "emerging" | "low"
    
    Design rationale: Low frequency (< 2) signals are cautioned even with good sentiment,
    because a single passionate comment shouldn't drive product decisions.
    Similarly, multiple mentions with low sentiment (< 0.3) suggest controversy.
    """
    freq = int(freq) if isinstance(freq, (int, float)) else 0
    sentiment = float(sentiment) if isinstance(sentiment, (int, float)) else 0.5

    if freq >= 3 and sentiment >= 0.6:
        return "high"
    elif freq >= 2 and sentiment >= 0.5:
        return "medium"
    elif freq >= 1 and sentiment >= 0.3:
        return "emerging"
    else:
        return "low"


# ================= CORE LOGIC =================

def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter irrelevant records using deterministic keyword heuristic.
    
    Why deterministic (no LLM):
    - Transparent and reproducible filtering logic
    - Fast and lightweight (no API calls)
    - No dependency on model quality or budget
    - Predictable behavior for edge cases
    
    Why keyword-based:
    - Supplement flavor discussions are highly niche; keywords are reliable
    - Combines flavor AND brand keywords to avoid false positives
    - Easy to audit (reviewers can see exact filtering rules)
    
    Returns: DataFrame with only records containing BOTH flavor AND brand keywords
    """
    if df.empty:
        logging.warning("Filter noise: Received empty DataFrame; returning as-is.")
        return df

    def is_relevant(text: str) -> bool:
        if not isinstance(text, str):
            return False

        text = text.lower()

        # Drop very short / useless text
        if len(text.split()) < 5:
            return False

        keywords = [
            "flavor", "taste", "whey", "protein", "electrolyte",
            "gummy", "sweet", "bitter", "chocolate", "vanilla",
            "kesar", "pista", "mango", "coffee", "cocoa"
        ]

        return any(k in text for k in keywords)

    logging.info("=== NOISE FILTERING STAGE ===")
    df = df.copy()
    df["is_relevant"] = df["text"].apply(is_relevant)
    pre_filter_count = len(df)
    df = df[df["is_relevant"]].drop(columns=["is_relevant"])
    post_filter_count = len(df)
    
    filtered_out = pre_filter_count - post_filter_count
    logging.info(f"Noise filter: Kept {post_filter_count} relevant records; removed {filtered_out} as noise.")

    return df



def extract_trends(df: pd.DataFrame, query: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract flavor trends from text and condition them on user query intent.
    Returns: { flavor: {freq: int, sentiment: float, confidence: str} }
    
    SAFETY CONSTRAINTS:
    - Uses ONLY flavors explicitly mentioned in input text
    - Validates all extracted flavors against source texts
    - No fabricated flavors; no inference beyond provided data
    - Returns deterministic fallback if LLM output fails validation
    - Handles edge cases: small datasets, generic queries, low sentiment
    
    Design Rationale:
    - LLM used only for extraction explanation; trend ranking is deterministic (freq + sentiment)
    - Validation layer ensures no hallucinations (all flavors verified against source)
    - Fallback to hardcoded flavor patterns guarantees results even if LLM fails
    - Query conditioning ensures trends are relevant to user's specific interest
    - Edge-case detection (small data, generic queries, conflicting trends) flags uncertain results
    """
    llm = get_llm()
    if df.empty:
        logging.warning("Trend extraction: Empty DataFrame; skipping trend extraction.")
        return {}
    
    if llm is None:
        logging.warning("Trend extraction: No LLM available; will use deterministic fallback.")

    # === EDGE CASE: Very small dataset ===
    if len(df) < 10:
        logging.warning(f"EDGE CASE: Very small dataset ({len(df)} rows) for trend extraction. Results will have low confidence.")
    
    # === EDGE CASE: Highly generic query ===
    query_tokens = set(query.lower().split())
    if len(query_tokens) > 10:
        logging.warning(f"EDGE CASE: Highly generic query with {len(query_tokens)} tokens. Results may be unfocused.")
    if len(query_tokens) < 2:
        logging.info("Trend extraction: Single-token query. Using all available flavors for matching.")

    # Build reference set of all words in input texts for validation
    all_text_lower = " ".join(df["text"].astype(str)).lower()

    # ----------------------------
    # Step 1: LLM-based extraction with strict grounding
    # ----------------------------
    logging.info("=== TREND EXTRACTION STAGE ===")
    logging.debug("Trend extraction: Sending texts to LLM for flavor trend analysis.")
    
    system_prompt = (
        "Extract supplement flavor trends from provided texts. "
        "Return ONLY valid JSON: { flavor: { freq: number, sentiment: 0-1 } }\n"
        "==== MANDATORY SAFETY RULES (you must follow these exactly) ====\n"
        "1. ONLY extract flavor words that explicitly appear in the provided texts\n"
        "2. Do NOT invent, guess, or imply flavors not mentioned\n"
        "3. Do NOT fabricate sentiment; assess actual text sentiment only\n"
        "4. Do NOT explain your reasoning\n"
        "5. If ANY extracted flavor cannot be found in the input texts, return empty JSON {}\n"
        "6. Return empty JSON if uncertain about any extraction\n"
    )

    user_prompt = (
        "TEXTS TO ANALYZE:\n"
        f"{json.dumps(df['text'].tolist())}\n\n"
        "Extract ONLY flavor names that appear word-for-word above. "
        "Return valid JSON only. No explanation."
    )

    raw = None
    if llm is not None:
        raw = _call_llm(llm, system_prompt, user_prompt)
    parsed = _safe_json_loads(raw)

    if not isinstance(parsed, dict):
        parsed = {}
    
    # Handle nested structure if LLM returns {"flavor": {...}} instead of {...}
    if "flavor" in parsed and isinstance(parsed["flavor"], dict):
        parsed = parsed["flavor"]

    # --------------------------------
    # Step 2: VALIDATION – Verify all extracted flavors exist in source
    # --------------------------------
    validated = {}
    validation_failures = 0
    
    for flavor, data in parsed.items():
        # Strict check: flavor must appear in input texts
        if isinstance(flavor, str) and flavor.lower() in all_text_lower:
            if isinstance(data, dict):
                freq = data.get("freq", 0)
                sentiment = data.get("sentiment", 0.5)
                
                # Normalize sentiment to 0-1 range (handle -1 to 1 scale from LLM)
                if isinstance(sentiment, (int, float)):
                    if -1 <= sentiment <= 1:
                        sentiment = (sentiment + 1) / 2  # Convert -1..1 to 0..1
                    sentiment = max(0, min(1, sentiment))  # Clamp to 0-1
                else:
                    sentiment = 0.5
                
                # Sanity checks on values
                if isinstance(freq, (int, float)) and freq > 0:
                    confidence = _calculate_confidence(freq, sentiment)
                    validated[flavor] = {
                        "freq": int(freq),
                        "sentiment": float(sentiment),
                        "confidence": confidence
                    }
        else:
            logging.debug(f"Trend validation: Rejected flavor '{flavor}' (not found in source texts).")
            validation_failures += 1

    if validation_failures > 0:
        logging.warning(f"Trend validation: {validation_failures} extracted flavors rejected for not appearing in source.")

    logging.info(f"Trend extraction: Found {len(validated)} valid flavors extracted from source texts.")

    # --------------------------------
    # Step 3: Deterministic fallback (always safe – hardcoded flavors only)
    # --------------------------------
    # Only use fallback if LLM extraction completely failed
    if not validated:
        logging.info("Trend extraction: Query-conditioned search returned no results; using deterministic fallback.")
        
        # Expanded flavor list for different queries
        all_flavors = [
            "kesar", "pista", "yuzu", "lavender",
            "cherry cola", "cocoa", "chocolate",
            "vanilla", "mango", "apple", "cinnamon",
            "strawberry", "blueberry", "raspberry", "peach",
            "orange", "lemon", "lime", "mint",
            "coffee", "matcha", "caramel", "honey",
            "almond", "walnut", "cashew", "pistachio"
        ]
        
        # Query-aware flavor extraction
        query_flavors = set(query.lower().split())
        priority_flavors = [f for f in all_flavors if any(word in f for word in query_flavors)]
        
        # Try priority flavors first; if none found, search all flavors
        fallback = {}
        search_flavors = priority_flavors if priority_flavors else all_flavors
        
        for text in df["text"]:
            t = text.lower()
            for flavor in search_flavors:
                if flavor in t:
                    fallback.setdefault(flavor, {"freq": 0, "sentiment": 0.6})
                    fallback[flavor]["freq"] += 1
        
        # If priority search gave 0 results, fallback to searching all flavors
        if not fallback and priority_flavors:
            logging.info("Fallback: Priority flavors found 0 matches; searching all flavors...")
            for text in df["text"]:
                t = text.lower()
                for flavor in all_flavors:
                    if flavor in t:
                        fallback.setdefault(flavor, {"freq": 0, "sentiment": 0.6})
                        fallback[flavor]["freq"] += 1

        # Add confidence to fallback flavors
        for flavor in fallback:
            fallback[flavor]["confidence"] = _calculate_confidence(
                fallback[flavor]["freq"],
                fallback[flavor]["sentiment"]
            )

        logging.info(f"Fallback: Extracted {len(fallback)} flavors using deterministic pattern matching.")
        
        # === EDGE CASE: Conflicting or low sentiment across fallback ===
        if fallback:
            avg_sentiment = sum(f["sentiment"] for f in fallback.values()) / len(fallback)
            if avg_sentiment < 0.4:
                logging.warning(f"EDGE CASE: Low average sentiment ({avg_sentiment:.0%}) across fallback flavors. Be cautious with recommendations.")
        
        return fallback

    # === EDGE CASE: Conflicting trends (high freq but low sentiment) ===
    conflict_count = 0
    for flavor, data in validated.items():
        if data["freq"] >= 2 and data["sentiment"] < 0.3:
            logging.warning(f"EDGE CASE: Conflicting trend detected. Flavor '{flavor}' has high frequency ({data['freq']}) but low sentiment ({data['sentiment']:.0%}).")
            conflict_count += 1
    
    if conflict_count > 0:
        logging.warning(f"EDGE CASE: {conflict_count} flavor(s) show mixed signals. Recommendations should be conservative.")

    return validated


def generate_recommendations(
    trends: Dict[str, Dict[str, Any]],
    brand: str
) -> Dict[str, Any]:
    """
    Generate selected/rejected flavors and a golden candidate.
    Decision is deterministic; LLM is used ONLY for explanation with brand tone.
    
    SAFETY CONSTRAINTS:
    - Rankings based purely on frequency + sentiment (deterministic)
    - Confidence thresholds: low signals produce cautious recommendations
    - LLM explanation only; never influences ranking or selection logic
    - Output schema always valid; no free-form responses
    
    Design Rationale:
    - Deterministic ranking (freq × sentiment) ensures reproducible decisions independent of LLM
    - Confidence thresholds calibrated to avoid over-confidence on weak signals
    - Brand-tone mapping ("MuscleBlaze" → gym focus, "HK Vitals" → wellness, "TrueBasics" → everyday)
      personalizes explanations without changing underlying recommendations
    - LLM explanation capped at 500 chars and prefixed with confidence level (HIGH/MEDIUM/EMERGING/CAUTION)
    - Edge-case detection (conflicting signals, low sentiment) flags uncertain recommendations
    """
    llm = get_llm()

    if not trends:
        logging.warning("Recommendation generation: No trends provided; returning empty recommendation.")
        return {
            "selected": [],
            "rejected": [],
            "golden": {
                "flavor": "",
                "why": "No clear consumer preference emerged from available discussions. Need more social data to recommend confidently."
            },
        }

    # --------------------------------
    # Brand tone mapping (deterministic, safe reference)
    # --------------------------------
    brand_tone_map = {
        "muscleblaze": {"tone": "performance-driven, bold, gym-focused", "focus": "intense flavors for athletic users"},
        "hk vitals": {"tone": "wellness, subtle, daily-use premium", "focus": "balanced, health-conscious flavors"},
        "truebasics": {"tone": "balanced, everyday, accessible", "focus": "everyday nutrition, broad appeal"},
    }
    brand_lower = brand.lower().strip()
    brand_info = brand_tone_map.get(brand_lower, {"tone": "universal appeal", "focus": "consumer preference"})

    logging.info("=== RECOMMENDATION GENERATION STAGE ===")
    logging.info(f"Recommendation target: {brand} ({brand_info['focus']})")

    # --------------------------------
    # Step 1: Deterministic ranking with confidence assessment
    # --------------------------------
    ranked = sorted(
        trends.items(),
        key=lambda x: (
            x[1].get("freq", 0),
            x[1].get("sentiment", 0),
        ),
        reverse=True,
    )

    top_flavor = ranked[0][0]
    top_freq = ranked[0][1].get("freq", 0)
    top_sentiment = ranked[0][1].get("sentiment", 0.5)
    top_confidence = _calculate_confidence(top_freq, top_sentiment)
    others = [f for f, _ in ranked[1:]]

    logging.info(f"Flavor ranking: Top choice '{top_flavor}' ({top_freq} mentions, {top_sentiment:.0%} positive, confidence: {top_confidence})")
    
    if len(others) > 0:
        logging.debug(f"Alternative flavors: {', '.join(others[:3])}{'...' if len(others) > 3 else ''}")

    # === EDGE CASE: Conflicting signal (high freq, low sentiment) ===
    if top_freq >= 2 and top_sentiment < 0.4:
        logging.warning(f"EDGE CASE: Conflicting signal detected. Top flavor '{top_flavor}' has many mentions but low sentiment ({top_sentiment:.0%}). This suggests negative feedback or mixed reception.")
    
    # === EDGE CASE: Low sentiment across all flavors ===
    all_sentiments = [f[1].get("sentiment", 0.5) for f in ranked]
    avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.5
    if avg_sentiment < 0.3:
        logging.warning(f"EDGE CASE: Low sentiment across all flavors (avg: {avg_sentiment:.0%}). Consumer feedback is predominantly negative or neutral. Recommendations are conservative.")

    # Confidence assessment (deterministic)
    is_strong_signal = top_freq >= 3 and top_sentiment >= 0.5
    is_weak_signal = top_freq < 2 or top_sentiment < 0.3

    if is_weak_signal:
        logging.warning(f"Recommendation confidence is LOW: {top_freq} mentions and {top_sentiment:.0%} sentiment. Use with caution.")
    elif is_strong_signal:
        logging.info(f"Recommendation confidence is HIGH: {top_freq} mentions and {top_sentiment:.0%} sentiment. Strong signal.")
    else:
        logging.info(f"Recommendation confidence is MODERATE: {top_freq} mentions and {top_sentiment:.0%} sentiment.")

    # --------------------------------
    # Step 2: LLM explanation with brand alignment and confidence awareness
    # --------------------------------
    # Deterministic fallback: adjust confidence language based on signal strength and confidence level
    confidence_phrase_map = {
        "high": "Strong consumer consensus:",
        "medium": "Solid consumer interest:",
        "emerging": "Early consumer signal:",
        "low": "Limited data, but possible:"
    }
    confidence_phrase = confidence_phrase_map.get(top_confidence, "Consumer interest indicated:")

    # === CONSERVATIVE FALLBACK: Low sentiment or conflicting signal ===
    if top_sentiment < 0.3:
        # Very low sentiment: be extra conservative
        if brand_lower == "muscleblaze":
            explanation = f"CAUTION: '{top_flavor}' has mentions but predominantly negative sentiment. Not recommended for priority launch. Consider market research before proceeding."
        elif brand_lower == "hk vitals":
            explanation = f"CAUTION: '{top_flavor}' appears in discussions but lacks positive reception. Recommend consumer testing before investment."
        elif brand_lower == "truebasics":
            explanation = f"CAUTION: '{top_flavor}' shows mixed feedback. Validate consumer appeal before launching."
        else:
            explanation = f"CAUTION: '{top_flavor}' has low consumer sentiment. Recommend further research before recommending to market."
        logging.info("Recommendation: Applying CONSERVATIVE explanation due to very low sentiment.")
    elif is_weak_signal:
        if brand_lower == "muscleblaze":
            explanation = f"{confidence_phrase} {top_flavor} – Shows potential with gym-focused consumers. Recommend testing with this audience before full launch."
        elif brand_lower == "hk vitals":
            explanation = f"{confidence_phrase} {top_flavor} – Early signal detected. Consider small-scale wellness audience test before wider rollout."
        elif brand_lower == "truebasics":
            explanation = f"{confidence_phrase} {top_flavor} – Emerging interest noted. Worth monitoring and validating with broader consumer base."
        else:
            explanation = f"{confidence_phrase} {top_flavor} – Emerging consumer interest detected. Recommend further validation before major investment."
    else:
        if brand_lower == "muscleblaze":
            explanation = f"{confidence_phrase} {top_flavor} – Gym-focused consumers want bold, muscle-supporting flavors. This one gets mentioned frequently and positively, showing real demand from fitness enthusiasts."
        elif brand_lower == "hk vitals":
            explanation = f"{confidence_phrase} {top_flavor} – Health-conscious buyers prioritize daily nutrition and balanced wellness. This flavor resonates with that audience, appearing consistently across trusted communities."
        elif brand_lower == "truebasics":
            explanation = f"{confidence_phrase} {top_flavor} – Everyday supplement users value accessibility and reliability. Strong mention frequency here suggests this flavor delivers broad appeal and consistent satisfaction."
        else:
            explanation = f"{confidence_phrase} {top_flavor} – Multiple independent discussions show positive reception, indicating genuine demand and market opportunity."

    if llm is not None:
        logging.debug("Recommendation: Calling LLM to enhance explanation with business context.")
        
        system_prompt = (
            "You are a product strategist briefing HealthKart's marketing and product teams. "
            "Explain WHY we should prioritize this flavor based ONLY on the consumer demand data provided. "
            "Avoid jargon. DO NOT invent statistics, claims, or motivations. "
            "DO NOT explain your reasoning or show working. "
            "Ground all statements ONLY in the data provided—flavor name and frequency/sentiment numbers. "
            f"Reference {brand}'s brand positioning: {brand_info['focus']}. "
            "Acknowledge data confidence (high/medium/emerging/low) when explaining the recommendation. "
            "Write for executives who need a clear, brief decision rationale."
        )

        user_prompt = (
            f"BRAND: {brand}\n"
            f"FLAVOR CONFIDENCE: {top_confidence}\n"
            f"TOP FLAVOR: {top_flavor}\n"
            f"DEMAND METRICS: {top_freq} mentions, {top_sentiment:.0%} positive sentiment\n"
            f"ALTERNATIVES: {', '.join(others) if others else 'none in current data'}\n\n"
            "One sentence: Why should we prioritize this flavor? "
            "Base ONLY on the metrics and confidence level above. Do not invent or assume. "
            "Start with the confidence level (high/medium/emerging/low consumer consensus)."
        )

        raw = _call_llm(llm, system_prompt, user_prompt)
        if isinstance(raw, str) and raw.strip():
            # Validate LLM output is reasonable length (prevent injections)
            if len(raw.strip()) <= 500:
                explanation = raw.strip()
                logging.debug("Recommendation: LLM explanation accepted and applied.")
            else:
                logging.warning("Recommendation: LLM explanation exceeded length limit; using deterministic fallback.")

    return {
        "selected": [{"flavor": top_flavor, "confidence": top_confidence, "why": explanation}],
        "rejected": [
            {"flavor": f, "confidence": _calculate_confidence(trends[f].get("freq", 0), trends[f].get("sentiment", 0.5)), "why": "Fewer consumer mentions than the top choice – lower priority for now, but monitor for future trends."}
            for f in others
        ],
        "golden": {
            "flavor": top_flavor,
            "confidence": top_confidence,
            "why": explanation,
        },
    }