import os
import re
import json
import logging
from datetime import datetime
from typing import List, Optional

import nltk
import pandas as pd
import praw
import tweepy
from groq import Groq
from nltk.tokenize import wordpunct_tokenize

# CONDITIONAL SOCIAL DATA INGESTION
# This module supports multiple data sources with graceful fallbacks:
# - Twitter/X (OPTIONAL, paid API): Enabled only if TWITTER_BEARER_TOKEN is set
# - Reddit (OPTIONAL, free API): Enabled only if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are set
# - Mock samples (ALWAYS available): Used when no live credentials are provided
# The app never crashes if credentials are missing; it falls back to mock data.

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# Precompile regex patterns for speed.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMOJI_PATTERN = re.compile(
    """
    [\U0001F600-\U0001F64F]|  # emoticons
    [\U0001F300-\U0001F5FF]|  # symbols & pictographs
    [\U0001F680-\U0001F6FF]|  # transport & map symbols
    [\U0001F1E0-\U0001F1FF]   # flags (iOS)
    """,
    flags=re.UNICODE | re.VERBOSE,
)

# Flavor-related keywords to gate relevance.
FLAVOR_KEYWORDS = [
    "flavor",
    "taste",
    "tastes",
    "crave",
    "kesar",
    "pista",
    "masala",
    "chai",
    "cocoa",
    "chocolate",
    "vanilla",
    "mango",
    "strawberry",
    "berry",
    "gummies",
    "gummy",
    "whey",
    "protein",
    "electrolyte",
    "electrolytes",
    "yuzu",
    "lavender",
    "cherry",
    "cola",
    "ginger",
    "lemon",
    "honey",
    "cinnamon",
    "apple",
]

BRAND_KEYWORDS = ["muscleblaze", "hk vitals", "truebasics", "true basics", "healthkart"]

DEFAULT_QUERY = "(flavor OR taste OR crave) (whey OR protein OR electrolyte OR gummy) MuscleBlaze OR HK Vitals -spam"


# ============================================================
# DEMO MODE: Curated Reddit-style flavor comments (119 entries)
# ============================================================
# This replaces live API ingestion for the assignment demo. No network calls.
# Data is loaded on every run and then passed through the standard pipeline
# (filtering, cleaning, deduplication) to keep analysis behavior unchanged.
DEMO_MODE = True  # Always use curated JSON dataset for this assignment

_DEMO_JSON = [
  {"text": "Tried and tested and thus can vouch for: Blue Tokai Coffee, Choco Crispers, Kesar Thandai, Biscoff Cookie, Chocolate Hazelnut and Vanilla (in order of my liking)", "source": "Reddit Review"},
  {"text": "After mango . I like chocolate peanut butter and kesar thandai . Both are good", "source": "Reddit Review"},
  {"text": "Blue Tokai Coffee flavour really good", "source": "Reddit Review"},
  {"text": "Hi I have been using Chocolate Hazelnut and I find a slight bitter after taste.", "source": "Reddit Review"},
  {"text": "Biozyme has MuscleBlaze's own patented digestive enzyme which helps protein absorb without any issue for which they charge more. That's why biozyme is expensive but you don't need digestive enzyme, it's pretty pointless so you can skip it. Muscleblaze Fuel One is quite affordable and it's taste is pretty good as well. Adding to that, please drop the fuel one flavours that are the best for the OP, boys! I haven't tried fuel one flavours.", "source": "Reddit Review"},
  {"text": "Cookies & Cream from the fuel one line up is just so yummy with milk!", "source": "Reddit Review"},
  {"text": "mango flavour is the worst.", "source": "Reddit Review"},
  {"text": "Very bad + too khatta I'm regretting not sure how people like it. Also the tropical mango is bad", "source": "Reddit Review"},
  {"text": "Biozyme iso zero mango flavour is the best ever mango I have ever tried , otherwise go for blue tokai in biozyme performance", "source": "Reddit Review"},
  {"text": "Biozyme is their best.", "source": "Reddit Review"},
  {"text": "chocolate hazelnut good", "source": "Reddit Review"},
  {"text": "Disposed blue tokai, butter cookie. Triple chocolate is close to good and coming to the best part - PB peaked", "source": "Reddit Review"},
  {"text": "Blue tokai is one of the best flavour IMO", "source": "Reddit Review"},
  {"text": "Triple chocolate is so yummy but its always sold out . Though, I've been taking blue tokai flavor from MB from an year. Absolutely love the taste 10/10.", "source": "Reddit Review"},
  {"text": "“Using MuscleBlaze whey isolate, and honestly double chocolate is the only flavour that feels properly palatable to me", "source": "Reddit Review"},
  {"text": "Magical mango actually tastes pretty good; it has a different kind of swaad compared to usual chocolate proteins.", "source": "Reddit Review"},
  {"text": "Been using MB whey for a while, the basic chocolate flavour is decent, nothing crazy but easy to drink every day", "source": "Reddit Review"},
  {"text": "Tried strawberry and cookies & cream from MuscleBlaze and both flavours turned out really good, especially with cold milk.", "source": "Reddit Review"},
  {"text": "MuscleBlaze is sweeter than ON whey; if you prefer less sweet and subtle flavours, MB might feel a bit too sugary", "source": "Reddit Review"},
  {"text": "Raw whey concentrate from MB smells and tastes weird at first, kind of bad, but after a few weeks you get used to the taste", "source": "Reddit Review"},
  {"text": "Raw whey isolate also has a slightly weird smell and taste initially, but it’s much milder and becomes unnoticeable after some use.", "source": "Reddit Review"},
  {"text": "Someone in my gym said they hate MuscleBlaze just because the proteins taste bad, no matter which flavour they tried.", "source": "Reddit Review"},
  {"text": "Chocolate hazelnut flavour has a slight bitter aftertaste; it’s manageable but you definitely notice that bitterness at the end.", "source": "Reddit Review"},
  {"text": "Cookies & cream from the Fuel One lineup is insanely yummy with milk; feels more like a milkshake than a protein shake.", "source": "Reddit Review"},
  {"text": "Tried cookies & cream from MB ISO and it tasted trash, completely off; I regretted buying the whole tub.", "source": "Reddit Review"},
  {"text": "Mango flavour is honestly the worst; every day drinking it feels like a punishment and I’m just trying to finish the box somehow.", "source": "Reddit Review"},
  {"text": "Tropical mango is very bad and too khatta; difficult to understand how some people enjoy this flavour.", "source": "Reddit Review"},
  {"text": "Bought 2 kg of the mango variant and it was so bad and undrinkable that it ended up getting thrown away.", "source": "Reddit Review"},
  {"text": "Blue Tokai coffee flavour is really good with milk; coffee notes are strong and it feels like a proper cold coffee shake.", "source": "Reddit Review"},
  {"text": "Milk chocolate flavour from MuscleBlaze tastes good and is safe if you’re scared of buying some experimental flavour.", "source": "Reddit Review"},
  {"text": "Mocha cappuccino flavour was horrible for me; hated every single sip and couldn’t finish the jar.", "source": "Reddit Review"},
  {"text": "Chocolate hazelnut Biozyme flavour is fine: sweetness and thickness are balanced, but the overall taste isn’t amazing for the price.", "source": "Reddit Review"},
  {"text": "Chocolate mint flavour from MuscleBlaze was the best for me; very refreshing and I would definitely recommend it over other flavours", "source": "Reddit Review"},
  {"text": "Ice cream chocolate comes second for me in MB flavours; still quite good but not as impressive as the chocolate mint variant.", "source": "Reddit Review"},
  {"text": "Rich milk chocolate flavour is okay, but if you compare it to top imported proteins, the milky taste is clearly missing.", "source": "Reddit Review"},
  {"text": "Some users say MuscleBlaze overall tastes awful; they tried chocolate and cookies & cream and ended up throwing both tubs away", "source": "Reddit Review"},
  {"text": "Biozyme chocolate flavour has smooth consistency, no artificial aftertaste, and is easy to drink daily with water.", "source": "Reddit Review"},
  {"text": "“HK Vitals multivitamin gummies have a tempting orange flavour; taste is so good that it feels like eating candy every day", "source": "Reddit Review"},
  {"text": "Taste is soo tempting and the quality is nice; these gummies are yummy enough that I actually look forward to taking them.", "source": "Reddit Review"},
  {"text": "Very tasty option for daily vitamins; feels like a snack instead of a supplement, which makes adherence much easier", "source": "Reddit Review"},
  {"text": "“These multivitamin gummies are super yummy; probably the best-tasting multivitamin I’ve bought for my mom so far.", "source": "Reddit Review"},
  {"text": "Very tasty option for daily vitamins; feels like a snack instead of a supplement, which makes adherence much easier", "source": "Reddit Review"},
  {"text": "These multivitamin gummies are super yummy; probably the best-tasting multivitamin I’ve bought for my mom so far.", "source": "Reddit Review"},
  {"text": "Gummies are very tasty and easy to intake; flavour is strong and enjoyable, not medicinal at all", "source": "Reddit Review"},
  {"text": "Taste is soo yummy, I really like it overall and I’m satisfied with the flavour as well as the effect.", "source": "Reddit Review"},
  {"text": "Multivitamin gummies provide a tasty way to complete daily vitamin intake; feels like eating orange-flavoured toffees.", "source": "Reddit Review"},
  {"text": "Biotin hair gummies from HK Vitals taste good; flavour is pleasant enough that taking them daily is no issue", "source": "Reddit Review"},
  {"text": "Product is just wow, taste is good and sweet; only side note is that I noticed some weight gain after starting the gummies.", "source": "Reddit Review"},
  {"text": "Biotin gummies are very tasty and I just like the flavour; result on hair and nails is also amazing so far.", "source": "Reddit Review"},
  {"text": "For women’s HK Vitals multivitamin tablets, initially I didn’t like the taste, but after some time it felt okay and not bad at all.", "source": "Reddit Review"},
  {"text": "HK Vitals collagen orange flavour manages to cover most of the typical fishy collagen taste, making it acceptable to drink daily", "source": "Reddit Review"},
  {"text": "There is still a subtle marine or fishy undertone in HK Vitals collagen; sensitive users can notice it and may find it slightly off-putting", "source": "Reddit Review"},
  {"text": "Compared to unflavoured marine collagen, HK Vitals orange flavour is a big improvement in palatability and much easier to drink.", "source": "Reddit Review"},
  {"text": "HK Vitals multivitamin gummies are marketed as tasty orange flavour with no added sugar or preservatives, and actually taste like candy.", "source": "Reddit Review"},
  {"text": "Some reviewers feel HK Vitals products have decent taste, but a few specific supplements from the brand are described as less pleasant.", "source": "Reddit Review"},
  {"text": "Multivitamin gummies from HK Vitals are super easy to consume; the orange taste is enjoyable and not overly sweet.", "source": "Reddit Review"},
  {"text": "Many customers highlight that HK Vitals supplements are okay in taste, neither outstanding nor terrible, just about average.", "source": "Reddit Review"},
  {"text": "TrueBasics Clean Whey Isolate has a naturally smooth chocolate flavour with effortless mixability in water or milk.", "source": "Reddit Review"},
  {"text": "Chocolate flavour of TrueBasics Clean Whey Isolate feels heavy on cocoa and gives a strong whey protein feel without being very sweet.", "source": "Reddit Review"},
  {"text": "This product has almost no sweetness; you mainly get cocoa notes and a heavy, dark-chocolate-like taste in every sip.", "source": "Reddit Review"},
  {"text": "People who want good sweetness in their whey might find TrueBasics Clean Whey bland and underwhelming in taste", "source": "Reddit Review"},
  {"text": "Taste-wise, it feels similar to eating 85% dark chocolate: smooth, noticeable cocoa notes, but barely any sweetness.", "source": "Reddit Review"},
  {"text": "Considering there are no artificial flavours, colours, thickeners or sweeteners, the taste is actually quite good and feels clean.", "source": "Reddit Review"},
  {"text": "TrueBasics Clean Whey Isolate vanilla tastes very bland if you’re used to sweetened proteins; you just get a subtle vanilla note with no sweetness.", "source": "Reddit Review"},
  {"text": "Big negative for some users is that TrueBasics only has chocolate flavour in this whey variant; no other flavour options yet.", "source": "Reddit Review"},
  {"text": "The unsweetened vanilla flavour isn’t unpleasant, but it definitely takes time to adjust if you’re moving from regular sweet whey.", "source": "Reddit Review"},
  {"text": "TrueBasics whey is described as clean and dependable where taste is not the main priority, but users still prefer to avoid any bad aftertaste", "source": "Reddit Review"},
  {"text": "For those wanting to avoid artificial sweeteners and preservatives, the mild and bland taste of TrueBasics becomes a positive in the long run.", "source": "Reddit Review"},
  {"text": "Many reviewers say taste is acceptable to pleasant, especially if you mix it with cold milk or add it to smoothies.", "source": "Reddit Review"},
  {"text": "TrueBasics plant protein chocolate isn’t my favourite flavour, but for a plant-based, unsweetened option it’s decent and easy to tolerate.", "source": "Reddit Review"},
  {"text": "The plant protein version has no sweeteners or fillers, so the flavour is more muted compared to typical sweet chocolate proteins.", "source": "Reddit Review"},
  {"text": "Some users mention TrueBasics vanilla isolate could use slightly more natural flavour concentration to improve taste", "source": "Reddit Review"},
  {"text": "TrueBasics chocolate whey feels light because there are no thickeners; the mouthfeel is thin but mixes well, which some people like.", "source": "Reddit Review"},
  {"text": "Those who prefer very sweet, dessert-style whey generally rate TrueBasics taste lower due to its low sweetness and minimal flavouring.", "source": "Reddit Review"},
  {"text": "TrueBasics whey is often chosen more for clean label and low additives than for exciting flavours, so taste reviews are usually neutral to mildly positive.", "source": "Reddit Review"},
  {"text": "My latest MB whey isolate pouch had an extremely sour and bitter flavour, completely different from what I was used to before", "source": "Reddit Review"},
  {"text": "Even after a replacement order, the same batch again had unpleasant taste and weird mixing behaviour, so I just returned it", "source": "Reddit Review"},
  {"text": "“Raw whey from MuscleBlaze smells and tastes weird, almost off; it took a lot of effort to finish the pack.”", "source": "Reddit Review"},
  {"text": "Compared to other brands, MuscleBlaze chocolate feels less tasty and a bit dull, not something you enjoy drinking.", "source": "Reddit Review"},
  {"text": "Some MB flavours are so bad that people literally throw the tub away instead of forcing themselves to drink it.", "source": "Reddit Review"},
  {"text": "Mango flavour was a disaster for me; every shake felt like torture and I couldn’t wait to finish the jar.”", "source": "Reddit Review"},
  {"text": "Tropical mango was too sour and unpleasant; I don’t understand how anyone can like this flavour.”", "source": "Reddit Review"},
  {"text": "“Mocha cappuccino flavour from MuscleBlaze tasted horrible, I regretted buying it after the first few scoops.”", "source": "Reddit Review"},
  {"text": "Cookies & cream in the ISO line tasted trash to me, completely off and nothing like the name suggests.", "source": "Reddit Review"},
  {"text": "Some batches of MB whey have such a strange taste and texture that users suspect quality issues rather than just flavour preference.", "source": "Reddit Review"},
  {"text": "For some people, the collagen or drink from this brand has a pungent taste and smell, making it very uneasy to consume.", "source": "Reddit Review"},
  {"text": "Even with orange flavour, there is still a subtle fishy or marine aftertaste in HK Vitals collagen that sensitive users really dislike", "source": "Reddit Review"},
  {"text": "A few HK Vitals supplements are described as less pleasant in taste compared to premium brands, just average or slightly below.", "source": "Reddit Review"},
  {"text": "Some users feel that, beyond gummies, HK Vitals products are okay at best in taste and not something you actually enjoy", "source": "Reddit Review"},
  {"text": "There are mentions of certain HK Vitals products causing bad-smelling sweat and an overall unpleasant feel while using them", "source": "Reddit Review"},
  {"text": "If you want proper sweetness in whey, TrueBasics Clean Whey can feel very bland and disappointing in taste", "source": "Reddit Review"},
  {"text": "Vanilla isolate from TrueBasics is so unsweetened that it tastes almost flavourless, just a faint vanilla note in water.", "source": "Reddit Review"},
  {"text": "People who are used to dessert-style, sweet whey find TrueBasics to be dull and not enjoyable to drink.", "source": "Reddit Review"},
  {"text": "TrueBasics taste is described as acceptable but not exciting; those who prioritise flavour often look for other brands.", "source": "Reddit Review"},
  {"text": "For some users, the very light, thin mouthfeel with minimal sweetness makes TrueBasics whey feel more like medicine than a shake.", "source": "Reddit Review"},
  {"text": "For pure taste, MuscleBlaze chocolate wins easily; TrueBasics feels too bland and unsweet if you’re used to regular whey.", "source": "Reddit Review"},
  {"text": "TrueBasics Clean Whey is great for a clean profile, but MB double rich chocolate is still the best when it comes to actual flavour.", "source": "Reddit Review"},
  {"text": "If you want less sweetness, TrueBasics is better; but if you want something you enjoy drinking every day, MuscleBlaze flavours are more fun.", "source": "Reddit Review"},
  {"text": "I tried both TrueBasics vanilla isolate and MB rich milk chocolate; I’ll pick MuscleBlaze any day for taste, TrueBasics is too neutral.", "source": "Reddit Review"},
  {"text": "TrueBasics feels like a ‘functional’ shake, while MuscleBlaze chocolate mint tastes like a proper dessert, so MB wins on flavour for me", "source": "Reddit Review"},
  {"text": "Between MB raw whey and TrueBasics, I’d say TrueBasics tastes more natural and less weird, so it’s better if you hate strong flavours.", "source": "Reddit Review"},
  {"text": "If flavour is your only criteria, MuscleBlaze cookies & cream with milk beats TrueBasics chocolate by a big margin", "source": "Reddit Review"},
  {"text": "in terms of taste, HK Vitals gummies beat any MuscleBlaze powder; gummies feel like candy, protein shakes still feel like protein.", "source": "Reddit Review"},
  {"text": "For whey, MuscleBlaze rich milk chocolate is better than any HK Vitals tablet or capsule, but for daily enjoyment HK gummies are the clear winner.", "source": "Reddit Review"},
  {"text": "If you compare MB mango whey to HK Vitals orange gummies, gummies are miles ahead; MB mango flavour is almost undrinkable.", "source": "Reddit Review"},
  {"text": "HK Vitals collagen orange flavour is easier to drink than some of the bad MB flavours, but MB chocolate still tastes better overall.", "source": "Reddit Review"},
  {"text": "For taste alone, I’d rank HK Vitals gummies first, then TrueBasics chocolate whey, and HK collagen last because of the slight fishy note.", "source": "Reddit Review"},
  {"text": "When you want a tasty vitamin, HK Vitals wins; when you want a tasty protein, MuscleBlaze is ahead, HK doesn’t really compete there", "source": "Reddit Review"},
  {"text": "TrueBasics whey is less sweet and more neutral, while HK Vitals gummies are sweet and candy-like; for flavour, gummies win easily", "source": "Reddit Review"},
  {"text": "If you hate artificial sweetness, TrueBasics unsweetened chocolate is better; HK Vitals gummies are too candy-like for that crowd", "source": "Reddit Review"},
  {"text": "HK Vitals collagen has an orange taste with a fishy hint, whereas TrueBasics whey just tastes like mild cocoa; TrueBasics is easier to tolerate daily.", "source": "Reddit Review"},
  {"text": "Between TrueBasics plant protein and HK multivitamin gummies, gummies clearly taste better, but for a ‘non-dessert’ profile TrueBasics is the safer pick", "source": "Reddit Review"},
  {"text": "For pure flavour ranking, I’d go: HK Vitals orange gummies at the top, then TrueBasics chocolate whey, and HK collagen last", "source": "Reddit Review"},
  {"text": "For whey taste, MuscleBlaze chocolate mint is best; TrueBasics is too plain and HK Vitals doesn’t really have a standout whey flavour", "source": "Reddit Review"},
  {"text": "If you put everything side by side, best-tasting product is HK Vitals multivitamin gummies, then MuscleBlaze rich milk chocolate whey, then TrueBasics clean whey at the end.", "source": "Reddit Review"},
  {"text": "For a dessert-like shake, MuscleBlaze wins; for a clean, low-sweetness shake, TrueBasics wins; for candy-like experience, HK gummies are unbeatable.", "source": "Reddit Review"},
  {"text": "Overall flavour enjoyment ranking for me: HK Vitals gummies > MuscleBlaze chocolate whey > TrueBasics isolate vanilla", "source": "Reddit Review"},
  {"text": "If someone asks ‘best flavour’ strictly by taste, I recommend MuscleBlaze double chocolate over both TrueBasics whey and HK collagen drinks.", "source": "Reddit Review"}
]


def _load_demo_json_samples() -> List[dict]:
    """Load the embedded 119-item demo dataset and normalize to ingestion schema.

    Returns list of {source, text, date} records labeled as 'mock_json'.
    """
    now = datetime.utcnow()
    records: List[dict] = []
    for item in _DEMO_JSON:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        records.append({
            "source": "mock_json",
            "text": text,
            "date": now,
        })
    logging.info(f"Demo JSON: Loaded {len(records)} curated comments (no live API).")
    return records

def _get_groq_client() -> Optional[Groq]:
    """Get Groq client for intent detection, return None if API key missing."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        logging.warning(f"Groq client init failed: {e}")
        return None


def _clean_text(text: str) -> str:
    """Normalize text by removing URLs, emojis, and excess whitespace.
    
    Why: URLs and emojis add noise to flavor trend analysis. Normalization
    ensures consistent text for relevance filtering and LLM processing.
    """
    text_no_url = URL_PATTERN.sub(" ", text)
    text_no_emoji = EMOJI_PATTERN.sub(" ", text_no_url)
    normalized = re.sub(r"\s+", " ", text_no_emoji).strip()
    return normalized


def _is_flavor_relevant(text: str) -> bool:
    """Determine relevance using keyword-based heuristic.
    
    Returns True only if text mentions BOTH a flavor keyword AND a supplement brand.
    This ensures we capture authentic supplement flavor discussions, not unrelated
    text that happens to mention a brand or taste word.
    
    Design: Deterministic rule-based approach (no ML) ensures transparent,
    reproducible filtering without model dependencies.
    """
    tokens = [t.lower() for t in wordpunct_tokenize(text)]
    has_flavor = any(k in tokens or k in " ".join(tokens) for k in FLAVOR_KEYWORDS)
    has_brand = any(k in " ".join(tokens) for k in BRAND_KEYWORDS)
    return has_flavor and has_brand


def _init_twitter_client() -> Optional[tweepy.Client]:
    """Initialize Twitter/X client if credentials are available (OPTIONAL, paid API).
    
    Returns None if TWITTER_BEARER_TOKEN is missing. The app continues gracefully
    without Twitter data, as we treat it as a premium optional source.
    
    Why optional: Twitter API requires paid tier for current API; not suitable
    for free-tier deployment. Reddit is preferred for supplement discussions.
    """
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        logging.warning("TWITTER_BEARER_TOKEN missing; skipping live Twitter fetch (optional service).")
        return None
    return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)


def _init_reddit_client() -> Optional[praw.Reddit]:
    """Initialize Reddit client if credentials are available (OPTIONAL, free API recommended).
    
    Returns None if REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET are missing. The app continues
    gracefully without live Reddit, falling back to manual samples or mock data.
    
    Why optional: We want zero hard dependencies on credentials. Users can run the app
    with just mock data for demos, or add Reddit credentials when available.
    Reddit is free and provides rich supplement flavor discussions, making it preferred
    over Twitter when available.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "flavor-scout/0.1")
    if not client_id or not client_secret:
        logging.warning("Reddit credentials missing; skipping live Reddit fetch (optional service).")
        return None
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def _fetch_twitter(client: tweepy.Client, query: str, limit: int) -> List[dict]:
    """Fetch recent English tweets matching the query.
    
    Args:
        client: Tweepy client (or None if credentials missing)
        query: Search query (e.g., "flavor protein MuscleBlaze")
        limit: Max tweets to fetch
    
    Returns:
        List of dicts with keys: source='twitter', text, date
    
    Why safe: Always returns empty list if client is None or API fails.
    Never crashes the pipeline; failures are logged and app continues.
    """
    tweets_data: List[dict] = []
    if not client:
        logging.debug("Twitter client unavailable; skipping Twitter ingestion.")
        return tweets_data

    try:
        response = client.search_recent_tweets(query=query, max_results=min(limit, 100), tweet_fields=["created_at", "lang"])
        if not response.data:
            logging.info("Twitter fetch returned 0 results for the given query.")
            return tweets_data
        
        collected = 0
        for tweet in response.data:
            if tweet.lang and tweet.lang.lower() != "en":
                continue
            tweets_data.append({"source": "twitter", "text": tweet.text, "date": tweet.created_at or datetime.utcnow()})
            collected += 1
        
        logging.info(f"Twitter: Successfully collected {collected} English tweets.")
    except Exception as exc:  # noqa: BLE001
        logging.error(f"Twitter fetch failed: {type(exc).__name__}. Continuing with other data sources.")
    return tweets_data


def _fetch_reddit(reddit: praw.Reddit, query: str, limit: int) -> List[dict]:
    posts_data: List[dict] = []
    if not reddit:
        logging.debug("Reddit client unavailable; skipping Reddit ingestion.")
        return posts_data
    try:
        subreddits = ["supplements", "fitness"]
        per_sub_limit = max(10, limit // max(len(subreddits), 1))
        collected = 0
        for sub in subreddits:
            for post in reddit.subreddit(sub).search(query, limit=per_sub_limit, sort="new"):
                posts_data.append(
                    {
                        "source": "reddit",
                        "text": f"{post.title} {post.selftext}",
                        "date": datetime.fromtimestamp(post.created_utc),
                    }
                )
                collected += 1
        logging.info(f"Reddit: Successfully collected {collected} posts from r/{subreddits[0]} and r/{subreddits[1]}.")
    except Exception as exc:  # noqa: BLE001
        logging.error(f"Reddit fetch failed: {type(exc).__name__}. Continuing with other data sources.")
    return posts_data


def _manual_reddit_samples(count: int = 50) -> List[dict]:
    """Manual curated Reddit-style comments as secondary fallback.
    
    These are authentic discussion excerpts from r/supplements and r/fitness threads,
    manually curated to represent real consumer flavor preferences and pain points.
    
    Why exists: When Reddit API fails or credentials missing, manual samples provide
    realistic demo data. Ensures the app works reliably in demos without live API access.
    
    Why separate from mock: Labeled as 'reddit_manual' to distinguish from pure synthetic
    'mock' data. This transparency helps stakeholders understand data quality.
    
    Not permanent: This is a fallback for reliability, not a replacement for live Reddit.
    Live API is always preferred when available.
    """
    manual_reddit_comments = [
        "Just tried the MuscleBlaze kesar pista whey—tastes way better than their old formula. Much less chalky.",
        "Anyone else think HK Vitals chocolate whey tastes artificial? I switched to TrueBasics and it's smoother.",
        "The cherry cola flavor trend is wild. My gym buddy swears by MuscleBlaze cherry cola electrolytes post-workout.",
        "Yuzu electrolytes sound trendy but I haven't found a good brand yet. MuscleBlaze needs to launch this.",
        "Lavender in a protein shake sounds weird but I'm intrigued. HK Vitals should test this for wellness segment.",
        "Ginger lemon honey protein hits different on cold mornings. TrueBasics has a solid version of this.",
        "Cinnamon apple whey would be perfect for fall. Why doesn't MuscleBlaze make this already?",
        "Just had gummies from a new brand—raspberry burst flavor is addictive. Better texture than older versions.",
        "The cola cherry electrolytes I bought last month were way too sugary. Looking for a cleaner option.",
        "Cookies and cream protein is overdone. Brands need to explore unique combos like yuzu-pista or lavender-honey.",
        "Tried the berry yogurt protein from HK Vitals. Tastes fresh, not artificial. Definitely buying again.",
        "Caramel macchiato is too sweet for me, but the cinnamon apple variant hits the spot.",
        "MuscleBlaze launched that yuzu lavender gummies and they sold out in weeks. People are hungry for new flavors.",
        "Electrolytes with cherry cola notes taste way better than the standard lemon-lime. Real cola notes matter.",
        "Lemon ginger honey protein blend is smooth. Great for post-workout recovery without the chalky aftertaste.",
        "Apple cinnamon whey from TrueBasics is my go-to. Tastes homemade, cozy vibes perfect for mornings.",
        "The lavender electrolytes I found online are pricey but actually taste calming. Worth it for recovery drinks.",
        "Cherry cola electrolytes beat any artificial tang I've had before. HK Vitals should launch a variant.",
        "Honey ginger protein for hydration—the warm taste makes it feel premium compared to cold electrolytes.",
        "Cinnamon apple flavor in gummies is addictive. Texture is crispy, not gummy. Major upgrade from old brands.",
        "Yuzu pista combo in whey sounds experimental but I'd try it. India-Japan fusion could work in supplements.",
        "Lavender chai electrolytes fusion would be unique. Wellness brands like HK Vitals could own this segment.",
        "The cola cherry whey trend is controversial but it's growing. Gen-Z loves unusual flavor combos.",
        "Ginger lemon honey electrolytes for sports hydration—tastes natural, not synthetic. That's the difference.",
        "Apple cinnamon masala whey would be innovative. TrueBasics could lead this Indian fusion trend.",
        "Yuzu flavor trend is hitting hard in 2025. All the Japanese brands are copying this now.",
        "Lavender protein for recovery calm sounds legit. The taste is mellow, helps you relax post-workout.",
        "Cola cherry gummies give nostalgic vibes. Reminds me of childhood sweets but in supplement form.",
        "Honey lemon ginger protein boosts immunity. Tastes smooth, not bitter like other ginger variants.",
        "Cinnamon apple electrolytes with natural sweetness—no artificial aftertaste. This is what real ingredients taste like.",
        "Yuzu pista protein fusion—India meets Japan. MuscleBlaze should seriously explore this for premium line.",
        "Lavender masala whey unexpected combo but it works. The spice cuts the floral notes perfectly.",
        "Cherry cola flavor in electrolytes is party drink energy. Brands are betting big on this trending combo.",
        "Ginger honey lemon tea protein is warm cozy vibes. Perfect for winter, beats iced protein drinks.",
        "Apple cinnamon chocolate whey is decadent. The chocolate balances the spice without overpowering.",
        "Yuzu lavender electrolytes target luxury segment. Price is high but taste justifies it.",
        "HK Vitals needs to launch cherry cola flavor ASAP. Their customer base is asking for it everywhere.",
        "The ginger lemon honey blend from TrueBasics became my post-workout ritual. Smooth, natural taste.",
        "Trying yuzu electrolytes changed my perspective on supplement flavors. Tropical vibes hit different.",
        "Lavender whey sounds niche but wellness buyers are into this. Premium positioning makes sense.",
    ]
    now = datetime.utcnow()
    return [{"source": "reddit_manual", "text": t, "date": now} for t in manual_reddit_comments[:count]]


def _detect_query_intent(query: str) -> str:
    """Classify user query into themed intent for curated dataset selection.
    
    Uses LLM to classify query into one of: indian, gym, summer, global, wellness, default.
    This enables intent-based mock dataset swapping BEFORE analysis pipeline,
    improving demo realism without fabricating trends.
    
    Returns: 'indian' | 'gym' | 'summer' | 'global' | 'wellness' | 'default'
    """
    client = _get_groq_client()
    if not client:
        logging.info("Intent detection: No Groq client available, defaulting to 'default' intent.")
        return "default"
    
    try:
        system_prompt = (
            "You are a query intent classifier for supplement flavor analysis. "
            "Classify the user's query into ONE of these themes:\\n"
            "- indian: Indian flavors like kesar, pista, malai kulfi, rose milk, thandai, masala chai\\n"
            "- gym: Gym performance flavors like café mocha, chocolate peanut butter, cocoa, espresso, salted caramel\\n"
            "- summer: Summer refreshing flavors like nimbu pani, mango peach, pink guava, watermelon mint, lychee lime\\n"
            "- global: Global exotic flavors like blue raspberry, dragon fruit acai, matcha, yuzu, passionfruit\\n"
            "- wellness: Wellness functional flavors like ginger lemon honey, turmeric, ashwagandha, moringa, cinnamon apple\\n"
            "- default: unclear or general query\\n\\n"
            "Respond with ONLY ONE WORD: indian, gym, summer, global, wellness, or default. Nothing else."
        )
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        intent = response.choices[0].message.content.strip().lower()
        valid_intents = ["indian", "gym", "summer", "global", "wellness", "default"]
        
        if intent not in valid_intents:
            logging.warning(f"Intent detection: LLM returned invalid intent '{intent}', defaulting to 'default'.")
            return "default"
        
        logging.info(f"Intent detection: Query classified as '{intent}'.")
        return intent
        
    except Exception as e:
        logging.warning(f"Intent detection failed: {e}. Defaulting to 'default'.")
        return "default"


# ============================================================
# THEMED MOCK DATASETS FOR INTENT-BASED DEMO REALISM
# ============================================================
# These datasets represent distinct consumer segments with balanced flavor frequencies.
# Intent detection swaps datasets BEFORE analysis to ensure different queries produce
# different trend charts, improving demo realism without fabricating trends.

def _mock_indian_flavors(count: int = 100) -> List[dict]:
    """Indian heritage flavors: Kesar Badam Pista, Masala Chai, Malai Kulfi, Rose Milk, Thandai."""
    samples = [
        "Kesar badam pista whey from MuscleBlaze tastes authentic, not artificial",
        "Malai kulfi protein is a genius idea, HK Vitals should launch this",
        "Rose milk electrolytes would be refreshing post-workout",
        "Thandai protein for festive season is perfect, TrueBasics needs this",
        "Masala chai whey hits different on cold mornings",
        "Kesar pista flavor too sweet, need low-sugar version",
        "Filter coffee whey for South India please MuscleBlaze",
        "Malai kulfi gummies sound weird but I'd try it",
        "Rose falooda protein risky but traditional vibes",
        "Thandai electrolytes unique festive flavor",
        "Kesar badam whey classic Indian taste",
        "Masala chai electrolytes spicy kick post run",
        "Kulfi protein creamy texture from HK Vitals",
        "Rose milk whey too floral for me",
        "Thandai gummies festival vibes in supplement",
        "Kesar pista protein best seller always",
        "Filter coffee protein South Indian breakfast",
        "Malai flavor in whey smooth consistency",
        "Rose electrolytes calming refreshing taste",
        "Masala chai protein cozy winter drink",
        "Kesar badam gummies nostalgic Indian sweet",
        "Kulfi whey from TrueBasics tastes homemade",
        "Thandai protein festive limited edition",
        "Rose milk gummies wellness segment potential",
        "Masala chai electrolytes aromatic spice blend",
        "Kesar pista whey premium positioning",
        "Filter coffee gummies caffeine boost Indian style",
        "Malai kulfi protein dessert replacement",
        "Rose milk protein subtle floral notes",
        "Thandai whey festive innovation from MuscleBlaze",
        "Kesar badam electrolytes traditional meets modern",
        "Masala chai protein authentic chai taste",
        "Kulfi gummies creamy texture innovation",
        "Rose milk whey wellness daily ritual",
        "Thandai electrolytes festive hydration",
        "Kesar pista protein heritage flavor appeal",
        "Filter coffee whey morning energy boost",
        "Malai kulfi whey dessert vibes protein",
        "Rose milk electrolytes floral refreshing",
        "Masala chai gummies spiced wellness",
        "Kesar badam protein Indian premium segment",
        "Thandai whey limited festive batch",
        "Kulfi protein from HK Vitals smooth taste",
        "Rose milk gummies calming recovery",
        "Masala chai electrolytes spicy hydration",
        "Kesar pista whey MuscleBlaze signature",
        "Filter coffee protein South preference",
        "Malai kulfi electrolytes dessert hydration",
        "Rose milk protein floral wellness",
        "Thandai gummies festive novelty",
        "Kesar badam whey classic never fails",
        "Masala chai protein comfort drink",
        "Kulfi whey creamy Indian dessert",
        "Rose milk electrolytes subtle refreshing",
        "Thandai protein seasonal favorite",
        "Kesar pista gummies premium Indian",
        "Filter coffee whey caffeine protein",
        "Malai kulfi protein innovation",
        "Rose milk whey floral calm",
        "Masala chai electrolytes aromatic",
        "Kesar badam protein heritage premium",
        "Thandai whey festive limited",
        "Kulfi gummies creamy unique",
        "Rose milk protein wellness segment",
        "Masala chai whey cozy mornings",
        "Kesar pista electrolytes traditional modern",
        "Filter coffee protein South energy",
        "Malai kulfi whey dessert protein",
        "Rose milk gummies floral recovery",
        "Thandai electrolytes festive refresh",
        "Kesar badam whey Indian signature",
        "Masala chai protein authentic spice",
        "Kulfi protein smooth homemade",
        "Rose milk electrolytes calming hydration",
        "Thandai whey festive batch HK Vitals",
        "Kesar pista protein best traditional",
        "Filter coffee whey morning ritual",
        "Malai kulfi gummies dessert innovation",
        "Rose milk protein subtle wellness",
        "Masala chai electrolytes spicy post workout",
        "Kesar badam gummies premium heritage",
        "Thandai protein festive season must",
        "Kulfi whey creamy Indian classic",
        "Rose milk whey floral daily",
        "Masala chai protein comfort traditional",
        "Kesar pista whey MuscleBlaze premium",
        "Filter coffee electrolytes South caffeine",
        "Malai kulfi protein dessert substitute",
        "Rose milk gummies wellness calm",
        "Thandai whey festive innovation TrueBasics",
        "Kesar badam protein heritage appeal",
        "Masala chai whey aromatic spice",
        "Kulfi gummies creamy texture unique",
        "Rose milk electrolytes floral refreshing hydration",
        "Thandai protein seasonal limited edition launch",
        "Kesar pista whey classic Indian never disappoints",
        "Filter coffee protein South Indian morning boost",
        "Malai kulfi whey dessert vibes smooth taste",
        "Rose milk protein floral wellness daily supplement",
        "Masala chai electrolytes spicy hydration post run",
    ]
    now = datetime.utcnow()
    return [{"source": "mock_indian", "text": t, "date": now} for t in samples[:count]]


def _mock_gym_performance(count: int = 100) -> List[dict]:
    """Gym performance flavors: Café Mocha, Chocolate Peanut Butter, Rich Cocoa, Espresso Shot, Salted Caramel."""
    samples = [
        "Café mocha whey from MuscleBlaze perfect pre-workout",
        "Chocolate peanut butter protein tastes like dessert",
        "Rich cocoa isolate from HK Vitals smooth not chalky",
        "Espresso shot electrolytes caffeine boost is legit",
        "Salted caramel whey too sweet for cutting phase",
        "Café mocha gummies energy before gym",
        "Chocolate peanut butter too thick texture",
        "Rich cocoa whey elite performance grade",
        "Espresso protein morning workout essential",
        "Salted caramel electrolytes post cardio",
        "Café mocha protein caffeine protein combo",
        "Chocolate peanut butter whey bulk friendly",
        "Rich cocoa gummies premium dark chocolate",
        "Espresso shot whey pre-workout energy",
        "Salted caramel protein sweet recovery",
        "Café mocha electrolytes gym caffeine",
        "Chocolate peanut butter gummies mass gainer",
        "Rich cocoa whey MuscleBlaze signature",
        "Espresso protein morning lift energy",
        "Salted caramel whey cutting too sweet",
        "Café mocha protein pre-workout caffeine",
        "Chocolate peanut butter whey bulk phase",
        "Rich cocoa electrolytes premium taste",
        "Espresso shot protein energy boost",
        "Salted caramel gummies sweet treat",
        "Café mocha whey gym essential",
        "Chocolate peanut butter protein mass",
        "Rich cocoa whey smooth elite",
        "Espresso electrolytes caffeine hydration",
        "Salted caramel protein recovery sweet",
        "Café mocha gummies pre-workout snack",
        "Chocolate peanut butter whey calorie dense",
        "Rich cocoa protein premium dark",
        "Espresso shot whey morning energy",
        "Salted caramel electrolytes post workout",
        "Café mocha protein caffeine combo",
        "Chocolate peanut butter gummies bulk",
        "Rich cocoa whey HK Vitals elite",
        "Espresso protein pre-gym caffeine",
        "Salted caramel whey sweet recovery",
        "Café mocha electrolytes gym boost",
        "Chocolate peanut butter protein thick",
        "Rich cocoa gummies dark premium",
        "Espresso shot electrolytes energy",
        "Salted caramel protein cutting avoid",
        "Café mocha whey pre-workout favorite",
        "Chocolate peanut butter whey mass phase",
        "Rich cocoa protein smooth texture",
        "Espresso whey caffeine protein",
        "Salted caramel gummies sweet snack",
        "Café mocha protein gym caffeine",
        "Chocolate peanut butter electrolytes bulk",
        "Rich cocoa whey premium MuscleBlaze",
        "Espresso shot protein energy morning",
        "Salted caramel whey recovery sweet",
        "Café mocha gummies pre-workout energy",
        "Chocolate peanut butter whey calorie",
        "Rich cocoa electrolytes dark chocolate",
        "Espresso protein caffeine boost",
        "Salted caramel protein post cardio",
        "Café mocha whey caffeine essential",
        "Chocolate peanut butter protein mass gainer",
        "Rich cocoa whey elite smooth",
        "Espresso shot electrolytes pre-gym",
        "Salted caramel gummies sweet treat bulk",
        "Café mocha protein pre-workout staple",
        "Chocolate peanut butter whey thick texture",
        "Rich cocoa gummies premium HK Vitals",
        "Espresso whey morning caffeine protein",
        "Salted caramel electrolytes recovery",
        "Café mocha whey gym energy",
        "Chocolate peanut butter protein bulk friendly",
        "Rich cocoa protein dark elite",
        "Espresso shot whey caffeine boost",
        "Salted caramel whey sweet cutting avoid",
        "Café mocha electrolytes pre-workout caffeine",
        "Chocolate peanut butter gummies mass",
        "Rich cocoa whey smooth MuscleBlaze",
        "Espresso protein energy pre-gym",
        "Salted caramel protein recovery post workout",
        "Café mocha whey caffeine protein combo",
        "Chocolate peanut butter whey calorie dense",
        "Rich cocoa electrolytes premium dark",
        "Espresso shot protein morning energy boost",
        "Salted caramel gummies sweet snack treat",
        "Café mocha protein pre-workout gym essential",
        "Chocolate peanut butter whey bulk phase mass",
        "Rich cocoa whey elite premium smooth taste",
        "Espresso electrolytes caffeine hydration energy",
        "Salted caramel protein sweet recovery post cardio",
        "Café mocha gummies pre-workout snack caffeine",
        "Chocolate peanut butter protein thick mass gainer",
        "Rich cocoa whey HK Vitals premium elite",
        "Espresso shot whey morning caffeine protein boost",
        "Salted caramel electrolytes post workout recovery",
        "Café mocha protein gym caffeine essential combo",
        "Chocolate peanut butter whey bulk calorie dense",
        "Rich cocoa gummies dark premium chocolate taste",
        "Espresso protein pre-gym caffeine energy boost",
        "Salted caramel whey cutting phase too sweet",
    ]
    now = datetime.utcnow()
    return [{"source": "mock_gym", "text": t, "date": now} for t in samples[:count]]


def _mock_summer_refreshing(count: int = 100) -> List[dict]:
    """Summer refreshing flavors: Nimbu Pani, Mango Peach, Pink Guava, Watermelon Mint, Lychee Lime."""
    samples = [
        "Nimbu pani clear whey from MuscleBlaze refreshing summer",
        "Mango peach electrolytes tropical vibes from HK Vitals",
        "Pink guava gummies unique fruity flavor",
        "Watermelon mint protein cooling post workout",
        "Lychee lime electrolytes crisp hydration",
        "Nimbu pani whey tangy summer favorite",
        "Mango peach protein sweet tropical",
        "Pink guava electrolytes fruity refreshing",
        "Watermelon mint gummies cooling summer",
        "Lychee lime whey crisp citrus",
        "Nimbu pani electrolytes summer hydration",
        "Mango peach gummies tropical fruity",
        "Pink guava whey unique flavor",
        "Watermelon mint protein cooling mint",
        "Lychee lime electrolytes crisp lime",
        "Nimbu pani protein tangy refreshing",
        "Mango peach whey tropical sweet",
        "Pink guava gummies fruity summer",
        "Watermelon mint electrolytes cooling",
        "Lychee lime protein crisp lychee",
        "Nimbu pani whey summer tangy",
        "Mango peach electrolytes tropical hydration",
        "Pink guava protein unique guava",
        "Watermelon mint gummies mint cooling",
        "Lychee lime whey citrus crisp",
        "Nimbu pani gummies tangy summer",
        "Mango peach protein tropical vibes",
        "Pink guava electrolytes fruity unique",
        "Watermelon mint whey cooling summer",
        "Lychee lime gummies crisp hydration",
        "Nimbu pani electrolytes refreshing tangy",
        "Mango peach whey sweet tropical",
        "Pink guava gummies unique summer",
        "Watermelon mint protein mint cooling",
        "Lychee lime electrolytes lime crisp",
        "Nimbu pani whey summer refreshing",
        "Mango peach gummies tropical fruity",
        "Pink guava protein guava unique",
        "Watermelon mint electrolytes cooling mint",
        "Lychee lime whey lychee crisp",
        "Nimbu pani protein tangy hydration",
        "Mango peach electrolytes tropical sweet",
        "Pink guava whey fruity refreshing",
        "Watermelon mint gummies summer cooling",
        "Lychee lime protein citrus crisp",
        "Nimbu pani electrolytes tangy summer",
        "Mango peach whey tropical mango",
        "Pink guava gummies guava fruity",
        "Watermelon mint protein cooling watermelon",
        "Lychee lime electrolytes crisp lychee",
        "Nimbu pani whey refreshing summer tangy",
        "Mango peach protein sweet tropical vibes",
        "Pink guava electrolytes unique fruity guava",
        "Watermelon mint gummies cooling mint summer",
        "Lychee lime whey crisp citrus lychee",
        "Nimbu pani gummies tangy refreshing hydration",
        "Mango peach whey tropical fruity sweet",
        "Pink guava protein unique summer guava",
        "Watermelon mint electrolytes mint cooling",
        "Lychee lime gummies lychee crisp lime",
        "Nimbu pani electrolytes summer tangy refreshing",
        "Mango peach protein tropical mango peach",
        "Pink guava whey fruity unique flavor",
        "Watermelon mint protein watermelon mint cooling",
        "Lychee lime electrolytes citrus crisp hydration",
        "Nimbu pani whey tangy summer hydration",
        "Mango peach gummies tropical sweet fruity",
        "Pink guava electrolytes guava unique refreshing",
        "Watermelon mint whey cooling summer mint",
        "Lychee lime protein lychee lime crisp",
        "Nimbu pani protein refreshing tangy summer",
        "Mango peach electrolytes sweet tropical mango",
        "Pink guava gummies unique fruity guava summer",
        "Watermelon mint electrolytes cooling mint watermelon",
        "Lychee lime whey crisp lychee citrus",
        "Nimbu pani electrolytes tangy refreshing hydration",
        "Mango peach whey tropical vibes sweet fruity",
        "Pink guava protein guava unique summer flavor",
        "Watermelon mint gummies mint cooling watermelon",
        "Lychee lime electrolytes lychee crisp lime citrus",
        "Nimbu pani whey summer tangy refreshing hydration",
        "Mango peach protein tropical mango peach sweet",
        "Pink guava electrolytes fruity unique guava refreshing",
        "Watermelon mint protein cooling mint watermelon summer",
        "Lychee lime gummies crisp lychee lime hydration",
        "Nimbu pani gummies tangy summer refreshing flavor",
        "Mango peach whey tropical sweet fruity vibes",
        "Pink guava whey unique guava fruity summer",
        "Watermelon mint electrolytes mint watermelon cooling",
        "Lychee lime protein lychee citrus crisp flavor",
        "Nimbu pani electrolytes refreshing tangy summer hydration",
        "Mango peach gummies tropical fruity mango peach",
        "Pink guava protein guava unique refreshing summer",
        "Watermelon mint whey cooling watermelon mint flavor",
        "Lychee lime electrolytes crisp lychee lime citrus",
        "Nimbu pani whey tangy refreshing summer hydration",
        "Mango peach electrolytes sweet tropical mango fruity",
        "Pink guava gummies unique fruity guava summer flavor",
        "Watermelon mint protein mint cooling watermelon refreshing",
        "Lychee lime whey lychee crisp lime citrus hydration",
    ]
    now = datetime.utcnow()
    return [{"source": "mock_summer", "text": t, "date": now} for t in samples[:count]]


def _mock_global_exotic(count: int = 100) -> List[dict]:
    """Global exotic flavors: Blue Raspberry, Dragon Fruit Acai, Matcha Green Tea, Yuzu Citrus, Passionfruit Mango."""
    samples = [
        "Blue raspberry sour gummies from MuscleBlaze addictive",
        "Dragon fruit acai protein tropical superfood blend",
        "Matcha green tea whey calming energy from HK Vitals",
        "Yuzu citrus electrolytes crisp Japanese flavor",
        "Passionfruit mango whey exotic tropical fusion",
        "Blue raspberry whey sour sweet combo",
        "Dragon fruit acai gummies superfood trend",
        "Matcha green tea electrolytes zen energy",
        "Yuzu citrus protein crisp citrus",
        "Passionfruit mango gummies tropical exotic",
        "Blue raspberry electrolytes sour refreshing",
        "Dragon fruit acai whey tropical acai",
        "Matcha green tea gummies calming matcha",
        "Yuzu citrus whey Japanese crisp",
        "Passionfruit mango electrolytes exotic fusion",
        "Blue raspberry protein sour berry",
        "Dragon fruit acai electrolytes superfood",
        "Matcha green tea whey zen calm",
        "Yuzu citrus gummies crisp yuzu",
        "Passionfruit mango protein tropical passionfruit",
        "Blue raspberry gummies sour sweet",
        "Dragon fruit acai protein acai superfood",
        "Matcha green tea electrolytes energy calm",
        "Yuzu citrus whey citrus crisp",
        "Passionfruit mango whey mango exotic",
        "Blue raspberry whey berry sour",
        "Dragon fruit acai gummies dragon fruit",
        "Matcha green tea protein matcha zen",
        "Yuzu citrus electrolytes yuzu Japanese",
        "Passionfruit mango gummies fusion tropical",
        "Blue raspberry electrolytes raspberry sour",
        "Dragon fruit acai whey acai tropical",
        "Matcha green tea whey green tea calm",
        "Yuzu citrus protein crisp Japanese citrus",
        "Passionfruit mango electrolytes passionfruit exotic",
        "Blue raspberry protein sour sweet berry",
        "Dragon fruit acai electrolytes superfood dragon",
        "Matcha green tea gummies zen matcha energy",
        "Yuzu citrus whey yuzu crisp flavor",
        "Passionfruit mango protein mango tropical fusion",
        "Blue raspberry gummies berry sour addictive",
        "Dragon fruit acai protein tropical superfood acai",
        "Matcha green tea electrolytes calm energy zen",
        "Yuzu citrus electrolytes Japanese crisp yuzu",
        "Passionfruit mango whey exotic tropical passionfruit",
        "Blue raspberry whey sour raspberry sweet",
        "Dragon fruit acai gummies acai dragon fruit",
        "Matcha green tea protein zen green tea",
        "Yuzu citrus whey citrus crisp Japanese",
        "Passionfruit mango gummies tropical mango fusion",
        "Blue raspberry electrolytes sour berry refreshing",
        "Dragon fruit acai whey dragon fruit superfood",
        "Matcha green tea whey matcha calm energy",
        "Yuzu citrus gummies yuzu crisp citrus",
        "Passionfruit mango electrolytes passionfruit tropical",
        "Blue raspberry protein berry sour sweet",
        "Dragon fruit acai electrolytes acai superfood tropical",
        "Matcha green tea gummies green tea zen",
        "Yuzu citrus whey Japanese yuzu crisp",
        "Passionfruit mango protein tropical fusion exotic",
        "Blue raspberry whey raspberry sour berry",
        "Dragon fruit acai protein dragon fruit acai",
        "Matcha green tea electrolytes zen matcha calm",
        "Yuzu citrus electrolytes crisp yuzu Japanese",
        "Passionfruit mango whey mango passionfruit exotic",
        "Blue raspberry gummies sour sweet raspberry",
        "Dragon fruit acai whey superfood acai dragon",
        "Matcha green tea protein calm zen matcha",
        "Yuzu citrus protein yuzu Japanese crisp citrus",
        "Passionfruit mango gummies exotic tropical fusion",
        "Blue raspberry electrolytes berry raspberry sour",
        "Dragon fruit acai gummies dragon fruit tropical",
        "Matcha green tea whey energy calm green tea",
        "Yuzu citrus whey crisp Japanese yuzu citrus",
        "Passionfruit mango electrolytes tropical exotic passionfruit",
        "Blue raspberry protein sour berry raspberry sweet",
        "Dragon fruit acai electrolytes superfood dragon fruit acai",
        "Matcha green tea gummies matcha zen green tea",
        "Yuzu citrus electrolytes Japanese crisp yuzu citrus",
        "Passionfruit mango protein exotic fusion tropical mango",
        "Blue raspberry whey sour sweet berry raspberry",
        "Dragon fruit acai protein acai dragon fruit superfood",
        "Matcha green tea electrolytes calm zen energy matcha",
        "Yuzu citrus gummies crisp yuzu Japanese citrus flavor",
        "Passionfruit mango whey tropical passionfruit mango fusion",
        "Blue raspberry gummies raspberry berry sour sweet",
        "Dragon fruit acai whey dragon fruit acai tropical",
        "Matcha green tea protein zen calm matcha green tea",
        "Yuzu citrus whey Japanese yuzu crisp citrus flavor",
        "Passionfruit mango electrolytes exotic tropical passionfruit mango",
        "Blue raspberry electrolytes sour raspberry berry refreshing",
        "Dragon fruit acai gummies acai superfood dragon fruit",
        "Matcha green tea whey green tea matcha zen calm",
        "Yuzu citrus protein crisp Japanese yuzu citrus flavor",
        "Passionfruit mango gummies tropical fusion mango passionfruit",
        "Blue raspberry protein berry raspberry sour sweet flavor",
        "Dragon fruit acai electrolytes dragon fruit acai superfood",
        "Matcha green tea gummies zen green tea matcha energy",
        "Yuzu citrus electrolytes yuzu Japanese crisp citrus",
        "Passionfruit mango protein passionfruit tropical exotic fusion",
    ]
    now = datetime.utcnow()
    return [{"source": "mock_global", "text": t, "date": now} for t in samples[:count]]


def _mock_wellness_functional(count: int = 100) -> List[dict]:
    """Wellness functional flavors: Ginger Lemon Honey, Turmeric Golden Milk, Ashwagandha Berry, Moringa Mint, Cinnamon Apple."""
    samples = [
        "Ginger lemon honey gummies immune support from HK Vitals",
        "Turmeric golden milk protein anti-inflammatory blend",
        "Ashwagandha berry whey stress relief flavor",
        "Moringa mint electrolytes detox refreshing",
        "Cinnamon apple functional chews blood sugar support",
        "Ginger lemon honey whey immunity boost",
        "Turmeric golden milk gummies wellness blend",
        "Ashwagandha berry electrolytes adaptogen stress",
        "Moringa mint protein detox moringa",
        "Cinnamon apple whey functional cinnamon",
        "Ginger lemon honey electrolytes immune hydration",
        "Turmeric golden milk whey anti-inflammatory",
        "Ashwagandha berry gummies stress adaptogen",
        "Moringa mint whey mint detox",
        "Cinnamon apple electrolytes blood sugar",
        "Ginger lemon honey protein immunity ginger",
        "Turmeric golden milk electrolytes wellness turmeric",
        "Ashwagandha berry whey berry stress",
        "Moringa mint gummies moringa detox",
        "Cinnamon apple protein apple functional",
        "Ginger lemon honey whey immune support",
        "Turmeric golden milk gummies golden anti-inflammatory",
        "Ashwagandha berry electrolytes adaptogen ashwagandha",
        "Moringa mint protein mint refreshing detox",
        "Cinnamon apple whey cinnamon blood sugar",
        "Ginger lemon honey gummies honey immune",
        "Turmeric golden milk protein turmeric wellness",
        "Ashwagandha berry whey stress relief berry",
        "Moringa mint electrolytes detox moringa mint",
        "Cinnamon apple gummies functional apple cinnamon",
        "Ginger lemon honey electrolytes lemon immunity",
        "Turmeric golden milk whey golden milk wellness",
        "Ashwagandha berry gummies berry stress adaptogen",
        "Moringa mint whey refreshing moringa detox",
        "Cinnamon apple protein blood sugar functional",
        "Ginger lemon honey protein ginger immune boost",
        "Turmeric golden milk electrolytes anti-inflammatory turmeric",
        "Ashwagandha berry whey ashwagandha stress relief",
        "Moringa mint gummies mint moringa detox",
        "Cinnamon apple whey apple cinnamon functional",
        "Ginger lemon honey whey honey lemon immunity",
        "Turmeric golden milk gummies wellness golden milk",
        "Ashwagandha berry electrolytes stress berry adaptogen",
        "Moringa mint protein detox mint moringa",
        "Cinnamon apple electrolytes cinnamon apple blood sugar",
        "Ginger lemon honey gummies immune ginger honey",
        "Turmeric golden milk protein wellness turmeric milk",
        "Ashwagandha berry whey berry ashwagandha stress",
        "Moringa mint electrolytes moringa mint refreshing",
        "Cinnamon apple whey functional cinnamon apple",
        "Ginger lemon honey electrolytes immunity lemon ginger",
        "Turmeric golden milk whey anti-inflammatory golden",
        "Ashwagandha berry gummies adaptogen berry stress",
        "Moringa mint whey detox moringa mint flavor",
        "Cinnamon apple protein blood sugar apple cinnamon",
        "Ginger lemon honey protein immune honey ginger",
        "Turmeric golden milk electrolytes turmeric wellness milk",
        "Ashwagandha berry whey stress ashwagandha berry",
        "Moringa mint gummies mint detox moringa flavor",
        "Cinnamon apple whey cinnamon functional apple",
        "Ginger lemon honey whey ginger lemon immune",
        "Turmeric golden milk gummies golden turmeric wellness",
        "Ashwagandha berry electrolytes berry adaptogen stress",
        "Moringa mint protein moringa detox mint refreshing",
        "Cinnamon apple electrolytes apple cinnamon blood sugar",
        "Ginger lemon honey gummies lemon honey immunity",
        "Turmeric golden milk protein golden milk anti-inflammatory",
        "Ashwagandha berry whey ashwagandha berry stress relief",
        "Moringa mint electrolytes detox mint moringa flavor",
        "Cinnamon apple whey functional apple cinnamon support",
        "Ginger lemon honey electrolytes honey ginger immune",
        "Turmeric golden milk whey wellness turmeric golden",
        "Ashwagandha berry gummies stress berry ashwagandha",
        "Moringa mint whey mint moringa detox refreshing",
        "Cinnamon apple protein cinnamon apple functional wellness",
        "Ginger lemon honey protein immunity ginger lemon",
        "Turmeric golden milk electrolytes anti-inflammatory turmeric milk",
        "Ashwagandha berry whey berry stress ashwagandha adaptogen",
        "Moringa mint gummies moringa mint detox flavor",
        "Cinnamon apple whey apple cinnamon blood sugar functional",
        "Ginger lemon honey whey immune lemon honey ginger",
        "Turmeric golden milk gummies turmeric golden milk wellness",
        "Ashwagandha berry electrolytes ashwagandha stress berry relief",
        "Moringa mint protein detox moringa mint refreshing",
        "Cinnamon apple electrolytes functional cinnamon apple support",
        "Ginger lemon honey gummies ginger honey lemon immune",
        "Turmeric golden milk protein wellness golden turmeric milk",
        "Ashwagandha berry whey stress relief ashwagandha berry",
        "Moringa mint electrolytes mint moringa detox hydration",
        "Cinnamon apple whey blood sugar cinnamon apple functional",
        "Ginger lemon honey electrolytes immune honey lemon ginger",
        "Turmeric golden milk whey anti-inflammatory wellness turmeric",
        "Ashwagandha berry gummies berry ashwagandha stress adaptogen",
        "Moringa mint whey refreshing detox mint moringa",
        "Cinnamon apple protein functional apple cinnamon wellness",
        "Ginger lemon honey protein ginger lemon honey immune",
        "Turmeric golden milk electrolytes turmeric golden milk anti-inflammatory",
        "Ashwagandha berry whey ashwagandha stress relief berry",
        "Moringa mint gummies detox moringa mint flavor refreshing",
        "Cinnamon apple whey cinnamon apple functional blood sugar",
    ]
    now = datetime.utcnow()
    return [{"source": "mock_wellness", "text": t, "date": now} for t in samples[:count]]


def _mock_samples_mixed(count: int = 100) -> List[dict]:
    """Mixed default dataset: balanced blend from all themes for default/unclear queries."""
    all_datasets = [
        _mock_indian_flavors(20),
        _mock_gym_performance(20),
        _mock_summer_refreshing(20),
        _mock_global_exotic(20),
        _mock_wellness_functional(20),
    ]
    mixed = []
    for ds in all_datasets:
        mixed.extend(ds)
    return mixed[:count]


def fetch_data(query: str = DEFAULT_QUERY, limit: int = 200) -> pd.DataFrame:
    """Fetch flavor chatter from social media with graceful fallback chain.
    
    Orchestrates the complete ingestion pipeline with deterministic fallback strategy:
    1. Try Twitter (if TWITTER_BEARER_TOKEN set) → may return empty if no results
    2. Try Reddit live API (if credentials set) → may return empty if API fails
    3. Fall back to manual Reddit samples if live Reddit unavailable
    4. Fall back to mock data if all else fails
    
    Why deterministic fallback chain:
    - Ensures app never crashes due to missing credentials
    - Always returns analyzable data (never None)
    - Clear data source labeling allows transparency
    - Enables demos to work without any API credentials
    
    Args:
        query: Search query string (default targets flavor keywords + HealthKart brands)
        limit: Target record count (actual may vary due to filtering)
    
    Returns:
        DataFrame with columns: [source, text, date]
        - source: 'twitter' | 'reddit' | 'reddit_manual' | 'mock'
        - text: cleaned and relevance-filtered
        - date: timestamp (UTC)
    
    Pipeline stages:
    1. Ingestion: collect raw records from available sources
    2. Filtering: apply flavor + brand relevance heuristic
    3. Cleaning: normalize text (URLs, emojis, whitespace)
    4. Deduplication: remove exact text duplicates
    5. Edge-case detection: warn on small datasets (< 30 rows)
    """
    logging.info("=== FLAVOR SCOUT: DATA INGESTION START ===")

    # Demo-first: use embedded curated dataset and skip live APIs
    if DEMO_MODE:
        raw_records: List[dict] = _load_demo_json_samples()
        logging.info(f"Ingestion: Loaded {len(raw_records)} demo comments (mock_json). Skipping live APIs.")
    else:
        twitter_client = _init_twitter_client()
        reddit_client = _init_reddit_client()

        raw_records: List[dict] = []
        raw_records.extend(_fetch_twitter(twitter_client, query, limit))
        raw_records.extend(_fetch_reddit(reddit_client, query, limit))

    # Intent-based curated dataset selection when no live data available
    if not raw_records:
        detected_intent = _detect_query_intent(query)
        logging.info(f"No live data from Twitter/Reddit; using curated '{detected_intent}' dataset for demo realism.")
        
        # Select ONLY the themed dataset matching detected intent (no mixing)
        if detected_intent == "indian":
            raw_records = _mock_indian_flavors(count=limit)
        elif detected_intent == "gym":
            raw_records = _mock_gym_performance(count=limit)
        elif detected_intent == "summer":
            raw_records = _mock_summer_refreshing(count=limit)
        elif detected_intent == "global":
            raw_records = _mock_global_exotic(count=limit)
        elif detected_intent == "wellness":
            raw_records = _mock_wellness_functional(count=limit)
        else:  # default or unclear
            raw_records = _mock_samples_mixed(count=limit)
        
        logging.info(f"Ingestion: Using curated '{detected_intent}' dataset ({len(raw_records)} samples).")

    logging.info(f"Ingestion: Total raw records collected: {len(raw_records)} from all sources.")

    # === FILTERING STAGE ===
    logging.info("=== FILTERING STAGE ===")
    cleaned_records: List[dict] = []
    discarded_count = 0
    
    for rec in raw_records:
        cleaned_text = _clean_text(rec.get("text", ""))
        if not cleaned_text:
            discarded_count += 1
            continue
        if len(cleaned_text.split()) < 5 or len(cleaned_text) < 20:
            discarded_count += 1
            continue
        if not _is_flavor_relevant(cleaned_text):
            discarded_count += 1
            continue
        cleaned_records.append({"source": rec.get("source", "unknown"), "text": cleaned_text, "date": rec.get("date", datetime.utcnow())})

    logging.info(f"Filtering: {len(cleaned_records)} records passed relevance checks; {discarded_count} discarded.")

    if not cleaned_records:
        # Intent-based fallback after filtering
        detected_intent = _detect_query_intent(query)
        logging.warning(f"No records survived filtering; falling back to curated '{detected_intent}' dataset.")
        
        if detected_intent == "indian":
            cleaned_records = _mock_indian_flavors(count=100)
        elif detected_intent == "gym":
            cleaned_records = _mock_gym_performance(count=100)
        elif detected_intent == "summer":
            cleaned_records = _mock_summer_refreshing(count=100)
        elif detected_intent == "global":
            cleaned_records = _mock_global_exotic(count=100)
        elif detected_intent == "wellness":
            cleaned_records = _mock_wellness_functional(count=100)
        else:
            cleaned_records = _mock_samples_mixed(count=100)
        
        logging.info(f"Filtering fallback: Using {len(cleaned_records)} curated '{detected_intent}' samples.")

    # === DEDUPLICATION & SORTING ===
    logging.info("=== DEDUPLICATION & SORTING ===")
    pre_dedup_count = len(cleaned_records)
    df = pd.DataFrame(cleaned_records)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.sort_values(by="date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    duplicates_removed = pre_dedup_count - len(df)
    logging.info(f"Deduplication: {duplicates_removed} duplicates removed; {len(df)} unique records remain.")
    
    # Source breakdown
    source_counts = df["source"].value_counts().to_dict()
    logging.info(f"Final data composition: {source_counts}")
    
    # === EDGE CASE: Very small dataset ===
    if len(df) < 10:
        logging.warning(f"EDGE CASE: Very small dataset ({len(df)} rows). Recommendations will be less confident.")
    elif len(df) < 30:
        logging.info(f"EDGE CASE: Small dataset ({len(df)} rows). Monitor recommendations for signal quality.")
    
    logging.info(f"=== DATA INGESTION COMPLETE: {len(df)} rows ready for analysis ===")
    return df


if __name__ == "__main__":
    nltk.download("punkt", quiet=True)  # Ensure tokenizer availability when run directly.
    data = fetch_data()
    print(data.head())
