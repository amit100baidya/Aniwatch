"""
nlp_chat.py  —  Conversational NLP Anime Recommender
=====================================================
Fixes applied vs original:
  1. Duplicate 'scary' key in MOOD_MAP removed
  2. type hints str|None / list[str] replaced with Optional[str]/List[str] for Python 3.9 compat
  3. Episodes column coerced to numeric before comparison (avoids TypeError on string values)
  4. genre_list safe-guard uses same _safe_genre_list logic as app.py
  5. _semantic_search: 'text_features' column existence checked before use
"""

import re, random, os
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(_pkg, quiet=True)

_STOP = set(stopwords.words("english"))
_LEM  = WordNetLemmatizer()


# ── Popular alias map ─────────────────────────────────────────────────────────
ALIAS_MAP = {
    "attack on titan":                "Shingeki no Kyojin",
    "aot":                            "Shingeki no Kyojin",
    "sword art online":               "Sword Art Online",
    "sao":                            "Sword Art Online",
    "my hero academia":               "Boku no Hero Academia",
    "mha":                            "Boku no Hero Academia",
    "fullmetal alchemist brotherhood":"Fullmetal Alchemist: Brotherhood",
    "fmab":                           "Fullmetal Alchemist: Brotherhood",
    "fma brotherhood":                "Fullmetal Alchemist: Brotherhood",
    "demon slayer":                   "Kimetsu no Yaiba",
    "jujutsu kaisen":                 "Jujutsu Kaisen",
    "jjk":                            "Jujutsu Kaisen",
    "hunter x hunter":                "Hunter x Hunter (2011)",
    "hxh":                            "Hunter x Hunter (2011)",
    "tokyo ghoul":                    "Tokyo Ghoul",
    "violet evergarden":              "Violet Evergarden",
    "re zero":                        "Re:Zero kara Hajimeru Isekai Seikatsu",
    "re:zero":                        "Re:Zero kara Hajimeru Isekai Seikatsu",
    "one punch man":                  "One Punch Man",
    "opm":                            "One Punch Man",
    "dragon ball z":                  "Dragon Ball Z",
    "dbz":                            "Dragon Ball Z",
    "one piece":                      "One Piece",
    "naruto":                         "Naruto",
    "bleach":                         "Bleach",
    "death note":                     "Death Note",
    "steins gate":                    "Steins;Gate",
    "steins;gate":                    "Steins;Gate",
    "cowboy bebop":                   "Cowboy Bebop",
    "neon genesis evangelion":        "Neon Genesis Evangelion",
    "nge":                            "Neon Genesis Evangelion",
    "evangelion":                     "Neon Genesis Evangelion",
    "made in abyss":                  "Made in Abyss",
    "vinland saga":                   "Vinland Saga",
    "mob psycho":                     "Mob Psycho 100",
    "mob psycho 100":                 "Mob Psycho 100",
    "spy x family":                   "Spy x Family",
    "spy family":                     "Spy x Family",
    "chainsaw man":                   "Chainsaw Man",
    "black clover":                   "Black Clover",
    "dr stone":                       "Dr. Stone",
    "dr. stone":                      "Dr. Stone",
    "haikyuu":                        "Haikyuu!!",
    "haikyu":                         "Haikyuu!!",
    "your lie in april":              "Shigatsu wa Kimi no Uso",
    "a silent voice":                 "Koe no Katachi",
    "your name":                      "Kimi no Na wa.",
    "spirited away":                  "Sen to Chihiro no Kamikakushi",
    "princess mononoke":              "Mononoke Hime",
    "howls moving castle":            "Howl no Ugoku Shiro",
    "grave of the fireflies":         "Hotaru no Haka",
    "akira":                          "Akira",
    "ghost in the shell":             "Ghost in the Shell",
    "serial experiments lain":        "Serial Experiments Lain",
    "berserk":                        "Berserk",
    "fruits basket":                  "Fruits Basket",
    "clannad":                        "Clannad",
    "anohana":                        "Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai.",
}

# ── Genre lexicon ─────────────────────────────────────────────────────────────
ALL_GENRES = [
    "Action", "Adventure", "Avant Garde", "Award Winning", "Boys Love",
    "Comedy", "Drama", "Fantasy", "Girls Love", "Gourmet", "Horror",
    "Mystery", "Romance", "Sci-Fi", "Slice of Life", "Sports",
    "Supernatural", "Suspense", "Ecchi", "Josei", "Mahou Shoujo",
    "Mecha", "Military", "Music", "Parody", "Psychological", "Racing",
    "School", "Seinen", "Shoujo", "Shounen", "Space", "Vampire",
    "Magic", "Thriller", "Isekai", "Historical", "Martial Arts",
    "Samurai", "Demons", "Super Power",
]

# ── Mood → genre mapping (FIX 1: removed duplicate 'scary' key) ──────────────
MOOD_MAP = {
    "happy":            ["Comedy", "Slice of Life"],
    "funny":            ["Comedy", "Parody"],
    "hilarious":        ["Comedy", "Parody"],
    "laugh":            ["Comedy"],
    "excited":          ["Action", "Sports", "Adventure"],
    "scary":            ["Horror", "Thriller"],        # merged both scary entries
    "horror":           ["Horror"],
    "romantic":         ["Romance"],
    "love":             ["Romance", "Drama"],
    "sad":              ["Drama", "Slice of Life"],
    "emotional":        ["Drama", "Romance"],
    "cry":              ["Drama", "Romance"],
    "tearjerker":       ["Drama"],
    "depressing":       ["Drama", "Psychological"],
    "dark":             ["Horror", "Psychological", "Drama", "Seinen"],
    "gritty":           ["Seinen", "Drama", "Action"],
    "chill":            ["Slice of Life", "Comedy"],
    "relaxing":         ["Slice of Life", "Comedy", "Music"],
    "peaceful":         ["Slice of Life"],
    "iyashikei":        ["Slice of Life"],
    "epic":             ["Action", "Fantasy", "Adventure"],
    "magical":          ["Fantasy", "Magic", "Mahou Shoujo"],
    "adventurous":      ["Adventure", "Fantasy", "Action"],
    "cool":             ["Action", "Sci-Fi"],
    "cute":             ["Slice of Life", "Comedy", "Romance"],
    "deep":             ["Psychological", "Drama"],
    "mysterious":       ["Mystery", "Thriller", "Psychological"],
    "mystery":          ["Mystery"],
    "thriller":         ["Thriller", "Suspense"],
    "suspense":         ["Suspense", "Thriller"],
    "psychological":    ["Psychological", "Mystery"],
    "mind-bending":     ["Psychological", "Sci-Fi"],
    "mind blowing":     ["Psychological", "Sci-Fi", "Mystery"],
    "philosophical":    ["Psychological", "Drama"],
    "nostalgic":        ["Slice of Life", "Drama", "School"],
    "school":           ["School", "Comedy", "Romance"],
    "sports":           ["Sports"],
    "mecha":            ["Mecha", "Sci-Fi"],
    "robot":            ["Mecha", "Sci-Fi"],
    "isekai":           ["Isekai", "Fantasy"],
    "fantasy":          ["Fantasy"],
    "sci-fi":           ["Sci-Fi"],
    "scifi":            ["Sci-Fi"],
    "science fiction":  ["Sci-Fi"],
    "historical":       ["Historical", "Samurai"],
    "samurai":          ["Samurai", "Historical", "Action"],
    "ninja":            ["Action", "Shounen"],
    "vampire":          ["Vampire", "Supernatural", "Horror"],
    "supernatural":     ["Supernatural", "Fantasy"],
    "military":         ["Military", "Action"],
    "music":            ["Music"],
    "comedy":           ["Comedy"],
    "action":           ["Action"],
    "drama":            ["Drama"],
    "romance":          ["Romance"],
    "slice of life":    ["Slice of Life"],
}

# ── Length keywords ───────────────────────────────────────────────────────────
LENGTH_MAP = {
    "short":         (1, 13),
    "quick":         (1, 13),
    "mini":          (1,  6),
    "movie":         (1,  1),
    "film":          (1,  1),
    "long":          (24, 9999),
    "ongoing":       (100, 9999),
    "binge":         (12, 50),
    "season":        (10, 26),
    "single episode":(1, 1),
    "one episode":   (1, 1),
}

TYPE_MAP = {
    "movie": "Movie", "film": "Movie",
    "series": "TV",   "show": "TV",  "tv": "TV",
    "ova": "OVA",     "special": "Special", "ona": "ONA",
}

ERA_MAP = {
    "classic": (1990, 2005), "old": (1990, 2009), "retro": (1980, 2000),
    "new":     (2018, 2030), "modern": (2015, 2030),
    "recent":  (2020, 2030), "latest": (2022, 2030),
    "2000s":   (2000, 2009), "90s": (1990, 1999),
    "80s":     (1980, 1989),
}

INTENT_RE = {
    "greet":     r"\b(hi|hello|hey|howdy|sup|good\s*(morning|evening|afternoon))\b",
    "thanks":    r"\b(thank|thanks|thx|ty|appreciate)\b",
    "recommend": r"\b(recommend|suggest|find|show|give|want|looking|search|need|"
                 r"what.*watch|can you|could you|please|pick|discover|help|give me)\b",
    "similar":   r"\b(similar|like|same|such as|reminds|comparable|fan of|"
                 r"love.*same|more.*like|based on)\b",
    "mood":      r"\b(mood|feel|feeling|want something|in the|vibe)\b",
    "top":       r"\b(top|best|highest|greatest|most popular|must.?watch|must.?see|"
                 r"all time|goat|underrated|hidden gem)\b",
    "more":      r"\b(more|another|again|else|different)\b",
    "help":      r"\b(help|how|what can|what do|guide|what.*do)\b",
    "bye":       r"\b(bye|goodbye|see you|later|quit|exit)\b",
}

CONFIRM_WORDS = {"yes","yeah","yep","sure","ok","okay","great","love","perfect","more","another"}
REJECT_WORDS  = {"no","nope","nah","different","other","else","instead","try again","change"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"[^\w\s'-]", " ", text.lower().strip())

def _toks(text: str) -> List[str]:
    try:    toks = word_tokenize(_norm(text))
    except: toks = _norm(text).split()
    return [_LEM.lemmatize(t) for t in toks if t.isalpha()]

# FIX 4 — same safe genre-list guard as app.py
def _safe_gl(gl) -> List[str]:
    if isinstance(gl, list):
        return [x.lower() for x in gl]
    if isinstance(gl, str):
        return [x.strip().lower() for x in gl.split(",") if x.strip()]
    return []


def _find_title(lower_text: str, df: pd.DataFrame) -> Optional[str]:
    # 1. alias map
    for alias, canonical in ALIAS_MAP.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', lower_text):
            if df["Name"].str.lower().str.contains(canonical.lower(), regex=False).any():
                return canonical
            partial = df[df["Name"].str.lower().str.contains(
                canonical.split(":")[0].lower(), regex=False)]
            if not partial.empty:
                return partial.iloc[0]["Name"]

    # 2. direct substring match
    SINGLE_WORD_BL = {
        "dark","light","short","long","good","best","great","new","old","free",
        "real","true","god","war","one","two","hero","zero","time","life","love",
        "hate","gold","soul","game","play","day","night","world","blood","heart",
        "space","magic","pass","note","girl","boy","man","woman","king","queen",
        "angel","demon","dragon","monster","master","ghost","black","white","blue",
        "red","green","silver","super","ultra","mega","neo","score","action","drama",
        "anime","romance","horror","comedy","mystery","school","music","sports",
        "psycho","death","attack","sword","special","high","classic","retro",
    }
    titles_lower = {t.lower(): t for t in df["Name"].dropna()}
    best = (None, 0)
    for t_lower, t_orig in titles_lower.items():
        if len(t_lower) < 7:
            continue
        words = t_lower.split()
        if len(words) == 1 and t_lower in SINGLE_WORD_BL:
            continue
        if re.search(r'\b' + re.escape(t_lower) + r'\b', lower_text):
            if len(t_lower) > best[1]:
                best = (t_orig, len(t_lower))
    return best[0]


def _extract_entities(text: str, df: pd.DataFrame) -> dict:
    lower = _norm(text)
    ent = {
        "genres": [], "excluded_genres": [], "mood_genres": [],
        "mood": [], "length": None, "type_": None,
        "min_score": 6.0, "era": None, "title_ref": None, "n": 6,
        "raw_text": text,
    }

    # genres
    for g in ALL_GENRES:
        if re.search(r'\b' + re.escape(g.lower()) + r'\b', lower):
            if re.search(r'\b(no|not|without|avoid|hate|dislike)\s+' + re.escape(g.lower()), lower):
                ent["excluded_genres"].append(g)
            else:
                ent["genres"].append(g)

    # mood → genre
    for mood_kw, mood_genres in MOOD_MAP.items():
        if re.search(r'\b' + re.escape(mood_kw) + r'\b', lower):
            if mood_kw not in ent["mood"]:
                ent["mood"].append(mood_kw)
            for mg in mood_genres:
                if mg not in ent["genres"] and mg not in ent["mood_genres"]:
                    ent["mood_genres"].append(mg)

    # length
    for kw, (mn, mx) in LENGTH_MAP.items():
        if kw in lower:
            ent["length"] = (mn, mx)
            break
    m = re.search(r'(under|less than|fewer than|at most|max)\s*(\d+)\s*ep', lower)
    if m: ent["length"] = (1, int(m.group(2)))
    m = re.search(r'(over|more than|at least|min)\s*(\d+)\s*ep', lower)
    if m: ent["length"] = (int(m.group(2)), 9999)

    # type
    for kw, tv in TYPE_MAP.items():
        if re.search(r'\b' + kw + r'\b', lower):
            ent["type_"] = tv
            break

    # score threshold
    if any(w in lower for w in ["best","top","greatest","must watch","must-watch","goat","masterpiece"]):
        ent["min_score"] = 8.0
    elif any(w in lower for w in ["good","decent","nice","solid","underrated","hidden gem"]):
        ent["min_score"] = 7.0
    m = re.search(r'score\s*(above|over|at least|>)\s*(\d+\.?\d*)', lower)
    if m: ent["min_score"] = float(m.group(2))

    # era
    for kw, (ys, ye) in ERA_MAP.items():
        if kw in lower:
            ent["era"] = (ys, ye)
            break
    m = re.search(r'\b(19|20)(\d{2})\b', lower)
    if m:
        yr = int(m.group())
        ent["era"] = (yr - 1, yr + 1)

    # title
    ent["title_ref"] = _find_title(lower, df)

    # result count
    m = re.search(r'\b(\d+)\b.{0,20}(anime|show|recommendation|result|suggest)', lower)
    if m: ent["n"] = min(max(int(m.group(1)), 1), 12)

    return ent


def _detect_intent(text: str) -> List[str]:
    lower = text.lower()
    found = [k for k, pat in INTENT_RE.items() if re.search(pat, lower)]
    return found if found else ["recommend"]


def _build_query(ent: dict, df: pd.DataFrame) -> pd.DataFrame:
    res = df[df["Score"] >= ent["min_score"]].copy()

    wanted = list(set(ent["genres"] + ent["mood_genres"]))
    if wanted:
        wanted_l = [g.lower() for g in wanted]
        # FIX 4 — use _safe_gl to guard non-list genre_list values
        res = res[res["genre_list"].apply(
            lambda gl: any(g in _safe_gl(gl) for g in wanted_l)
        )]

    if ent["excluded_genres"]:
        excl_l = [g.lower() for g in ent["excluded_genres"]]
        res = res[res["genre_list"].apply(
            lambda gl: not any(e in _safe_gl(gl) for e in excl_l)
        )]

    if ent["type_"]:
        res = res[res["Type"] == ent["type_"]]

    # FIX 3 — coerce Episodes to numeric before comparison
    if ent["length"]:
        mn, mx = ent["length"]
        eps = pd.to_numeric(res["Episodes"], errors="coerce").fillna(0)
        res = res[(eps >= mn) & (eps <= mx)]

    if ent["era"]:
        ys, ye = ent["era"]
        def _in_era(aired):
            m = re.search(r'\b(19|20)\d{2}\b', str(aired))
            return ys <= int(m.group()) <= ye if m else False
        res = res[res["Aired"].apply(_in_era)]

    if len(res) == 0:
        return res

    score_n = (res["Score"] - res["Score"].min()) / (res["Score"].max() - res["Score"].min() + 1e-9)
    pop_n   = np.log1p(res["Members"]) / (np.log1p(res["Members"].max()) + 1e-9)
    res = res.copy()
    res["_rank"] = 0.65 * score_n + 0.35 * pop_n
    return res.sort_values("_rank", ascending=False)


def _semantic_search(query: str, df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    # FIX 5 — check text_features column exists before using it
    if "text_features" in df.columns:
        corpus = (df["text_features"].fillna("") + " " + df["Synopsis"].fillna("")).tolist()
    else:
        corpus = df["Synopsis"].fillna("").tolist()
    corpus.append(query)
    vec  = TfidfVectorizer(max_features=8000, stop_words="english",
                           ngram_range=(1, 2), sublinear_tf=True)
    mat  = vec.fit_transform(corpus)
    sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
    top  = np.argsort(sims)[::-1][:n * 3]
    return df.iloc[top].copy()


# ── Reply templates ───────────────────────────────────────────────────────────
_GREET  = [
    "Hey there! 👋 Tell me what kind of anime you're in the mood for — a genre, a vibe, or even describe a plot!",
    "Hi! 🎌 I'm your anime guide. What are you looking for today?",
    "Hello! Ready to find your next anime obsession? Just tell me what you like! 😊",
]
_THANKS = [
    "Happy to help! 😊 Want more recommendations?",
    "Glad you liked it! Tell me more about what you're looking for.",
    "Anytime! Shall I find you something else? 🎌",
]
_BYE   = ["Happy watching! 🎌", "See you! Enjoy your anime marathon! 🍿", "Goodbye! ✨"]
_NORES = [
    "I couldn't find a perfect match 🤔 — try different genres or relax the filters?",
    "Nothing came up for that! Maybe try a broader search?",
    "Hmm, let me try differently — could you describe what kind of story you want?",
]
_HELP  = """Here's what I can help you find:

🎭 **Genre** — *"I want action anime"* / *"show me romance"*
😊 **Mood** — *"something funny"* / *"I want to cry"* / *"dark and mysterious"*
📺 **Length** — *"short anime"* / *"under 12 episodes"* / *"a movie"*
⭐ **Quality** — *"best anime ever"* / *"top-rated psychological"*
📖 **Plot** — *"an anime about a boy who gains powers"*
🔍 **Similar** — *"something like Death Note"*

Just type naturally — I'll figure it out! 😄"""


def _build_reply(intents: List[str], ent: dict, n_found: int) -> str:
    if set(intents) <= {"greet"}:   return random.choice(_GREET)
    if set(intents) <= {"thanks"}:  return random.choice(_THANKS)
    if "bye" in intents:            return random.choice(_BYE)
    if set(intents) <= {"help"}:    return _HELP
    if n_found == 0:                return random.choice(_NORES)

    genres = ent["genres"] + ent["mood_genres"]
    moods  = ent["mood"]
    ref    = ent["title_ref"]

    if ref:
        opener = f"Since you like **{ref}**,"
    elif moods:
        opener = f"For a **{moods[0]}** vibe,"
    elif genres:
        opener = f"For **{' + '.join(genres[:2])}**,"
    else:
        opener = random.choice([
            "Here's what I found —",
            "Check these out —",
            "Based on what you said —",
        ])

    closer = random.choice([
        f"here are **{n_found}** anime you'll love! ✨",
        f"I found **{n_found}** great picks for you! 🎌",
        f"these **{n_found}** should be perfect! 🍿",
    ])

    extras = []
    if ent["type_"]:
        extras.append(f"format: **{ent['type_']}**")
    if ent["length"]:
        mn, mx = ent["length"]
        if mx == 9999:       extras.append(f"**{mn}+** episodes")
        elif mn == mx == 1:  extras.append("movies only")
        else:                extras.append(f"**{mn}–{mx}** episodes")
    if ent["min_score"] >= 8.0:
        extras.append("top-rated ⭐")

    reply = f"{opener} {closer}"
    if extras:
        reply += f"\n*(Filters: {', '.join(extras)})*"
    return reply


def _needs_clarification(ent: dict, text: str) -> Optional[str]:
    genres = ent["genres"] + ent["mood_genres"]
    if (not genres and not ent["title_ref"] and not ent["type_"] and
            not ent["era"] and len(text.split()) <= 3 and
            not any(w in text.lower() for w in ["best","top","popular","good","anime"])):
        return random.choice([
            "What kind of anime are you in the mood for? Try saying something like *'funny and short'*, *'dark thriller'*, or *'similar to Naruto'*.",
            "Could you tell me more? For example — a genre like **Action** or **Romance**, a mood like **dark** or **relaxing**, or an anime you already love.",
        ])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main ChatBot class
# ══════════════════════════════════════════════════════════════════════════════

class AnimeChatBot:
    def __init__(self, df: pd.DataFrame):
        self.df             = df.copy()
        self.history        : List[dict] = []
        self._last_results  : Optional[pd.DataFrame] = None
        self._last_entities : Optional[dict] = None
        self._last_offset   : int = 0

    def _merge_ctx(self, new_ent: dict) -> dict:
        if not self._last_entities:
            return new_ent
        merged = dict(new_ent)
        if not new_ent["genres"] and not new_ent["mood_genres"]:
            merged["genres"]      = self._last_entities.get("genres", [])
            merged["mood_genres"] = self._last_entities.get("mood_genres", [])
            merged["mood"]        = self._last_entities.get("mood", [])
        merged["excluded_genres"] = list(set(
            self._last_entities.get("excluded_genres", []) + new_ent["excluded_genres"]
        ))
        return merged

    def _enrich(self, results: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [c for c in ["Image URL", "anime_id"] if c not in results.columns]
        if missing_cols:
            merge_cols = ["Name"] + [c for c in ["Image URL", "anime_id"] if c in self.df.columns]
            results = results.merge(self.df[merge_cols], on="Name", how="left")
        return results.drop_duplicates("Name").reset_index(drop=True)

    def chat(self, user_text: str) -> Tuple[str, Optional[pd.DataFrame]]:
        intents = _detect_intent(user_text)
        ent     = _extract_entities(user_text, self.df)
        lower   = _norm(user_text)
        words   = set(lower.split())

        # "more" / "show more"
        if (words & {"more","another","again"}) and self._last_results is not None and len(words) <= 5:
            self._last_offset += ent["n"]
            more = self._last_results.iloc[self._last_offset: self._last_offset + ent["n"]]
            if more.empty:
                return "I've run out of results! Try a different genre or lower the rating? 🤔", None
            return f"Here are **{len(more)}** more! 🎌", self._enrich(more)

        # rejection
        if words & REJECT_WORDS and self._last_entities:
            self._last_entities = None
            self._last_results  = None
            return "No problem! What should I try instead? Tell me a genre, mood, or anime you like. 😊", None

        # simple intents
        if set(intents) <= {"greet","thanks","bye","help"}:
            reply = _build_reply(intents, ent, 0)
            self._log(user_text, reply)
            return reply, None

        # clarification
        merged = self._merge_ctx(ent)
        clr = _needs_clarification(merged, user_text)
        if clr:
            self._log(user_text, clr)
            return clr, None

        # CB similar-title search
        if merged["title_ref"] and ("similar" in intents or
                re.search(r'\b(similar|like|based on|fan of|love)\b', lower)):
            try:
                from content_based import ContentBasedRecommender
                cb = ContentBasedRecommender.load() if os.path.exists("cb_model.pkl") \
                     else ContentBasedRecommender().fit(self.df)
                pool = cb.recommend(merged["title_ref"], n=merged["n"] * 4)
                pool = pool[pool["Score"] >= merged["min_score"]]
                if merged["type_"]:
                    pool = pool[pool["Type"] == merged["type_"]]
                merge_cols = ["Name"] + [c for c in ["Image URL","anime_id"] if c in self.df.columns]
                pool = pool.merge(self.df[merge_cols], on="Name", how="left")
                pool = pool.drop_duplicates("Name")
                results = pool.head(merged["n"])
                reply   = _build_reply(intents, merged, len(results))
                self._save(results, merged)
                self._log(user_text, reply)
                return reply, self._enrich(results) if len(results) else None
            except Exception:
                pass  # fall through

        # top / best with no genres → global top
        if "top" in intents and not merged["genres"] and not merged["mood_genres"]:
            merged["min_score"] = max(merged["min_score"], 8.0)

        # semantic fallback for plot descriptions
        use_semantic = (
            not merged["genres"] and not merged["mood_genres"] and
            not merged["title_ref"] and len(user_text.split()) > 5
        )

        if use_semantic:
            pool = _semantic_search(user_text, self.df, n=merged["n"] * 3)
            pool = pool[pool["Score"] >= merged["min_score"]]
            if merged["type_"]:
                pool = pool[pool["Type"] == merged["type_"]]
            if merged["length"]:
                mn, mx = merged["length"]
                eps = pd.to_numeric(pool["Episodes"], errors="coerce").fillna(0)
                pool = pool[(eps >= mn) & (eps <= mx)]
        else:
            pool = _build_query(merged, self.df)

        results = pool.head(merged["n"] * 4)
        shown   = results.head(merged["n"])

        reply = _build_reply(intents, merged, len(shown)) if not shown.empty else random.choice(_NORES)
        self._save(results, merged)
        self._log(user_text, reply)
        return reply, self._enrich(shown) if not shown.empty else None

    def _save(self, results: pd.DataFrame, ent: dict):
        self._last_results  = results
        self._last_entities = ent
        self._last_offset   = 0

    def _log(self, user_text: str, reply: str):
        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "bot",  "text": reply})

    def reset(self):
        self.history        = []
        self._last_results  = None
        self._last_entities = None
        self._last_offset   = 0