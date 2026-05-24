"""
app.py — AniWatch  (Netflix-style Anime Recommender)
Run:  streamlit run app.py

FIXES applied:
  1. Removed invalid 'Netflix+Sans' Google Font import
  2. Nav links are purely decorative HTML; real navigation via Streamlit buttons only (no void(0))
  3. Safe merge — only merges columns that actually exist in df
  4. Hero selection guarded with fallback when no anime scores ≥ 9.0
  5. Synopsis cast to str before slicing to avoid NaN crash
  6. genre_list membership check guarded against NaN/non-list values
  7. 'from collections import Counter' moved to top-level imports
  8. render_row wraps correctly — iterates df_slice rows, not enumerate(columns)
  9. Removed duplicate/conflicting HTML nav onclick handlers
 10. Session state page key initialised before any read
"""

import os
from collections import Counter   # FIX 7 — top-level import

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_preprocessing import load_raw_data, preprocess, load_processed, PROCESSED_DATA_PATH
from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from hybrid_recommender import HybridRecommender
import user_profile as up
from nlp_chat import AnimeChatBot

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AniWatch",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# FIX 1 — removed non-existent 'Netflix+Sans' font; kept Inter only
NETFLIX_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#141414 !important;color:#e5e5e5;}
.main,.block-container{background:#141414 !important;padding:0 !important;max-width:100% !important;}
[data-testid="stAppViewContainer"]{background:#141414 !important;}
[data-testid="stHeader"]{display:none;}
[data-testid="stSidebar"]{display:none;}
footer{display:none;}
.stDeployButton{display:none;}

/* ── Netflix navbar built from Streamlit buttons ── */
.nf-navbar-wrap{
  background:#141414;
  border-bottom:1px solid #222;
  position:sticky;
  top:0;
  z-index:100;
  display:flex;
  align-items:center;
  padding:0 56px;
  height:68px;
  gap:0;
}
.nf-logo{color:#e50914;font-size:28px;font-weight:700;letter-spacing:-1px;margin-right:24px;white-space:nowrap;}
.nf-nav-right{margin-left:auto;display:flex;align-items:center;}
.nf-avatar{width:34px;height:34px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#fff;}

/* Make the nav button container flush */
div[data-testid="stHorizontalBlock"].nav-row > div{padding:0 !important;}

/* Override ALL stButton styles for nav buttons only */
.nav-btn-wrap{overflow:hidden;height:68px;display:flex;align-items:center;}
.nav-btn-wrap > div{width:100%;}
.nav-btn-wrap button{
  background:transparent !important;
  color:#ccc !important;
  border:none !important;
  border-radius:0 !important;
  font-size:13px !important;
  font-weight:400 !important;
  padding:0 10px !important;
  height:68px !important;
  min-height:68px !important;
  max-height:68px !important;
  line-height:68px !important;
  width:100% !important;
  min-width:0 !important;
  transition:color .15s !important;
  box-shadow:none !important;
  white-space:nowrap !important;
  overflow:hidden !important;
  text-overflow:ellipsis !important;
  display:block !important;
}
.nav-btn-wrap button:hover{color:#fff !important;background:transparent !important;}
.nav-btn-wrap.active button{color:#fff !important;font-weight:600 !important;border-bottom:2px solid #e50914 !important;}

.nf-hero{
  background:#1a1a2e;
  padding:80px 56px 60px;
  position:relative;
  overflow:hidden;
  margin-bottom:32px;
}
.nf-hero-badge{background:#e50914;color:#fff;font-size:11px;font-weight:700;padding:3px 10px;border-radius:2px;display:inline-block;margin-bottom:16px;letter-spacing:.5px;}
.nf-hero-title{font-size:48px;font-weight:700;color:#fff;line-height:1.1;margin-bottom:12px;max-width:560px;}
.nf-hero-meta{color:#aaa;font-size:15px;margin-bottom:20px;display:flex;gap:16px;align-items:center;}
.nf-hero-desc{color:#ccc;font-size:15px;max-width:500px;line-height:1.6;margin-bottom:28px;}

.nf-section{padding:0 56px;margin-bottom:40px;}
.nf-section-title{font-size:20px;font-weight:700;color:#e5e5e5;margin-bottom:16px;display:flex;align-items:center;gap:10px;}

.nf-row{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;}
.nf-card{
  border-radius:4px;
  overflow:hidden;
  background:#1f1f1f;
  cursor:pointer;
  transition:transform .2s,box-shadow .2s;
  position:relative;
}
.nf-card:hover{transform:scale(1.05);box-shadow:0 8px 32px rgba(0,0,0,.8);z-index:2;}
.nf-card-img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#2a2a2a;}
.nf-card-img-placeholder{width:100%;aspect-ratio:2/3;background:#2a2a2a;display:flex;align-items:center;justify-content:center;font-size:32px;}
.nf-card-body{padding:10px;}
.nf-card-title{font-size:12px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;}
.nf-card-score{font-size:11px;color:#46d369;font-weight:600;}
.nf-card-genres{font-size:10px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px;}

.nf-list-item{display:flex;gap:16px;background:#1f1f1f;border-radius:4px;padding:12px;margin-bottom:8px;align-items:center;}
.nf-list-thumb{width:60px;height:90px;border-radius:4px;object-fit:cover;flex-shrink:0;background:#2a2a2a;}
.nf-list-thumb-ph{width:60px;height:90px;border-radius:4px;background:#2a2a2a;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.nf-list-info{flex:1;min-width:0;}
.nf-list-title{font-size:15px;font-weight:600;color:#fff;margin-bottom:4px;}
.nf-list-meta{font-size:12px;color:#aaa;margin-bottom:6px;}
.nf-list-genres{font-size:11px;color:#666;}

.nf-tag{display:inline-block;background:#2a2a2a;color:#ccc;border-radius:3px;padding:3px 10px;font-size:12px;margin:3px 4px 3px 0;cursor:pointer;border:1px solid #333;transition:all .15s;}
.nf-tag:hover,.nf-tag.selected{background:#e50914;color:#fff;border-color:#e50914;}
.nf-match{color:#46d369;font-size:12px;font-weight:600;}

input[type="text"],input[type="search"],[data-baseweb="input"] input{background:#333 !important;color:#fff !important;border:1px solid #555 !important;border-radius:4px !important;}
[data-baseweb="select"] div{background:#333 !important;color:#fff !important;}
label,[data-testid="stWidgetLabel"]{color:#ccc !important;}
/* ── Chat UI ── */
.chat-wrap{max-width:860px;margin:0 auto;padding:0 24px 120px;}
.chat-bubble-row{display:flex;margin-bottom:18px;align-items:flex-end;gap:10px;}
.chat-bubble-row.user{flex-direction:row-reverse;}
.chat-avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.chat-avatar.bot{background:#e50914;}
.chat-avatar.user{background:#3b82f6;}
.chat-bubble{max-width:72%;padding:12px 16px;border-radius:16px;font-size:14px;line-height:1.6;}
.chat-bubble.bot{background:#1f1f1f;color:#e5e5e5;border-bottom-left-radius:4px;}
.chat-bubble.user{background:#e50914;color:#fff;border-bottom-right-radius:4px;}
.chat-results{margin-top:12px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.chat-card{background:#2a2a2a;border-radius:6px;overflow:hidden;display:flex;gap:10px;padding:8px;align-items:center;}
.chat-card-img{width:40px;height:60px;border-radius:4px;object-fit:cover;flex-shrink:0;background:#333;}
.chat-card-img-ph{width:40px;height:60px;border-radius:4px;background:#333;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;}
.chat-card-info{flex:1;min-width:0;}
.chat-card-title{font-size:11px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.chat-card-score{font-size:10px;color:#46d369;margin-top:2px;}
.chat-card-genre{font-size:10px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:1px;}
.chat-input-bar{position:fixed;bottom:0;left:0;right:0;background:#141414;border-top:1px solid #222;padding:16px 56px;z-index:200;}
.chat-typing{color:#888;font-size:13px;padding:4px 0 8px 42px;font-style:italic;}


.nf-profile-card{background:#1f1f1f;border-radius:8px;padding:28px;text-align:center;}
.nf-profile-avatar{width:84px;height:84px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:32px;font-weight:700;color:#fff;margin:0 auto 16px;}
.nf-profile-name{font-size:22px;font-weight:700;color:#fff;}
.nf-profile-since{font-size:13px;color:#888;margin-top:4px;}
.nf-stat-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px;}
.nf-stat-box{background:#2a2a2a;border-radius:6px;padding:16px;text-align:center;}
.nf-stat-n{font-size:28px;font-weight:700;color:#e50914;}
.nf-stat-l{font-size:12px;color:#888;margin-top:4px;}

.nf-empty{text-align:center;padding:60px 20px;color:#555;}
.nf-empty-icon{font-size:52px;margin-bottom:16px;}
.nf-empty-text{font-size:16px;}

.nf-page-hero{background:#1a1a1a;padding:40px 56px 32px;border-bottom:1px solid #222;margin-bottom:28px;}
.nf-page-title{font-size:32px;font-weight:700;color:#fff;margin-bottom:6px;}
.nf-page-sub{font-size:15px;color:#888;}

.nf-divider{border:none;border-top:1px solid #222;margin:24px 0;}

/* ── Mood picker pills ── */
.mood-bar{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;}
.mood-pill{
  display:inline-flex;align-items:center;gap:8px;
  background:#1f1f1f;border:1.5px solid #333;border-radius:999px;
  padding:10px 22px;font-size:14px;font-weight:600;color:#ccc;
  cursor:pointer;transition:all .18s;white-space:nowrap;
}
.mood-pill:hover{border-color:#e50914;color:#fff;background:#2a0a0a;}
.mood-pill.picked{background:#e50914;border-color:#e50914;color:#fff;}

/* ── Roulette spotlight card ── */
.roulette-card{
  background:linear-gradient(135deg,#1a1a2e,#16213e);
  border:1.5px solid #e50914;border-radius:10px;
  padding:28px 32px;display:flex;gap:24px;align-items:center;
  margin-bottom:8px;
}
.roulette-img{width:80px;height:120px;border-radius:6px;object-fit:cover;flex-shrink:0;background:#2a2a2a;}
.roulette-img-ph{width:80px;height:120px;border-radius:6px;background:#2a2a2a;display:flex;align-items:center;justify-content:center;font-size:36px;flex-shrink:0;}
.roulette-info{flex:1;}
.roulette-badge{background:#e50914;color:#fff;font-size:10px;font-weight:700;padding:2px 8px;border-radius:2px;display:inline-block;margin-bottom:8px;letter-spacing:.5px;}
.roulette-title{font-size:22px;font-weight:700;color:#fff;margin-bottom:6px;}
.roulette-meta{font-size:13px;color:#aaa;margin-bottom:6px;}
.roulette-desc{font-size:13px;color:#888;line-height:1.5;}

/* Red style for all action buttons EXCEPT nav */
.stButton:not(.nav-btn-wrap) button{
  background:#e50914 !important;
  color:#fff !important;
  border:none !important;
  border-radius:4px !important;
  font-weight:600 !important;
  /* fixed size — never grows with label */
  height:32px !important;
  min-height:32px !important;
  max-height:32px !important;
  padding:0 8px !important;
  font-size:12px !important;
  white-space:nowrap !important;
  overflow:hidden !important;
  text-overflow:ellipsis !important;
  width:100% !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
}
.stButton:not(.nav-btn-wrap) button:hover{background:#f40612 !important;}
[data-testid="stSelectbox"] > div > div{background:#1f1f1f !important;border-color:#555 !important;color:#fff !important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:#141414;}
::-webkit-scrollbar-thumb{background:#555;border-radius:2px;}
</style>
"""

st.markdown(NETFLIX_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data & model loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        return load_processed()
    return preprocess(load_raw_data())

@st.cache_resource(show_spinner=False)
def get_cb(df):
    return ContentBasedRecommender.load() if os.path.exists("cb_model.pkl") \
           else ContentBasedRecommender().fit(df)

@st.cache_resource(show_spinner=False)
def get_cf(df):
    return CollaborativeFilteringRecommender.load() if os.path.exists("cf_model.pkl") \
           else CollaborativeFilteringRecommender(n_factors=50).fit(df)

@st.cache_resource(show_spinner=False)
def get_hybrid(df):
    return HybridRecommender.load() if os.path.exists("hybrid_model.pkl") \
           else HybridRecommender().fit(df)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score(v):
    try:
        return float(v)
    except:
        return 0.0

def _genres_short(g, n=3):
    parts = [x.strip() for x in str(g).split(",") if x.strip() and x.strip() != "Unknown"]
    return " · ".join(parts[:n])

def _img(url):
    if isinstance(url, str) and url.startswith("http"):
        return url
    return None

def _eps_str(eps):
    try:
        e = int(float(eps))
        return f"{e} ep{'s' if e != 1 else ''}"
    except:
        return ""

# FIX 3 — safe merge that only requests columns present in the DataFrame
def _safe_merge(results, df):
    """Merge image/id columns from df into results, only if those columns exist."""
    cols_wanted = ["Name"]
    for col in ["Image URL", "image_url", "anime_id"]:
        if col in df.columns:
            cols_wanted.append(col)
    extra = [c for c in cols_wanted if c != "Name"]
    if not extra:
        return results
    return results.merge(df[cols_wanted], on="Name", how="left")


def render_card(row, key_prefix="c"):
    name     = row.get("Name") or row.get("name", "")
    score    = _score(row.get("Score") or row.get("score", 0))
    genres   = row.get("Genres") or row.get("genres", "")
    img      = _img(row.get("Image URL") or row.get("image_url", ""))
    anime_id = int(row.get("anime_id", 0))

    uid = f"{key_prefix}_{anime_id}"

    in_wl  = up.is_in("watchlist",  anime_id)
    in_fav = up.is_in("favourites", anime_id)
    in_sv  = up.is_in("saved",      anime_id)

    if img:
        img_html = f'<img class="nf-card-img" src="{img}" loading="lazy">'
    else:
        icons = ["🎬","⚔️","🧠","🌊","🔥","🌸","👊","💀","🗡️","🤖"]
        ico = icons[anime_id % len(icons)]
        img_html = f'<div class="nf-card-img-placeholder">{ico}</div>'

    score_color = "#46d369" if score >= 7.5 else "#f5a623" if score >= 6 else "#e50914"
    score_str = f"{score:.1f}" if score else "N/A"

    st.markdown(f"""
    <div class="nf-card">
      {img_html}
      <div class="nf-card-body">
        <div class="nf-card-title" title="{name}">{name[:28]}{"…" if len(name)>28 else ""}</div>
        <div class="nf-card-score" style="color:{score_color};">★ {score_str}</div>
        <div class="nf-card-genres">{_genres_short(genres, 2)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        label = "✓ Watch" if in_wl else "+ Watch"
        if st.button(label, key=f"wl_{uid}", use_container_width=True,
                     help="Add / remove from Watchlist"):
            up.toggle_watchlist(anime_id, name, genres, score,
                                row.get("Image URL") or row.get("image_url", ""))
            st.rerun()
    with b2:
        label = "♥" if in_fav else "♡"
        if st.button(label, key=f"fv_{uid}", use_container_width=True,
                     help="Add / remove from Favourites"):
            up.toggle_favourite(anime_id, name, genres, score,
                                row.get("Image URL") or row.get("image_url", ""))
            st.rerun()
    with b3:
        label = "🔖✓" if in_sv else "🔖"
        if st.button(label, key=f"sv_{uid}", use_container_width=True,
                     help="Save for later"):
            up.toggle_saved(anime_id, name, genres, score,
                            row.get("Image URL") or row.get("image_url", ""))
            st.rerun()


# FIX 8 — render_row now iterates over df_slice rows correctly,
# so ALL rows are rendered even when len > cols
def render_row(df_slice, cols=6, key_prefix="row"):
    n = len(df_slice)
    if n == 0:
        return
    # Render in chunks of `cols` so overflow wraps to new rows
    for chunk_start in range(0, n, cols):
        chunk = df_slice.iloc[chunk_start:chunk_start + cols]
        columns = st.columns(len(chunk))
        for i, col in enumerate(columns):
            with col:
                render_card(chunk.iloc[i], key_prefix=f"{key_prefix}_{chunk_start + i}")


def render_list_item(item, list_key, key_prefix="li"):
    name     = item.get("name", "")
    score    = _score(item.get("score", 0))
    genres   = item.get("genres", "")
    img      = _img(item.get("image_url", ""))
    added_at = item.get("added_at", "")
    anime_id = item.get("anime_id", 0)
    uid      = f"{key_prefix}_{anime_id}"

    if img:
        img_html = f'<img class="nf-list-thumb" src="{img}" loading="lazy">'
    else:
        icons = ["🎬","⚔️","🧠","🌊","🔥","🌸"]
        ico = icons[anime_id % len(icons)]
        img_html = f'<div class="nf-list-thumb-ph">{ico}</div>'

    sc = f"{score:.1f}" if score else "N/A"
    short_g = _genres_short(genres, 3)

    st.markdown(f"""
    <div class="nf-list-item">
      {img_html}
      <div class="nf-list-info">
        <div class="nf-list-title">{name}</div>
        <div class="nf-list-meta">★ {sc} &nbsp;|&nbsp; {short_g}</div>
        <div class="nf-list-genres">Added {added_at}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Remove", key=f"rm_{list_key}_{uid}"):
        up.remove_from(list_key, anime_id)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# State & data
# ─────────────────────────────────────────────────────────────────────────────

df           = get_data()
profile      = up.get_profile()
stats        = up.get_stats()
all_titles   = sorted(df["Name"].dropna().unique().tolist())

# FIX 6 — guard genre_list against NaN / non-list entries
def _safe_genre_list(g):
    if isinstance(g, list):
        return g
    if isinstance(g, str):
        return [x.strip() for x in g.split(",") if x.strip()]
    return []

all_genres = sorted(set(
    g for gl in df["genre_list"].apply(_safe_genre_list) for g in gl
))

# FIX 10 — initialise page before any read
if "page" not in st.session_state:
    st.session_state.page = "home"


# ─────────────────────────────────────────────────────────────────────────────
# Top navigation bar
# ─────────────────────────────────────────────────────────────────────────────

pages = [
    ("home",       "Home"),
    ("discover",   "Find Similar"),
    ("genres",     "Browse Genres"),
    ("chat",       "💬 Ask AI"),
    ("watchlist",  "My List"),
    ("favourites", "Favourites"),
    ("saved",      "Saved"),
    ("profile",    "Profile"),
]

initials     = "".join(w[0].upper() for w in profile["username"].split()[:2])
avatar_color = profile.get("avatar_color", "#e50914")

# Single navbar: logo + Streamlit nav buttons + avatar, all in one flex row
st.markdown(f"""
<div class="nf-navbar-wrap">
  <span class="nf-logo">AniWatch</span>
  <div id="nav-buttons-slot" style="display:flex;align-items:center;flex:1;height:68px;"></div>
  <div class="nf-nav-right">
    <div class="nf-avatar" style="background:{avatar_color};">{initials}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Render nav buttons — wrapped in .nav-btn-wrap divs for CSS targeting
nav_cols = st.columns(len(pages))
for i, (key, label) in enumerate(pages):
    is_active = "active" if st.session_state.page == key else ""
    with nav_cols[i]:
        st.markdown(f'<div class="nav-btn-wrap {is_active}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page


# ═════════════════════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════════════════════

if page == "home":

    # ── session state for new features ──────────────────────────────────────
    if "mood" not in st.session_state:
        st.session_state.mood = None
    if "roulette_idx" not in st.session_state:
        st.session_state.roulette_idx = None

    # ── Hero banner ─────────────────────────────────────────────────────────
    hero_pool = df[df["Score"] >= 9.0].nlargest(1, "Members")
    if hero_pool.empty:
        hero_pool = df.nlargest(1, "Score")
    hero = hero_pool.iloc[0]

    hero_img    = _img(hero.get("Image URL", ""))
    hero_score  = _score(hero["Score"])
    hero_eps    = _eps_str(hero["Episodes"])
    hero_genres = _genres_short(hero["Genres"], 3)
    synopsis    = str(hero.get("Synopsis", "") or "")[:220].strip()

    hero_img_style = (
        f'background-image:url("{hero_img}");background-size:cover;background-position:top center;'
        if hero_img else "background:#1a1a2e;"
    )

    st.markdown(f"""
    <div class="nf-hero" style="{hero_img_style}">
      <div style="background:linear-gradient(to right,#141414 45%,transparent);position:absolute;inset:0;"></div>
      <div style="position:relative;">
        <div class="nf-hero-badge">★ TOP RATED</div>
        <div class="nf-hero-title">{hero["Name"]}</div>
        <div class="nf-hero-meta">
          <span class="nf-match">★ {hero_score:.1f} / 10</span>
          <span style="color:#555;">|</span>
          <span>{hero_eps}</span>
          <span style="color:#555;">|</span>
          <span>{hero_genres}</span>
        </div>
        <div class="nf-hero-desc">{synopsis}{"…" if len(synopsis) == 220 else ""}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 1 — Mood Picker
    # ════════════════════════════════════════════════════════════════════════
    MOODS = {
        "😈 Dark & Intense":   ["Psychological", "Thriller", "Horror", "Mystery"],
        "😂 Funny & Light":    ["Comedy", "Slice of Life", "Parody"],
        "🥰 Wholesome":        ["Romance", "Drama", "School", "Shoujo"],
        "💥 Action-Packed":    ["Action", "Adventure", "Shounen", "Mecha"],
        "🌌 Mind-Bending":     ["Sci-Fi", "Psychological", "Supernatural", "Fantasy"],
        "😢 Emotional":        ["Drama", "Romance", "Tragedy"],
    }

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)
    st.markdown('<div class="nf-section-title">🎭 What\'s Your Mood?</div>', unsafe_allow_html=True)
    st.markdown('<div class="mood-bar">', unsafe_allow_html=True)

    mood_cols = st.columns(len(MOODS))
    for i, (mood_label, _) in enumerate(MOODS.items()):
        with mood_cols[i]:
            picked_class = "picked" if st.session_state.mood == mood_label else ""
            st.markdown(f'<div class="mood-pill {picked_class}" style="pointer-events:none;">{mood_label}</div>',
                        unsafe_allow_html=True)
            btn_label = "✓ Selected" if st.session_state.mood == mood_label else "Pick"
            if st.button(btn_label, key=f"mood_{i}", use_container_width=True):
                # toggle off if already selected
                st.session_state.mood = None if st.session_state.mood == mood_label else mood_label
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Show mood results if one is picked
    if st.session_state.mood:
        mood_genres = MOODS[st.session_state.mood]
        mood_mask   = df["genre_list"].apply(
            lambda g: any(mg in _safe_genre_list(g) for mg in mood_genres)
        )
        mood_results = df[mood_mask].nlargest(6, "Score")
        if not mood_results.empty:
            st.markdown(f'<div class="nf-section-title" style="margin-top:16px;">Top picks for <em>{st.session_state.mood}</em></div>',
                        unsafe_allow_html=True)
            render_row(mood_results, cols=6, key_prefix="mood_res")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 2 — Random Roulette
    # ════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="nf-section">', unsafe_allow_html=True)

    roul_col, spin_col = st.columns([5, 1])
    with roul_col:
        st.markdown('<div class="nf-section-title">🎰 Anime Roulette — Feeling Lucky?</div>',
                    unsafe_allow_html=True)
    with spin_col:
        if st.button("🎲 Spin!", key="roulette_spin", use_container_width=True):
            # Pick from top 500 so quality stays reasonable
            pool = df[df["Score"] >= 7.0]
            if pool.empty:
                pool = df
            st.session_state.roulette_idx = int(
                np.random.choice(pool.index)
            )
            st.rerun()

    if st.session_state.roulette_idx is not None:
        try:
            r = df.loc[st.session_state.roulette_idx]
            r_img    = _img(r.get("Image URL", ""))
            r_score  = _score(r["Score"])
            r_eps    = _eps_str(r["Episodes"])
            r_genres = _genres_short(r["Genres"], 3)
            r_syn    = str(r.get("Synopsis", "") or "")[:160].strip()

            if r_img:
                r_img_html = f'<img class="roulette-img" src="{r_img}">'
            else:
                icons = ["🎬","⚔️","🧠","🌊","🔥","🌸","👊","💀","🗡️","🤖"]
                ico   = icons[int(r.get("anime_id", 0)) % len(icons)]
                r_img_html = f'<div class="roulette-img-ph">{ico}</div>'

            st.markdown(f"""
            <div class="roulette-card">
              {r_img_html}
              <div class="roulette-info">
                <div class="roulette-badge">🎲 TODAY'S PICK</div>
                <div class="roulette-title">{r["Name"]}</div>
                <div class="roulette-meta">★ {r_score:.1f} &nbsp;|&nbsp; {r_eps} &nbsp;|&nbsp; {r_genres}</div>
                <div class="roulette-desc">{r_syn}{"…" if len(r_syn)==160 else ""}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.session_state.roulette_idx = None
    else:
        st.markdown('<div style="color:#555;font-size:14px;padding:16px 0;">Hit Spin to get a random anime recommendation!</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # ── Trending Now ─────────────────────────────────────────────────────────
    st.markdown('<div class="nf-section">', unsafe_allow_html=True)
    st.markdown('<div class="nf-section-title">🔥 Trending Now</div>', unsafe_allow_html=True)
    render_row(df.nlargest(6, "Members"), cols=6, key_prefix="trend")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # ── Top Rated ────────────────────────────────────────────────────────────
    st.markdown('<div class="nf-section">', unsafe_allow_html=True)
    st.markdown('<div class="nf-section-title">⭐ Top Rated of All Time</div>', unsafe_allow_html=True)
    render_row(df.nlargest(6, "Score"), cols=6, key_prefix="toprated")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 3 — Because You Watched X
    # ════════════════════════════════════════════════════════════════════════
    wl_items = up.get_profile()["watchlist"]
    if wl_items:
        # Pick the most recently added watchlist item as the seed
        seed_name = wl_items[0]["name"]
        try:
            cb        = get_cb(df)
            byw_res   = cb.recommend(seed_name, n=6)
            byw_res   = _safe_merge(byw_res, df)
            if not byw_res.empty:
                st.markdown('<div class="nf-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="nf-section-title">🎯 Because You Saved <em>{seed_name}</em></div>',
                            unsafe_allow_html=True)
                render_row(byw_res, cols=6, key_prefix="byw")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)
        except Exception:
            pass

    # ── Genre rows ───────────────────────────────────────────────────────────
    for genre_label, genre_name, emoji in [
        ("Action Picks",       "Action",  "⚔️"),
        ("Romance",            "Romance", "🌸"),
        ("Mystery & Thriller", "Mystery", "🕵️"),
    ]:
        mask = df["genre_list"].apply(lambda g: genre_name in _safe_genre_list(g))
        sub  = df[mask].nlargest(6, "Score")
        if not sub.empty:
            st.markdown('<div class="nf-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="nf-section-title">{emoji} {genre_label}</div>',
                        unsafe_allow_html=True)
            render_row(sub, cols=6, key_prefix=f"genre_{genre_name}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 4 — Hidden Gems
    # ════════════════════════════════════════════════════════════════════════
    gems = df[
        (df["Score"] >= 7.5) &
        (pd.to_numeric(df["Members"], errors="coerce").fillna(0) < 50_000)
    ].nlargest(6, "Score")

    if not gems.empty:
        st.markdown('<div class="nf-section">', unsafe_allow_html=True)
        st.markdown('<div class="nf-section-title">💎 Hidden Gems — Under the Radar</div>',
                    unsafe_allow_html=True)
        render_row(gems, cols=6, key_prefix="gems")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)

    # ── Recently Added ───────────────────────────────────────────────────────
    st.markdown('<div class="nf-section">', unsafe_allow_html=True)
    st.markdown('<div class="nf-section-title">🆕 Recently Added</div>', unsafe_allow_html=True)
    new_ones = df.sort_values("anime_id", ascending=False).head(6)
    render_row(new_ones, cols=6, key_prefix="new")
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# FIND SIMILAR
# ═════════════════════════════════════════════════════════════════════════════

elif page == "discover":
    st.markdown("""
    <div class="nf-page-hero">
      <div class="nf-page-title">Find Similar Anime</div>
      <div class="nf-page-sub">Loved an anime? We'll find more just like it.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        default_idx = all_titles.index("Death Note") if "Death Note" in all_titles else 0
        chosen = st.selectbox("Choose an anime you enjoyed", all_titles, index=default_idx)
    with col_b:
        n_results = st.selectbox("Show", [6, 12, 18, 24], index=1)
    with col_c:
        type_map = {"Any format": None, "TV Series": "TV", "Movie": "Movie",
                    "OVA": "OVA", "ONA": "ONA"}
        fmt = st.selectbox("Format", list(type_map.keys()))

    min_score = st.slider("Minimum rating", 1.0, 10.0, 6.5, 0.5)

    if st.button("✦ Find Similar Anime", use_container_width=False):
        with st.spinner(""):
            cb = get_cb(df)
            ft = type_map[fmt]
            try:
                results = cb.recommend(chosen, n=n_results * 3, filter_type=ft)
                results = results[results["Score"] >= min_score].head(n_results)
                results = _safe_merge(results, df)   # FIX 3

                if results.empty:
                    st.markdown('<div class="nf-empty"><div class="nf-empty-icon">🔍</div><div class="nf-empty-text">No results — try relaxing the filters.</div></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="nf-section-title" style="margin-top:24px;">Because you like <em>{chosen}</em></div>',
                                unsafe_allow_html=True)
                    render_row(results, cols=6, key_prefix="sim")
            except ValueError as e:
                st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# BROWSE GENRES
# ═════════════════════════════════════════════════════════════════════════════

elif page == "genres":
    st.markdown("""
    <div class="nf-page-hero">
      <div class="nf-page-title">Browse by Genre</div>
      <div class="nf-page-sub">Pick what you're in the mood for.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)

    GENRE_GRID = [
        "Action", "Adventure", "Comedy", "Drama", "Fantasy",
        "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life",
        "Sports", "Supernatural", "Thriller", "Psychological",
        "Mecha", "Music", "School", "Shounen", "Seinen", "Isekai",
    ]

    st.markdown("**Pick genres you're in the mood for:**")

    selected = []
    rows = [GENRE_GRID[i:i+5] for i in range(0, len(GENRE_GRID), 5)]
    for row in rows:
        cols = st.columns(5)
        for j, g in enumerate(row):
            with cols[j]:
                if st.checkbox(g, key=f"gc_{g}"):
                    selected.append(g)

    col1, col2 = st.columns([1, 1])
    with col1:
        min_sc = st.slider("Minimum rating ★", 1.0, 10.0, 7.0, 0.5)
    with col2:
        n_g = st.selectbox("Results", [6, 12, 18], index=1)

    if st.button("🎯 Show Me Anime", use_container_width=False):
        if not selected:
            st.warning("Pick at least one genre above.")
        else:
            cb = get_cb(df)
            results = cb.recommend_by_genres(selected, n=n_g * 2, min_score=min_sc)
            results = _safe_merge(results, df).head(n_g)   # FIX 3
            if results.empty:
                st.markdown('<div class="nf-empty"><div class="nf-empty-icon">🎭</div><div class="nf-empty-text">Nothing found — try fewer genres or a lower rating.</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="nf-section-title" style="margin-top:24px;">Top {len(results)} picks for you</div>',
                            unsafe_allow_html=True)
                render_row(results, cols=6, key_prefix="genrerow")

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MY LIST (WATCHLIST)
# ═════════════════════════════════════════════════════════════════════════════

elif page == "watchlist":
    items = up.get_profile()["watchlist"]
    st.markdown(f"""
    <div class="nf-page-hero">
      <div class="nf-page-title">My List</div>
      <div class="nf-page-sub">{len(items)} anime saved to watch</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)

    if not items:
        st.markdown('<div class="nf-empty"><div class="nf-empty-icon">📺</div><div class="nf-empty-text">Your list is empty. Browse anime and hit "+ Watch" to add them here.</div></div>',
                    unsafe_allow_html=True)
    else:
        for item in items:
            render_list_item(item, "watchlist", key_prefix="wl")

        st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)
        st.markdown('<div class="nf-section-title">Because you saved these, you might like…</div>',
                    unsafe_allow_html=True)
        pick_name = items[0]["name"]
        try:
            cb  = get_cb(df)
            res = cb.recommend(pick_name, n=6)
            res = _safe_merge(res, df)   # FIX 3
            render_row(res, cols=6, key_prefix="wl_sim")
        except Exception:
            pass

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# FAVOURITES
# ═════════════════════════════════════════════════════════════════════════════

elif page == "favourites":
    items = up.get_profile()["favourites"]
    st.markdown(f"""
    <div class="nf-page-hero">
      <div class="nf-page-title">Favourites</div>
      <div class="nf-page-sub">{len(items)} anime you love</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="nf-empty"><div class="nf-empty-icon">♥</div><div class="nf-empty-text">No favourites yet. Hit ♡ on any anime card to add it here.</div></div>',
                    unsafe_allow_html=True)
    else:
        for item in items:
            render_list_item(item, "favourites", key_prefix="fav")
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SAVED
# ═════════════════════════════════════════════════════════════════════════════

elif page == "saved":
    items = up.get_profile()["saved"]
    st.markdown(f"""
    <div class="nf-page-hero">
      <div class="nf-page-title">Saved for Later</div>
      <div class="nf-page-sub">{len(items)} anime bookmarked</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="nf-empty"><div class="nf-empty-icon">🔖</div><div class="nf-empty-text">Nothing saved yet. Use the 🔖 button on any card.</div></div>',
                    unsafe_allow_html=True)
    else:
        for item in items:
            render_list_item(item, "saved", key_prefix="sv")
    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PROFILE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "profile":
    st.markdown("""
    <div class="nf-page-hero">
      <div class="nf-page-title">My Profile</div>
      <div class="nf-page-sub">Manage your account and preferences.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nf-section">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        initials = "".join(w[0].upper() for w in profile["username"].split()[:2])
        st.markdown(f"""
        <div class="nf-profile-card">
          <div class="nf-profile-avatar" style="background:{avatar_color};">{initials}</div>
          <div class="nf-profile-name">{profile["username"]}</div>
          <div class="nf-profile-since">Member since {profile["joined"]}</div>
          <div class="nf-stat-strip">
            <div class="nf-stat-box">
              <div class="nf-stat-n">{stats["watchlist"]}</div>
              <div class="nf-stat-l">Watchlist</div>
            </div>
            <div class="nf-stat-box">
              <div class="nf-stat-n">{stats["favourites"]}</div>
              <div class="nf-stat-l">Favourites</div>
            </div>
            <div class="nf-stat-box">
              <div class="nf-stat-n">{stats["saved"]}</div>
              <div class="nf-stat-l">Saved</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div style="background:#1f1f1f;border-radius:8px;padding:24px;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#fff;font-size:16px;font-weight:600;margin-bottom:16px;">Edit Profile</div>', unsafe_allow_html=True)

        new_name = st.text_input("Display name", value=profile["username"])
        color_options = {
            "Netflix Red":  "#e50914",
            "Purple":       "#6366f1",
            "Blue":         "#3b82f6",
            "Green":        "#10b981",
            "Orange":       "#f97316",
            "Pink":         "#ec4899",
        }
        current_color_name = next(
            (k for k, v in color_options.items() if v == profile.get("avatar_color")), "Netflix Red"
        )
        new_color_name = st.selectbox("Avatar colour", list(color_options.keys()),
                                      index=list(color_options.keys()).index(current_color_name))

        if st.button("Save Changes"):
            up.update_username(new_name)
            up.update_avatar_color(color_options[new_color_name])
            st.success("Saved!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Genre taste chart
    all_saved = (profile["watchlist"] + profile["favourites"] + profile["saved"])
    if all_saved:
        st.markdown('<hr class="nf-divider">', unsafe_allow_html=True)
        st.markdown('<div style="color:#fff;font-size:16px;font-weight:600;margin-bottom:16px;">Your Genre Taste</div>', unsafe_allow_html=True)

        # FIX 7 — Counter already imported at top of file
        gc: Counter = Counter()
        for item in all_saved:
            for g in str(item.get("genres", "")).split(","):
                g = g.strip()
                if g and g.lower() != "unknown":
                    gc[g] += 1

        if gc:
            top    = gc.most_common(8)
            labels = [t[0] for t in top]
            values = [t[1] for t in top]
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#1f1f1f")
            ax.set_facecolor("#1f1f1f")
            ax.barh(labels[::-1], values[::-1], color="#e50914", height=0.55)
            ax.tick_params(colors="#aaa")
            ax.set_xlabel("Count", color="#aaa")
            for spine in ax.spines.values():
                spine.set_color("#333")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 💬 ASK AI — NLP Chat Page
# ═════════════════════════════════════════════════════════════════════════════

elif page == "chat":

    st.markdown("""
    <div class="nf-page-hero">
      <div class="nf-page-title">💬 Ask the AI</div>
      <div class="nf-page-sub">Chat naturally — describe a mood, a genre, a plot, or an anime you love.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── initialise bot & chat history in session state ─────────────────────
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = AnimeChatBot(df)
    if "chat_messages" not in st.session_state:
        # seed with a greeting
        st.session_state.chat_messages = [
            {
                "role": "bot",
                "text": "Hey! 👋 I'm your AI anime guide. Tell me what you're in the mood for — a genre, a vibe, a plot, or an anime you already love!",
                "results": None,
            }
        ]

    bot: AnimeChatBot = st.session_state.chatbot

    # ── render conversation ────────────────────────────────────────────────
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        role      = msg["role"]
        text      = msg["text"]
        results   = msg.get("results")
        avatar    = "🎌" if role == "bot" else "👤"
        row_class = "bot" if role == "bot" else "user"

        # render bubble
        st.markdown(f"""
        <div class="chat-bubble-row {row_class}">
          <div class="chat-avatar {role}">{avatar}</div>
          <div class="chat-bubble {row_class}">{text}</div>
        </div>
        """, unsafe_allow_html=True)

        # render result cards inline in the chat
        if results is not None and not results.empty:
            icons = ["🎬","⚔️","🧠","🌊","🔥","🌸","👊","💀","🗡️","🤖"]
            cards_html = '<div class="chat-results">'
            for _, row in results.iterrows():
                r_name   = str(row.get("Name", ""))
                r_score  = _score(row.get("Score", 0))
                r_genres = _genres_short(row.get("Genres", ""), 2)
                r_img    = _img(row.get("Image URL", ""))
                r_id     = int(row.get("anime_id", 0))

                if r_img:
                    img_html = f'<img class="chat-card-img" src="{r_img}" loading="lazy">'
                else:
                    ico      = icons[r_id % len(icons)]
                    img_html = f'<div class="chat-card-img-ph">{ico}</div>'

                sc_str = f"★ {r_score:.1f}" if r_score else "N/A"
                cards_html += f"""
                <div class="chat-card">
                  {img_html}
                  <div class="chat-card-info">
                    <div class="chat-card-title" title="{r_name}">{r_name[:22]}{"…" if len(r_name)>22 else ""}</div>
                    <div class="chat-card-score">{sc_str}</div>
                    <div class="chat-card-genre">{r_genres}</div>
                  </div>
                </div>"""
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── fixed bottom input bar ─────────────────────────────────────────────
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    inp_col, btn_col, rst_col = st.columns([8, 1, 1])

    with inp_col:
        user_input = st.text_input(
            label="chat_input",
            label_visibility="collapsed",
            placeholder='Try: "something like Death Note" or "funny short anime" or "I want to cry"',
            key="chat_text_input",
        )
    with btn_col:
        send = st.button("Send ➤", key="chat_send", use_container_width=True)
    with rst_col:
        if st.button("Reset", key="chat_reset", use_container_width=True):
            bot.reset()
            st.session_state.chat_messages = [
                {
                    "role": "bot",
                    "text": "Fresh start! 🎌 What kind of anime are you looking for?",
                    "results": None,
                }
            ]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── process send ────────────────────────────────────────────────────────
    if send and user_input.strip():
        # add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "text": user_input.strip(),
            "results": None,
        })

        # get bot reply
        with st.spinner(""):
            reply, results = bot.chat(user_input.strip())

        # add bot reply
        st.session_state.chat_messages.append({
            "role": "bot",
            "text": reply,
            "results": results,
        })
        st.rerun()