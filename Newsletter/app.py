import streamlit as st
from datetime import datetime
import os, requests, json
from dotenv import load_dotenv
from typing import List, Dict, Any

# -----------------------------
# Google Gemini Flash client
# -----------------------------
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# -----------------------------
# Load API keys
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY")
MEDIASTACK_KEY = os.getenv("MEDIASTACK_KEY")
WEBZIO_KEY     = os.getenv("WEBZIO_KEY")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")

if GOOGLE_API_KEY and HAS_GENAI:
    genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# Keywords
# -----------------------------
KEYWORDS = [
    "AI","ML","machine learning","deep learning","NLP","LLM",
    "neural networks","transformer","reinforcement learning",
    "generative","computer vision","chatgpt","openai",
    "pytorch","tensorflow"
]

def filter_articles(articles: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    filtered = []
    for a in articles:
        title = (a.get("title") or "").lower()
        if any(kw.lower() in title for kw in KEYWORDS):
            filtered.append(a)
    return filtered

# -----------------------------
# News sources
# -----------------------------
def fetch_newsapi():
    url = "https://newsapi.org/v2/top-headlines"
    r = requests.get(url, params={
        "language":"en","pageSize":50,"apiKey":NEWSAPI_KEY
    }, timeout=15)
    r.raise_for_status()
    data = r.json().get("articles", [])
    return filter_articles([{
        "title":a.get("title"),
        "desc":a.get("description"),
        "url":a.get("url"),
        "img":a.get("urlToImage"),
        "date":a.get("publishedAt")
    } for a in data])

def fetch_mediastack():
    url = "http://api.mediastack.com/v1/news"
    r = requests.get(url, params={
        "access_key":MEDIASTACK_KEY,"languages":"en","limit":50
    }, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    return filter_articles([{
        "title":a.get("title"),
        "desc":a.get("description"),
        "url":a.get("url"),
        "img":a.get("image"),
        "date":a.get("published_at")
    } for a in data])

def fetch_webzio():
    url = "https://webz.io/api/news_api/v1"
    r = requests.get(url, params={"token":WEBZIO_KEY,"size":50}, timeout=15)
    r.raise_for_status()
    data = r.json().get("articles", [])
    return filter_articles([{
        "title":a.get("title"),
        "desc":a.get("text"),
        "url":a.get("url"),
        "img":a.get("image"),
        "date":a.get("published")
    } for a in data])

def get_all_news():
    all_articles = []
    for fn, name in [(fetch_newsapi,"NewsAPI"),
                     (fetch_mediastack,"MediaStack"),
                     (fetch_webzio,"WebZio")]:
        try:
            all_articles.extend(fn())
        except Exception as e:
            print(f"{name} failed:", e)
    return sorted(all_articles, key=lambda x: x.get("date") or "", reverse=True)

# -----------------------------
# GitHub papers
# -----------------------------
def fetch_github_papers(n=5):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = "https://api.github.com/search/repositories"
    q = "AI OR machine-learning OR deep-learning in:readme,description"
    r = requests.get(url, headers=headers,
                     params={"q":q,"sort":"updated","order":"desc","per_page":n},
                     timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    return [{
        "title": it["name"],
        "publisher": it["owner"]["login"],
        "desc": it.get("description"),
        "url": it["html_url"],
        "date": it["updated_at"]
    } for it in items]

# -----------------------------
# Format newsletter with Gemini Flash
# -----------------------------
def format_news_with_model(articles, papers, min_articles=3, max_articles=6):
    month_year = datetime.now().strftime("%B %Y")
    prompt = f"""
Format this into a clean HTML newsletter titled:
AI Generated News {month_year}

Constraints:
- Use {min_articles}–{max_articles} news articles.
- For each news article: Title, Image, 2-3 bullet description, "Read more" link.
- After that add a Research Papers section with Title, Publisher, Date, short description, link.
- Use only provided content.
- Return clean, mobile-friendly HTML.
"""
    content = {"articles": articles, "papers": papers}

    if GOOGLE_API_KEY and HAS_GENAI:
        resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(
            [prompt, json.dumps(content)],
            generation_config={"temperature":0.3}
        )
        html_content = resp.text
    else:
        raise RuntimeError("No Google LLM configured or API key missing")

    # Wrap HTML with styling for centered, uniform images
    html_page = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .article, .paper {{ margin-bottom: 40px; }}
            .article img, .paper img {{
                display: block;
                margin: 0 auto;
                width: 100%;
                max-width: 600px;
                height: auto;
                object-fit: cover;
                border-radius: 8px;
            }}
            .article-title, .paper-title {{ font-size: 1.2em; font-weight: bold; margin: 10px 0; }}
            .article-desc, .paper-desc {{ margin: 5px 0 10px 0; }}
            .read-more {{ color: #1a0dab; text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    return html_page

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Newsletter", layout="centered")
st.title("📰 AI Generated Newsletter")

min_articles = 3
max_articles = 10

if st.button("Fetch & Generate"):
    with st.spinner("Fetching data..."):
        articles = get_all_news()
        papers   = fetch_github_papers()

    st.write(f"Found {len(articles)} matching articles, {len(papers)} repos.")

    with st.spinner("Generating newsletter..."):
        html_page = format_news_with_model(
            articles, papers,
            min_articles=min_articles,
            max_articles=max_articles
        )

    st.subheader("HTML Preview")
    st.components.v1.html(html_page, height=800, scrolling=True)

    # Save newsletter locally
    outdir = "newsletters"
    os.makedirs(outdir, exist_ok=True)
    fname = f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fpath = os.path.join(outdir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html_page)

    # Download button
    st.download_button("Download HTML", data=html_page, file_name=fname, mime="text/html")
