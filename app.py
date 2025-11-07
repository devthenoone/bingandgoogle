#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
people_bio_streamlit.py — Streamlit UI for multi-keyword person search -> overlap links -> biography

Flow:
1) User enters a Person name (e.g., "Elysia Chan") and a comma-separated list of keywords (e.g., "McKinsey, Insurance, Accenture").
2) For each keyword, run a separate search ("person + keyword") on Bing (or Google) using Selenium.
3) Aggregate results (per-query CSV + combined CSV); display results in the app.
4) Identify overlapping links (appear in >1 query). Optionally verify via page titles/meta.
5) Show up to 3 profile candidates (image + name + LinkedIn when available); user selects.
6) Build a biography from verified/overlapping sources; Gemini if configured, else local fallback.

Requirements:
- Python 3.10+
- selenium, streamlit, pandas, beautifulsoup4, nltk
- (Optional) google-generativeai for Gemini
- Edge or Chrome installed (Selenium Manager fetches drivers automatically)
"""

import os
import re
import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse, quote_plus

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# NLTK (with safe ensure)
import nltk
from collections import Counter

# Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ---------------------------
# Utilities
# ---------------------------

def ensure_nltk():
    """Ensure NLTK resources; handle punkt_tab if present in newer NLTK."""
    needed = ["punkt", "stopwords"]
    try:
        import nltk.tokenize.punkt  # noqa
        needed.append("punkt_tab")
    except Exception:
        pass
    for pkg in needed:
        try:
            if pkg == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif pkg == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab/english/")
            else:
                nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def safe_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"\s+", "_", text)
    return text[:max_len] if len(text) > max_len else text


def make_driver(browser: str = "edge", headless: bool = True):
    """
    Create a Selenium WebDriver using Selenium Manager (no manual driver downloads).
    Supports 'edge' or 'chrome'.
    """
    browser = browser.lower().strip()
    if browser not in {"edge", "chrome"}:
        raise ValueError("browser must be 'edge' or 'chrome'")

    if browser == "edge":
        options = webdriver.EdgeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        return webdriver.Edge(options=options)
    else:
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        return webdriver.Chrome(options=options)


def parse_bing_results(driver):
    """Return list of {title, link, snippet} for current Bing SERP."""
    WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.ID, "b_results")))
    time.sleep(0.8)
    results = []
    cards = driver.find_elements(By.CSS_SELECTOR, "#b_results > li.b_algo, #b_results li[data-bm]")
    for item in cards:
        try:
            title, link, snippet = "", "", ""
            try:
                h2 = item.find_element(By.CSS_SELECTOR, "h2")
                title = (h2.text or "").strip()
                a = h2.find_element(By.CSS_SELECTOR, "a")
                link = (a.get_attribute("href") or "").strip()
            except NoSuchElementException:
                a = item.find_element(By.CSS_SELECTOR, "a")
                link = (a.get_attribute("href") or "").strip()
                title = (a.text or "").strip()

            snip_eles = item.find_elements(By.CSS_SELECTOR, ".b_caption p, .b_snippet, p, div.b_algoSlug")
            if snip_eles:
                snippet = (snip_eles[0].text or "").strip()

            if link:
                results.append({"title": title, "link": link, "snippet": snippet})
        except Exception:
            pass
    return results


def parse_google_results(driver):
    """Return list of {title, link, snippet} for current Google SERP (best-effort)."""
    WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#search")))
    time.sleep(0.8)
    results = []
    blocks = driver.find_elements(By.CSS_SELECTOR, "div#search div.g, div#search div[data-header-feature]")
    if not blocks:
        blocks = driver.find_elements(By.CSS_SELECTOR, "div#search a h3")
    for block in blocks:
        try:
            title, link, snippet = "", "", ""
            try:
                a = block.find_element(By.CSS_SELECTOR, "a")
                link = (a.get_attribute("href") or "").strip()
                h3 = block.find_element(By.CSS_SELECTOR, "h3")
                title = (h3.text or "").strip()
            except NoSuchElementException:
                try:
                    h3 = block.find_element(By.CSS_SELECTOR, "h3")
                    title = (h3.text or "").strip()
                    parent = h3.find_element(By.XPATH, "..")
                    link = (parent.get_attribute("href") or "").strip()
                except Exception:
                    pass

            for sel in ["div.VwiC3b", "span.aCOpRe", "div[data-sncf]"]:
                try:
                    s = block.find_element(By.CSS_SELECTOR, sel)
                    snippet = s.text.strip()
                    if snippet:
                        break
                except NoSuchElementException:
                    continue

            if link:
                results.append({"title": title, "link": link, "snippet": snippet})
        except Exception:
            pass
    return results


def run_search(query: str, engine: str = "bing", browser: str = "edge", headless: bool = True):
    """Run a single query on Bing or Google and return parsed results."""
    driver = make_driver(browser=browser, headless=headless)
    try:
        encoded = quote_plus(query)
        if engine == "google":
            driver.get(f"https://www.google.com/search?q={encoded}")
            results = parse_google_results(driver)
        else:
            driver.get(f"https://www.bing.com/search?q={encoded}")
            results = parse_bing_results(driver)
        return results
    finally:
        driver.quit()


def normalize_link(u: str) -> str:
    """Normalize URL for overlap grouping (domain + path w/o fragments/query)."""
    try:
        p = urlparse(u)
        path = re.sub(r"/+$", "", p.path or "/")
        return f"{p.scheme}://{p.netloc}{path}".lower()
    except Exception:
        return u.lower().strip()


def find_common_links(results_by_query: dict, min_appearances: int = 2):
    """
    results_by_query = { query: [ {title, link, snippet}, ... ], ... }
    returns list of dicts: {normalized_link, appearances, samples}
    """
    counter = {}
    samples = {}
    for q, rows in results_by_query.items():
        seen_for_q = set()
        for r in rows:
            nl = normalize_link(r["link"])
            if nl in seen_for_q:
                continue
            seen_for_q.add(nl)
            counter[nl] = counter.get(nl, 0) + 1
            samples.setdefault(nl, []).append({"query": q, **r})
    common = []
    for nl, c in counter.items():
        if c >= min_appearances:
            common.append({
                "normalized_link": nl,
                "appearances": c,
                "samples": samples.get(nl, [])[:5]
            })
    common.sort(key=lambda x: x["appearances"], reverse=True)
    return common


def harvest_page_text(url: str, browser: str = "edge", headless: bool = True, person: str = ""):
    """Fetch page with Selenium, return dict: {title, text, og_image, meta_desc, has_person_in_title}"""
    driver = make_driver(browser=browser, headless=headless)
    out = {"title": "", "text": "", "og_image": "", "meta_desc": "", "has_person_in_title": False}
    try:
        driver.set_page_load_timeout(25)
        driver.get(url)
        time.sleep(1.2)
        out["title"] = driver.title or ""
        html = driver.page_source or ""

        soup = BeautifulSoup(html, "html.parser")
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            out["og_image"] = og["content"]
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            out["meta_desc"] = md["content"]

        for s in soup(["script", "style", "noscript"]):
            s.extract()
        main_candidates = soup.find_all(["main", "article"])
        if main_candidates:
            text = main_candidates[0].get_text(separator="\n", strip=True)
        else:
            ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = "\n\n".join(ps)
        text = re.sub(r"\n{3,}", "\n\n", text)
        out["text"] = text[:200000]
        if person:
            out["has_person_in_title"] = bool(re.search(re.escape(person), out["title"], flags=re.I))
        return out
    except Exception:
        return out
    finally:
        driver.quit()


def get_top_keywords_from_texts(texts: list[str], person: str = "", companies: list[str] = None, top_n: int = 12):
    """Extract top keywords from texts; remove the person & company tokens."""
    ensure_nltk()
    from nltk.corpus import stopwords
    try:
        from nltk.tokenize import word_tokenize
        tokens = []
        stop = set(stopwords.words("english"))
        for t in texts:
            tokens.extend([w for w in word_tokenize(t.lower()) if w.isalnum() and w not in stop])
    except Exception:
        tokens = []
        common_stops = {"the", "and", "of", "to", "in", "for", "on", "with", "at", "by", "from"}
        for t in texts:
            tokens.extend([w for w in re.findall(r"[A-Za-z0-9]+", t.lower()) if w not in common_stops])

    freq = Counter(tokens)
    if person:
        for part in person.split():
            freq.pop(part.lower(), None)
    if companies:
        for c in companies:
            for part in c.split():
                freq.pop(part.lower(), None)

    return [w for w, _ in freq.most_common(top_n)]


def init_gemini(model_name: str = "gemini-2.5-flash", api_key: str | None = None):
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not available. Install google-generativeai.")
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def local_bio_from_chunks(person: str, company_hints: list[str], chunks: list[str]) -> str:
    """Small offline heuristic writer when Gemini isn't available."""
    text = "\n\n".join(chunks)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:40]
    intro = f"{person} appears to be a professional associated with {', '.join(company_hints)}." if company_hints else f"{person} appears to be a working professional."
    exp = " ".join(sentences[:6]) or "They have experience mentioned across multiple sources."
    edu = "Education details are not explicitly confirmed in the available sources."
    mentions = "Mentions and media references are drawn from overlapping links identified across searches."
    other = "Key strengths likely include domain knowledge, cross-functional collaboration, and an ability to operate across ambiguous contexts."

    return "\n".join([intro, edu, exp, mentions, other])


def gemini_bio_from_chunks(person: str, company_hints: list[str], chunks: list[str], model_name="gemini-2.5-flash", api_key=None) -> str:
    model = init_gemini(model_name, api_key)
    corpus = "\n\n".join(chunks)
    prompt = f"""
You are a professional biography writer. Using only the verifiable text below, write a concise 5-paragraph profile of {person}.
Rules:
- Paragraphs only (no headings/bullets), each 2–3 sentences.
- Be confident but do not fabricate specifics. If uncertain, hedge ("appears to", "reportedly").
- Prefer overlaps and repeated facts.
- We have hints that the person may be associated with: {", ".join(company_hints) if company_hints else "(no hints)"}.

Text:
---
{corpus[:15000]}
---
"""
    try:
        resp = model.generate_content(prompt.strip())
        return (getattr(resp, "text", "") or "").strip() or "[Empty response]"
    except Exception as e:
        return f"[Generation error: {e}]"


def guess_profile_candidates(all_rows: list[dict], person: str):
    """
    Guess up to 3 profile candidates:
    Prefer LinkedIn / people directories. Fetch og:image for card.
    Returns: [{name, url, image, source_title}]
    """
    pref = ["linkedin.com/in", "linkedin.com/pub", "crunchbase.com/person", "about", "team", "people"]
    ranked = []
    for r in all_rows:
        u = r.get("link", "")
        t = (r.get("title") or "").strip()
        if not u:
            continue
        score = 0
        u_l = u.lower()
        for i, key in enumerate(pref):
            if key in u_l:
                score += (100 - i * 10)
        if re.search(re.escape(person), t, flags=re.I):
            score += 20
        ranked.append((score, r))
    ranked.sort(key=lambda x: x[0], reverse=True)

    out = []
    seen = set()
    for _, r in ranked[:15]:
        url = r["link"]
        base = normalize_link(url)
        if base in seen:
            continue
        seen.add(base)
        meta = harvest_page_text(url, headless=True, person=person)
        name_guess = r["title"] or meta["title"] or person
        out.append({
            "name": name_guess.strip()[:120],
            "url": url,
            "image": meta["og_image"] or "",
            "source_title": r["title"]
        })
        if len(out) >= 3:
            break
    return out


# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Person Bio Finder", page_icon="🧭", layout="wide")
st.title("🔎 Person Bio Finder (multi-keyword overlap)")

# --- Session state initialization ---
if "search_done" not in st.session_state:
    st.session_state.search_done = False
    st.session_state.results_by_query = {}
    st.session_state.all_rows = []
    st.session_state.commons = []
    st.session_state.candidates = []
    st.session_state.person = ""
    st.session_state.keywords = []
    st.session_state.engine = "bing"
    st.session_state.browser = "edge"
    st.session_state.headless = True
    st.session_state.outdir = "output"
    st.session_state.gemini_model = "gemini-2.5-flash"
    st.session_state.gemini_key = ""

with st.sidebar:
    st.markdown("### Search Settings")
    engine = st.selectbox("Search engine", options=["bing", "google"], index=0)
    browser = st.selectbox("Browser (Selenium)", options=["edge", "chrome"], index=0)
    headless = st.checkbox("Headless", value=True)
    outdir = st.text_input("Output folder", value=st.session_state.outdir or "output")
    gemini_model = st.text_input("Gemini model (optional)", value=st.session_state.gemini_model or "gemini-2.5-flash")
    gemini_key = st.text_input("GEMINI_API_KEY (optional)", type="password", value=os.getenv("GEMINI_API_KEY", st.session_state.gemini_key))

col1, col2 = st.columns([2, 3])
with col1:
    person = st.text_input("Person name", value=st.session_state.person or "Elysia Chan")
    raw_keywords = st.text_input("Keywords (comma-separated)", value=", ".join(st.session_state.keywords) or "McKinsey, Insurance, Accenture")
with col2:
    st.markdown("Enter a person and 2–6 keywords. We’ll search each combo separately, find overlapping links, "
                "let you pick a likely profile, and then draft a bio from verifiable snippets.")

go = st.button("Search")

# --- SEARCH BUTTON HANDLER ---
if go:
    os.makedirs(outdir, exist_ok=True)
    keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
    if not person or not keywords:
        st.error("Please enter a person and at least one keyword.")
        st.stop()

    st.subheader("Step 1 — Run separate searches")
    results_by_query = {}
    all_rows = []

    with st.spinner("Running searches…"):
        prog = st.progress(0.0)
        for i, kw in enumerate(keywords, start=1):
            q = f"{person} {kw}"
            st.write(f"• Searching: **{q}** on {engine.title()} …")
            rows = run_search(q, engine=engine, browser=browser, headless=headless)
            for r in rows:
                r["query"] = q
            results_by_query[q] = rows
            all_rows.extend(rows)

            # write per-query csv
            pd.DataFrame(rows).to_csv(Path(outdir) / f"{safe_filename(q)}.csv", index=False)
            prog.progress(i / max(1, len(keywords)))

    combined_csv = Path(outdir) / "combined_results.csv"
    pd.DataFrame(all_rows).to_csv(combined_csv, index=False)
    st.success(f"Saved per-query CSVs and combined CSV → `{combined_csv}`")

    with st.expander("Preview combined results"):
        st.dataframe(pd.DataFrame(all_rows))

    st.subheader("Step 2 — Find overlaps across queries")
    commons = find_common_links(results_by_query, min_appearances=2 if len(keywords) > 1 else 1)
    if not commons:
        st.warning("No overlapping links found (appearing in multiple queries). Using strongest unique matches instead.")
        uniq = {}
        for r in all_rows:
            key = normalize_link(r["link"])
            if key not in uniq:
                uniq[key] = r
        commons = [{
            "normalized_link": k,
            "appearances": 1,
            "samples": [{"query": r["query"], **r}]
        } for k, r in list(uniq.items())[:5]]

    common_table = []
    for c in commons:
        sample = c["samples"][0]
        common_table.append({
            "Appearances": c["appearances"],
            "Link": c["normalized_link"],
            "Title": sample.get("title", ""),
            "From query": sample.get("query", "")
        })
    st.dataframe(pd.DataFrame(common_table))

    st.subheader("Step 3 — Pick a likely profile")
    candidates = guess_profile_candidates(all_rows, person)
    if not candidates:
        st.warning("Couldn’t auto-detect candidate profiles. You can still proceed.")
        candidates = []

    # Render candidate cards on this run
    if candidates:
        cols = st.columns(len(candidates))
        for i, cand in enumerate(candidates):
            with cols[i]:
                if cand.get("image"):
                    st.image(cand["image"], use_container_width=True)
                st.markdown(f"**{cand['name']}**")
                if "linkedin.com" in cand["url"].lower():
                    st.markdown(f"[LinkedIn profile]({cand['url']})")
                else:
                    st.markdown(f"[Open]({cand['url']})")

    # Persist to session_state for the Build Biography button rerun
    st.session_state.search_done = True
    st.session_state.results_by_query = results_by_query
    st.session_state.all_rows = all_rows
    st.session_state.commons = commons
    st.session_state.candidates = candidates
    st.session_state.person = person
    st.session_state.keywords = keywords
    st.session_state.engine = engine
    st.session_state.browser = browser
    st.session_state.headless = headless
    st.session_state.outdir = outdir
    st.session_state.gemini_model = gemini_model
    st.session_state.gemini_key = gemini_key

# --- POST-SEARCH UI (survives reruns) ---
if st.session_state.search_done:
    st.subheader("Step 3 — Pick a likely profile")
    candidates = st.session_state.candidates
    sel_idx = None
    if candidates:
        cols = st.columns(len(candidates))
        for i, cand in enumerate(candidates):
            with cols[i]:
                if cand.get("image"):
                    st.image(cand["image"], use_container_width=True)
                st.markdown(f"**{cand['name']}**")
                if "linkedin.com" in cand["url"].lower():
                    st.markdown(f"[LinkedIn profile]({cand['url']})")
                else:
                    st.markdown(f"[Open]({cand['url']})")
        sel_idx = st.radio(
            "Select profile",
            options=list(range(len(candidates))),
            index=0,
            format_func=lambda i: candidates[i]["name"]
        )

    proceed = st.button("Build Biography")

    if proceed:
        with st.spinner("Verifying overlap links and drafting biography…"):
            person = st.session_state.person
            keywords = st.session_state.keywords
            commons = st.session_state.commons
            all_rows = st.session_state.all_rows
            browser = st.session_state.browser
            headless = st.session_state.headless
            outdir = st.session_state.outdir
            gemini_model = st.session_state.gemini_model
            gemini_key = st.session_state.gemini_key

            # company hints
            companies_hint = [kw for kw in keywords if re.search(r"(mckinsey|accenture|deloitte|kpmg|pwc|ey)", kw, flags=re.I)] or keywords[:2]

            # harvest and verify
            harvest_chunks = []
            verified_cards = []
            for c in commons:
                url = c["normalized_link"]
                meta = harvest_page_text(url, browser=browser, headless=headless, person=person)
                title = (meta["title"] or "").lower()
                meta_desc = (meta["meta_desc"] or "").lower()
                hint_hit = any(h.lower() in title or h.lower() in meta_desc for h in companies_hint)
                if meta["has_person_in_title"] or hint_hit:
                    harvest_chunks.append("\n".join([
                        f"URL: {url}",
                        f"TITLE: {meta['title']}",
                        f"DESC: {meta['meta_desc']}",
                        meta["text"][:3000]
                    ]))
                    verified_cards.append({"url": url, "title": meta["title"], "desc": meta["meta_desc"]})

            with st.expander("Verified sources used for bio"):
                st.dataframe(pd.DataFrame(verified_cards))

            st.subheader("Step 5 — Extract keywords")
            descs = [r.get("snippet", "") for r in all_rows]
            top_kw = get_top_keywords_from_texts(harvest_chunks + descs, person=person, companies=companies_hint, top_n=12)
            st.write("Top keywords:", ", ".join(top_kw) if top_kw else "(none)")

            st.subheader("Step 6 — Draft biography")
            if GEMINI_AVAILABLE and (gemini_key or os.getenv("GEMINI_API_KEY")):
                bio = gemini_bio_from_chunks(person, companies_hint, harvest_chunks or descs, model_name=gemini_model, api_key=gemini_key or None)
            else:
                bio = local_bio_from_chunks(person, companies_hint, harvest_chunks or descs)

            st.text_area("Biography", value=bio, height=260)

            # Save outputs
            base = Path(outdir) / "bio_outputs"
            base.mkdir(parents=True, exist_ok=True)
            (base / f"{safe_filename(person)}_bio.txt").write_text(bio, encoding="utf-8")
            (base / f"{safe_filename(person)}_sources.json").write_text(json.dumps(verified_cards, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Saved bio and sources → {base}")

# Footer tip
st.caption("Tip: If Google blocks automated requests in your region, try Bing + Headless mode first.")
