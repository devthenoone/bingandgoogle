# people_bio_core.py
# Core helpers extracted from the Streamlit app (no Streamlit import)

import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse, quote_plus
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import nltk

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

def ensure_nltk():
    needed = ["punkt", "stopwords"]
    try:
        import nltk.tokenize.punkt
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

def make_driver(browser="edge", headless=True):
    browser = browser.lower().strip()
    if browser == "edge":
        opts = webdriver.EdgeOptions()
        if headless: opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu"); opts.add_argument("--disable-extensions")
        opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
        return webdriver.Edge(options=opts)
    elif browser == "chrome":
        opts = webdriver.ChromeOptions()
        if headless: opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu"); opts.add_argument("--disable-extensions")
        opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=opts)
    raise ValueError("browser must be edge|chrome")

def parse_bing_results(driver):
    WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.ID, "b_results")))
    time.sleep(0.8)
    results=[]
    cards = driver.find_elements(By.CSS_SELECTOR, "#b_results > li.b_algo, #b_results li[data-bm]")
    for item in cards:
        try:
            h2 = item.find_element(By.CSS_SELECTOR, "h2")
            a = h2.find_element(By.CSS_SELECTOR, "a")
            title = (h2.text or "").strip()
            link = (a.get_attribute("href") or "").strip()
            snip = ""
            for sel in [".b_caption p",".b_snippet","p","div.b_algoSlug"]:
                eles = item.find_elements(By.CSS_SELECTOR, sel)
                if eles: snip = (eles[0].text or "").strip(); break
            if link: results.append({"title": title, "link": link, "snippet": snip})
        except Exception:
            pass
    return results

def parse_google_results(driver):
    WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#search")))
    time.sleep(0.8)
    results=[]
    blocks = driver.find_elements(By.CSS_SELECTOR, "div#search div.g, div#search div[data-header-feature]")
    if not blocks: blocks = driver.find_elements(By.CSS_SELECTOR, "div#search a h3")
    for b in blocks:
        try:
            title, link, snippet = "", "", ""
            try:
                a = b.find_element(By.CSS_SELECTOR, "a"); link=(a.get_attribute("href") or "").strip()
                h3 = b.find_element(By.CSS_SELECTOR,"h3"); title=(h3.text or "").strip()
            except NoSuchElementException:
                try:
                    h3 = b.find_element(By.CSS_SELECTOR,"h3"); title=(h3.text or "").strip()
                    parent = h3.find_element(By.XPATH,".."); link=(parent.get_attribute("href") or "").strip()
                except Exception: pass
            for sel in ["div.VwiC3b","span.aCOpRe","div[data-sncf]"]:
                try:
                    s=b.find_element(By.CSS_SELECTOR,sel); snippet=s.text.strip(); break
                except NoSuchElementException: continue
            if link: results.append({"title": title, "link": link, "snippet": snippet})
        except Exception: pass
    return results

def run_search(query, engine="bing", browser="edge", headless=True):
    d = make_driver(browser=browser, headless=headless)
    try:
        encoded = quote_plus(query)
        if engine=="google":
            d.get(f"https://www.google.com/search?q={encoded}")
            return parse_google_results(d)
        else:
            d.get(f"https://www.bing.com/search?q={encoded}")
            return parse_bing_results(d)
    finally:
        d.quit()

def normalize_link(u: str) -> str:
    try:
        p = urlparse(u); path = re.sub(r"/+$","", p.path or "/")
        return f"{p.scheme}://{p.netloc}{path}".lower()
    except Exception:
        return u.lower().strip()

def find_common_links(results_by_query: dict, min_appearances=2):
    counter={}; samples={}
    for q, rows in results_by_query.items():
        seen=set()
        for r in rows:
            nl = normalize_link(r["link"])
            if nl in seen: continue
            seen.add(nl)
            counter[nl]=counter.get(nl,0)+1
            samples.setdefault(nl,[]).append({"query": q, **r})
    out=[]
    for nl, c in counter.items():
        if c>=min_appearances:
            out.append({"normalized_link": nl, "appearances": c, "samples": samples.get(nl,[])[:5]})
    out.sort(key=lambda x:x["appearances"], reverse=True)
    return out

def harvest_page_text(url, browser="edge", headless=True, person=""):
    d = make_driver(browser=browser, headless=headless)
    out={"title":"","text":"","og_image":"","meta_desc":"","has_person_in_title":False}
    try:
        d.set_page_load_timeout(25); d.get(url); time.sleep(1.2)
        out["title"]=d.title or ""
        soup = BeautifulSoup(d.page_source or "", "html.parser")
        og = soup.find("meta", property="og:image")
        if og and og.get("content"): out["og_image"]=og["content"]
        md = soup.find("meta", attrs={"name":"description"})
        if md and md.get("content"): out["meta_desc"]=md["content"]
        for s in soup(["script","style","noscript"]): s.extract()
        main = soup.find("main") or soup.find("article")
        if main: text = main.get_text("\n", strip=True)
        else:
            ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = "\n\n".join(ps)
        out["text"]=text[:200000]
        if person: out["has_person_in_title"]=bool(re.search(re.escape(person), out["title"], flags=re.I))
        return out
    finally:
        d.quit()

def get_top_keywords_from_texts(texts, person="", companies=None, top_n=12):
    ensure_nltk()
    from nltk.corpus import stopwords
    try:
        from nltk.tokenize import word_tokenize
        stop=set(stopwords.words("english")); toks=[]
        for t in texts: toks.extend([w for w in word_tokenize(t.lower()) if w.isalnum() and w not in stop])
    except Exception:
        toks=[]
        common={"the","and","of","to","in","for","on","with","at","by","from"}
        for t in texts: toks.extend([w for w in re.findall(r"[A-Za-z0-9]+", t.lower()) if w not in common])
    freq=Counter(toks)
    if person:
        for part in person.split(): freq.pop(part.lower(), None)
    if companies:
        for c in companies:
            for part in c.split(): freq.pop(part.lower(), None)
    return [w for w,_ in freq.most_common(top_n)]

def init_gemini(model_name="gemini-2.5-flash", api_key=None):
    if not GEMINI_AVAILABLE: raise RuntimeError("Gemini not available")
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)
