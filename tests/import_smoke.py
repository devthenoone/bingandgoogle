import sys

failures = []

try:
    import pandas
except Exception as e:
    failures.append(f"pandas: {e}")

try:
    import streamlit
except Exception as e:
    failures.append(f"streamlit: {e}")

try:
    from bs4 import BeautifulSoup
except Exception as e:
    failures.append(f"beautifulsoup4: {e}")

try:
    import selenium
except Exception as e:
    failures.append(f"selenium: {e}")

try:
    import nltk
except Exception as e:
    failures.append(f"nltk: {e}")

if failures:
    print("IMPORT FAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)

print("IMPORT CHECK OK")
