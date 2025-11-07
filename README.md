# bingandgoogle — People Bio Streamlit App

This repository contains a Streamlit app that searches the web for a person across multiple keyword queries, finds overlapping links and builds a short biography from verified sources.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# or (cmd): .\.venv\Scripts\activate.bat
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes:
- The app uses Selenium to fetch search results and pages — Edge or Chrome should be installed on the host. Selenium Manager will attempt to download drivers automatically.
- If you want Gemini integration, install `google-generativeai` and set `GEMINI_API_KEY` in the environment (see `app.py`).

Preparing to push to GitHub
- Create a repo on GitHub (UI or `gh repo create`).
- Add it as a remote and push (examples below). Replace <OWNER> and <REPO>:

```powershell
git remote add origin https://github.com/<OWNER>/<REPO>.git
git branch -M main
git push -u origin main
# Or use the GitHub CLI:
# gh repo create <OWNER>/<REPO> --public --source=. --push
```

CI
- A minimal GitHub Actions workflow is included at `.github/workflows/python-app.yml` that installs dependencies and runs a small import smoke test.
<<<<<<< HEAD
# bingandgoogle
=======
# People Bio Streamlit App

This repository contains a Streamlit app that searches the web for a person across multiple keyword queries, finds overlapping links and builds a short biography from verified sources.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# or (cmd): .\.venv\Scripts\activate.bat
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes:
- The app uses Selenium to fetch search results and pages — Edge or Chrome should be installed on the host. Selenium Manager will attempt to download drivers automatically.
- If you want Gemini integration, install `google-generativeai` and set `GEMINI_API_KEY` in the environment (see `app.py`).

Preparing to push to GitHub
- Create a repo on GitHub (UI or `gh repo create`).
- Add it as a remote and push (examples below). Replace <OWNER> and <REPO>:

```powershell
git remote add origin https://github.com/<OWNER>/<REPO>.git
git branch -M main
git push -u origin main
# Or use the GitHub CLI:
# gh repo create <OWNER>/<REPO> --public --source=. --push
```

CI
- A minimal GitHub Actions workflow is included at `.github/workflows/python-app.yml` that installs dependencies and runs a small import smoke test.
>>>>>>> a6b03e0 (Initial commit: prepare repo for GitHub)
