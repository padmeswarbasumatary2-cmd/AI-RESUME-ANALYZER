# ResumeLens AI

A Streamlit resume analyzer trained on the labeled CSV bundled in `data/Resume.csv`.
It predicts likely job categories and compares an uploaded resume with a job
description.

## Run locally on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The model trains from the included `data/Resume.csv` automatically when the app
starts. It expects the `Resume_str` and `Category` columns.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub. This setup includes `data/Resume.csv` in the
   repository so the app no longer needs a training-CSV upload.
2. In Streamlit Community Cloud, create an app from that repository, select
   `app.py` as the main file, and deploy.
3. The app trains automatically from the bundled CSV. Upload only the individual
   PDF, TXT, or DOCX resume that you want to analyze.

**Privacy warning:** the raw training CSV contains real resumes and is committed
to this public GitHub repository. Anyone may download it. The source resume PDF
corpus is not included. Uploaded resumes are processed by Streamlit Cloud when the
hosted app is used.

## What it does

- Trains a TF-IDF + logistic-regression classifier on the bundled labeled resume text.
- The resume-analysis uploader accepts PDF, TXT, and DOCX resumes. Text extraction
  also supports other document formats internally, but those are not currently
  selectable in the app. OCR is enabled for scanned PDFs.
- Ranks the three best-fitting dataset categories.
- Optionally estimates job-description alignment from text similarity and recognized
  skill coverage, and highlights missing skills and keywords.
- On Streamlit Cloud, uploaded resumes are sent to the Streamlit app host for
  analysis; they are not sent to a separate AI service by this app.

The category prediction and match score are informational heuristics, not hiring
recommendations or a measure of candidate quality. On Streamlit Cloud, `packages.txt`
installs Tesseract OCR for scanned PDF documents. Local OCR requires Tesseract to be
installed separately.

## Tests

```powershell
python -m unittest discover -s tests
```
