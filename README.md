# ResumeLens AI

A Streamlit resume analyzer trained on the labeled CSV datasets bundled in `data/`.
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

The model automatically loads bundled datasets when the app starts. It recognizes
`Resume_str`, `Resume`, or `Feature` as resume text columns, and `Category` or
`Label` as category columns. Comma-separated multi-category `Label` values are
expanded into individual category examples. The large Wahib dataset is bundled as
gzip-compressed CSV.

The five active CSV sources are `Resume.csv`, `jillani_resume_dataset.csv`,
`jithin_gpt_dataset.csv`, `noor_clean_resume_data.csv`, and `wahib_resume_data.csv.gz`.
For faster, more predictable startup on Streamlit Community Cloud, the classifier
trains on at most 500 examples per category while retaining all categories.
The separate 200,000-record structured candidate dataset is used to provide
aggregate hiring-rate charts from `data/hiring_outcomes_summary.json`; its
candidate-level rows are not included in the app. The app does not use that dataset
to predict an individual's hiring outcome because held-out evaluation found no
useful predictive signal.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub. The bundled training datasets are included in the
   repository so the app no longer needs a training-CSV upload.
2. In Streamlit Community Cloud, create an app from that repository, select
   `app.py` as the main file, and deploy.
3. The app trains automatically from the bundled CSV. Upload only the individual
   PDF, TXT, or DOCX resume that you want to analyze.

**Privacy warning:** the bundled training datasets contain resume information and
are committed to this public GitHub repository. Anyone may download them. The
Snehaan Bhawal CSV supplied for this update is byte-identical to the already
bundled `data/Resume.csv`, so it is not stored a second time. Uploaded resumes are
processed by Streamlit Cloud when the hosted app is used.

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
