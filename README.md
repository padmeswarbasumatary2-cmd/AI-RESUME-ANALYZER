# ResumeLens AI

A local resume analyzer trained on the supplied resume datasets. It predicts likely
job categories from the labeled CSV and category-folder PDFs, and can compare an
uploaded resume with a job description.

## Run locally on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser and uses the CSV from the provided archive:

- `C:\Users\radhi\Downloads\Mobile Devices\archive\Resume\Resume.csv`

The app sidebar can optionally include the PDF corpus from
`C:\Users\radhi\Downloads\Mobile Devices\archive\data\data`. Reading and extracting
all PDFs can take several minutes, so this option is off by default. Change the CSV
or PDF path in the sidebar, or set `RESUME_DATASET_PATH` and
`RESUME_PDF_DATASET_PATH` before launching Streamlit to use a different copy. The CSV
needs `Resume_str` and `Category` columns. The PDF folder should contain one
category-named subfolder per job family.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Do **not** add the resume CSV, PDF
   corpus, or other personal resume data to GitHub; `.gitignore` excludes common
   dataset files and local environment files.
2. In Streamlit Community Cloud, create an app from that repository, select
   `app.py` as the main file, and deploy.
3. Open the deployed app and upload the labeled CSV in the **Your dataset** sidebar.
   It must include `Resume_str` and `Category` columns. The app trains from that
   upload for the session; re-upload it if the app restarts.

The deployed app cannot read files from your computer's `C:\...` paths. Uploading
the training CSV sends it to the Streamlit-hosted app for analysis. Uploaded
training data is not committed to GitHub by this workflow. Only deploy resume
datasets to a Streamlit app whose access and data handling you are comfortable
with.

## What it does

- Trains a TF-IDF + logistic-regression classifier on labeled resume text.
- Extracts text from PDF, Word (`.doc`/`.docx`), OpenDocument (`.odt`), RTF, TXT,
  Markdown, HTML, CSV/JSON, Excel spreadsheets, and common image formats. OCR is
  enabled for images and scanned PDFs.
- Ranks the three best-fitting dataset categories.
- Optionally estimates job-description alignment from text similarity and recognized
  skill coverage, and highlights missing skills and keywords.
- When run locally, analyzes uploaded resumes on your computer. On Streamlit Cloud,
  uploaded resumes and training data are sent to the Streamlit app host for analysis;
  they are not sent to a separate AI service by this app.

The category prediction and match score are informational heuristics, not hiring
recommendations or a measure of candidate quality. The uploader accepts any
extension, but text extraction is supported for the formats listed above; convert
other formats to PDF or DOCX first. On Streamlit Cloud, `packages.txt` installs
Tesseract OCR and Antiword for scanned files and legacy `.doc` documents. Local
OCR and `.doc` conversion require Tesseract and Antiword to be installed separately.

## Tests

```powershell
python -m unittest discover -s tests
```
