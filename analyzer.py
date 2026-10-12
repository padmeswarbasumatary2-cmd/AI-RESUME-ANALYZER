"""Dataset loading, resume parsing, and machine-learning analysis helpers."""

from __future__ import annotations

import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pandas as pd
from docx import Document
from html.parser import HTMLParser
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import pymupdf
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from striprtf.striprtf import rtf_to_text


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Accounting": ("accounting", "bookkeeping", "quickbooks"),
    "Agile": ("agile", "scrum"),
    "Agriculture": ("agriculture", "crop management", "horticulture"),
    "Adobe Creative Suite": ("adobe photoshop", "photoshop", "illustrator", "indesign"),
    "Aviation": ("aviation", "flight operations", "aviation safety"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "Banking": ("banking",),
    "Budgeting": ("budgeting", "budget management"),
    "Business Development": ("business development",),
    "CAD": ("cad", "autocad", "revit"),
    "Clinical Care": ("clinical", "patient care", "nursing"),
    "Communication": ("communication", "written communication", "verbal communication"),
    "Compliance": ("compliance", "hipaa", "regulatory compliance"),
    "Construction": ("construction", "blueprint reading", "osha"),
    "Content Writing": ("content writing", "copywriting", "technical writing"),
    "Customer Service": ("customer service", "customer support"),
    "Data Analysis": ("data analysis", "data analytics", "statistical analysis"),
    "Data Visualization": ("data visualization", "tableau", "power bi"),
    "Database Management": ("database management", "mysql", "postgresql"),
    "Digital Marketing": ("digital marketing", "seo", "sem", "social media"),
    "Docker": ("docker",),
    "Education": ("teaching", "curriculum development", "lesson planning"),
    "Electrical Engineering": ("electrical engineering",),
    "Employee Relations": ("employee relations", "human resources"),
    "Engineering": ("engineering", "mechanical engineering", "civil engineering"),
    "Event Planning": ("event planning", "event management"),
    "Excel": ("excel", "microsoft excel", "spreadsheet"),
    "Financial Analysis": ("financial analysis", "financial modeling", "forecasting"),
    "Fitness Training": ("fitness", "personal training", "exercise science"),
    "Food Safety": ("food safety", "food preparation", "menu planning"),
    "Git": ("git", "github"),
    "Graphic Design": ("graphic design", "figma", "ui/ux", "user experience"),
    "Inventory Management": ("inventory management", "inventory control"),
    "Java": ("java",),
    "JavaScript": ("javascript", "react", "node.js", "nodejs"),
    "Jira": ("jira",),
    "Leadership": ("leadership", "team management", "team leadership"),
    "Logistics": ("logistics", "supply chain"),
    "Machine Learning": ("machine learning", "deep learning", "artificial intelligence"),
    "Marketing": ("marketing", "market research", "brand management"),
    "Microsoft Office": ("microsoft office", "word", "powerpoint"),
    "Negotiation": ("negotiation", "negotiating"),
    "Operations": ("operations", "process improvement", "operations management"),
    "Payroll": ("payroll",),
    "Project Management": ("project management", "project planning"),
    "Python": ("python",),
    "Quality Assurance": ("quality assurance", "quality control", "testing"),
    "Recruitment": ("recruitment", "recruiting", "talent acquisition"),
    "Research": ("research", "literature review"),
    "Risk Management": ("risk management", "risk assessment"),
    "SQL": ("sql",),
    "Sales": ("sales", "salesforce", "crm"),
    "Safety": ("safety", "workplace safety"),
    "Tableau": ("tableau",),
    "Tax Preparation": ("tax preparation", "tax compliance"),
    "Training": ("training", "staff development"),
    "Troubleshooting": ("troubleshooting", "technical support"),
    "TypeScript": ("typescript",),
    "UI/UX": ("ui/ux", "user interface", "user experience"),
    "Video Editing": ("video editing", "premiere pro"),
    "Web Development": ("web development", "html", "css", "rest api"),
}


def load_training_data(
    csv_path: str | Path | None = None,
    pdf_root: str | Path | None = None,
    csv_bytes: bytes | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load labeled CSV resumes and, when available, resumes in category PDF folders."""
    if csv_bytes is not None:
        csv_source = BytesIO(csv_bytes)
    elif csv_path is not None:
        csv_file = Path(csv_path).expanduser()
        if not csv_file.is_file():
            raise FileNotFoundError(f"Resume CSV not found: {csv_file}")
        csv_source = csv_file
    else:
        raise ValueError("Provide a CSV path or upload a CSV file.")

    csv_data = pd.read_csv(csv_source, usecols=["Resume_str", "Category"], encoding_errors="replace")
    csv_data = csv_data.rename(columns={"Resume_str": "text"})
    csv_data["text"] = csv_data["text"].fillna("").astype(str).str.strip()
    csv_data["category"] = csv_data["Category"].fillna("").astype(str).str.strip()
    csv_data = csv_data.loc[
        csv_data["text"].ne("") & csv_data["category"].ne(""),
        ["text", "category"],
    ]

    pdf_documents: list[dict[str, str]] = []
    pdf_failures = 0
    if pdf_root:
        root = Path(pdf_root).expanduser()
        if root.is_dir():
            for pdf_path in root.rglob("*.pdf"):
                category = pdf_path.parent.name.strip()
                try:
                    reader = PdfReader(pdf_path)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                except (OSError, ValueError, PdfReadError):
                    pdf_failures += 1
                    continue
                if category and text:
                    pdf_documents.append({"text": text, "category": category})

    pdf_data = pd.DataFrame(pdf_documents, columns=["text", "category"])
    combined = pd.concat([csv_data, pdf_data], ignore_index=True)
    stats = {
        "csv_resumes": len(csv_data),
        "pdf_resumes": len(pdf_data),
        "pdf_failures": pdf_failures,
        "categories": combined["category"].nunique(),
    }
    return combined, stats


def train_classifier(training_data: pd.DataFrame) -> Pipeline:
    """Fit a TF-IDF and logistic-regression classifier on the labeled resumes."""
    if training_data.empty:
        raise ValueError("No usable labeled resumes were found in the selected datasets.")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=30,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(training_data["text"].tolist(), training_data["category"].tolist())
    return model


class _VisibleHTMLText(HTMLParser):
    """Collect visible text from HTML while ignoring script and style contents."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._ignored_tags.append(tag.lower())

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_tags and tag.lower() == self._ignored_tags[-1]:
            self._ignored_tags.pop()

    def handle_data(self, data: str) -> None:
        if not self._ignored_tags:
            self.parts.append(data)


def _extract_ocr_image(image: Image.Image, filename: str) -> str:
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as error:
        raise ValueError(
            f"OCR is not available for {filename}. Try uploading a text-based PDF or DOCX instead."
        ) from error


def _extract_doc_text(content: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            result = subprocess.run(
                ["antiword", temporary_file.name],
                check=True,
                capture_output=True,
                timeout=30,
            )
    except FileNotFoundError as error:
        raise ValueError(
            "Legacy .doc files need a converter. Save the resume as .docx or PDF and upload it again."
        ) from error
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("The .doc file could not be converted. Try saving it as .docx or PDF.") from error
    return result.stdout.decode("utf-8", errors="replace")


def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract text or OCR from common document, image, and text upload formats."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".text", ".md", ".markdown", ".csv", ".json"}:
        text = content.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except PdfReadError as error:
            raise ValueError("The PDF could not be read. Try exporting it again.") from error
        if not text.strip():
            try:
                with pymupdf.open(stream=content, filetype="pdf") as document:
                    if len(document) > 30:
                        raise ValueError(
                            "This scanned PDF has more than 30 pages. Run OCR on it or split it into smaller files."
                        )
                    extracted_pages = []
                    for page in document:
                        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                        extracted_pages.append(_extract_ocr_image(image, filename))
                    text = "\n".join(extracted_pages)
            except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
                raise ValueError("The PDF could not be read. Try exporting it again.") from error
    elif suffix == ".docx":
        try:
            document = Document(BytesIO(content))
        except BadZipFile as error:
            raise ValueError("The DOCX file could not be read. Try exporting it again.") from error
        parts = [paragraph.text for paragraph in document.paragraphs]
        parts.extend(
            cell.text
            for table in document.tables
            for row in table.rows
            for cell in row.cells
        )
        text = "\n".join(parts)
    elif suffix == ".doc":
        text = _extract_doc_text(content)
    elif suffix == ".rtf":
        text = rtf_to_text(content.decode("utf-8", errors="replace"))
    elif suffix == ".odt":
        try:
            with ZipFile(BytesIO(content)) as document:
                root = ElementTree.fromstring(document.read("content.xml"))
        except (BadZipFile, KeyError, ElementTree.ParseError) as error:
            raise ValueError("The ODT file could not be read. Try exporting it again.") from error
        text = "\n".join(
            "".join(element.itertext())
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] in {"p", "h"}
        )
    elif suffix in {".html", ".htm"}:
        parser = _VisibleHTMLText()
        parser.feed(content.decode("utf-8", errors="replace"))
        text = " ".join(parser.parts)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        try:
            with Image.open(BytesIO(content)) as image:
                text = _extract_ocr_image(image, filename)
        except UnidentifiedImageError as error:
            raise ValueError("The image could not be opened. Try uploading a PNG or JPEG image.") from error
    elif suffix in {".xlsx", ".xls"}:
        try:
            workbook = pd.read_excel(BytesIO(content), header=None, dtype=str)
        except (ImportError, ValueError, OSError) as error:
            raise ValueError("The spreadsheet could not be read. Try exporting it as .xlsx or PDF.") from error
        text = "\n".join(
            " ".join(str(value) for value in row if pd.notna(value))
            for row in workbook.itertuples(index=False, name=None)
        )
    else:
        raise ValueError(
            f"The {suffix or 'unknown'} file type is not supported for text extraction. "
            "Try PDF, DOC/DOCX, ODT, RTF, TXT, HTML, CSV, XLS/XLSX, or an image file."
        )

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("No readable text was found. Check that the file contains text and is not blank.")
    return text


def extract_skills(text: str) -> set[str]:
    """Return skill labels found using case-insensitive whole-phrase matching."""
    normalized = text.casefold()
    found: set[str] = set()
    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?<!\w){re.escape(alias.casefold())}(?!\w)"
            if re.search(pattern, normalized):
                found.add(skill)
                break
    return found


def analyze_resume(
    model: Pipeline,
    resume_text: str,
    job_description: str = "",
) -> dict[str, object]:
    """Predict suitable dataset categories and optionally score a job-description match."""
    classifier = model.named_steps["classifier"]
    vectorizer = model.named_steps["tfidf"]
    probabilities = model.predict_proba([resume_text])[0]
    ranked_indices = probabilities.argsort()[::-1][:3]
    recommendations = [
        {
            "category": str(classifier.classes_[index]),
            "confidence": float(probabilities[index]),
        }
        for index in ranked_indices
    ]

    resume_skills = extract_skills(resume_text)
    result: dict[str, object] = {
        "recommendations": recommendations,
        "resume_skills": sorted(resume_skills),
    }

    job_description = job_description.strip()
    if not job_description:
        return result

    job_skills = extract_skills(job_description)
    matched_skills = sorted(resume_skills & job_skills)
    missing_skills = sorted(job_skills - resume_skills)

    resume_vector = vectorizer.transform([resume_text])
    job_vector = vectorizer.transform([job_description])
    similarity = float(cosine_similarity(resume_vector, job_vector)[0, 0])

    if job_skills:
        coverage = len(matched_skills) / len(job_skills)
        alignment = 100 * (0.75 * coverage + 0.25 * similarity)
    else:
        coverage = None
        alignment = 100 * similarity

    feature_names = vectorizer.get_feature_names_out()
    job_weights = job_vector.toarray()[0]
    resume_terms = set(feature_names[resume_vector.toarray()[0] > 0])
    ranked_terms = job_weights.argsort()[::-1]
    missing_keywords: list[str] = []
    for index in ranked_terms:
        term = str(feature_names[index])
        if job_weights[index] <= 0 or term in resume_terms:
            continue
        if any(term in keyword or keyword in term for keyword in missing_keywords):
            continue
        missing_keywords.append(term)
        if len(missing_keywords) == 8:
            break

    result.update(
        {
            "job_skills": sorted(job_skills),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_coverage": coverage,
            "text_similarity": similarity,
            "alignment": min(100.0, max(0.0, alignment)),
            "missing_keywords": missing_keywords,
        }
    )
    return result
