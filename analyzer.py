"""Dataset loading, resume parsing, and machine-learning analysis helpers."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from docx import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline


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


def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract text from supported upload formats without storing the upload."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except PdfReadError as error:
            raise ValueError("The PDF could not be read. Try exporting it again as a text-based PDF.") from error
    elif suffix == ".docx":
        try:
            document = Document(BytesIO(content))
        except BadZipFile as error:
            raise ValueError("The DOCX file could not be read. Try exporting it again.") from error
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        raise ValueError("Upload a PDF, DOCX, or TXT resume.")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError(
            "No readable text was found. Scanned PDFs need OCR before they can be analyzed."
        )
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
