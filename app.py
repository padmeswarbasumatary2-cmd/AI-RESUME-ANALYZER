from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import analyze_resume, extract_resume_text, load_training_data, train_classifier


DEFAULT_CSV_PATH = Path(
    os.environ.get(
        "RESUME_DATASET_PATH",
        r"C:\Users\radhi\Downloads\Mobile Devices\archive\Resume\Resume.csv",
    )
)
DEFAULT_PDF_PATH = Path(
    os.environ.get(
        "RESUME_PDF_DATASET_PATH",
        r"C:\Users\radhi\Downloads\Mobile Devices\archive\data\data",
    )
)
MAX_RESUME_BYTES = 20 * 1024 * 1024

st.set_page_config(
    page_title="ResumeLens AI",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary-color: #91b5ff;
        --ink: #e6eaf1;
        --muted: #a0a9b8;
        --line: #343b48;
        --blue: #91b5ff;
        --pale-blue: #283246;
        --surface: #222833;
        --canvas: #191c23;
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(25, 28, 35, .94); }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label { color: var(--ink); }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--muted); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #20242d 0%, #191c23 100%);
        border-right: 1px solid #303642;
    }
    [data-testid="stSidebar"] * { color: #e6eaf1; }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #252b35;
        border-color: #414957;
        color: #e6eaf1;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label p { color: #e6eaf1; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a0a9b8; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.3rem 2.6rem;
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 24px;
        background:
            radial-gradient(ellipse at 90% 5%, rgba(117, 151, 207, .22), transparent 34%),
            linear-gradient(120deg, #202d42 0%, #2a3e5e 62%, #354e73 100%);
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 44px rgba(0, 0, 0, .2);
    }
    .hero .eyebrow {
        color: #c4d4ed;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .hero h1 { color: #f2f4f8; margin: 0 0 .55rem 0; font-size: clamp(2rem, 4vw, 2.7rem); letter-spacing: -.04em; }
    .hero p { color: #d4dbe6; margin: 0; font-size: 1.04rem; line-height: 1.65; max-width: 650px; }
    .dataset-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: .65rem;
        padding: .8rem 1rem;
        margin: 0 0 1.4rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        color: var(--muted);
        font-size: .88rem;
    }
    .dataset-strip strong { color: var(--ink); }
    .dataset-dot {
        width: 8px; height: 8px; display: inline-block;
        border-radius: 50%; background: #31a879; margin-right: .25rem;
    }
    .section-title { color: var(--ink); font-weight: 750; letter-spacing: -.02em; }
    [data-testid="stTabs"] [role="tablist"] { gap: .4rem; border-bottom-color: var(--line); }
    [data-testid="stTabs"] button[role="tab"] { color: var(--muted); font-weight: 650; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--blue);
        border-bottom-color: var(--blue);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {
        background-color: var(--blue);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #596577;
        border-radius: 16px;
        background: #20252e;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #7596d1; background: #252c38; }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * { color: var(--ink); }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        border: 1px solid #414957;
        border-radius: 12px;
        background: #222833;
        color: var(--ink);
    }
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder { color: #858f9f; }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: 0 4px 14px rgba(0, 0, 0, .12);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.55rem; font-weight: 750; }
    .prediction-card {
        padding: .85rem 1rem;
        margin: .6rem 0;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
    }
    .prediction-top {
        display: flex; justify-content: space-between; align-items: center;
        gap: .8rem; margin-bottom: .6rem;
    }
    .prediction-name { color: var(--ink); font-size: .96rem; font-weight: 700; }
    .prediction-rank { color: var(--muted); font-size: .78rem; margin-right: .5rem; }
    .prediction-score { color: var(--blue); font-size: .88rem; font-weight: 750; }
    .prediction-track { height: 7px; border-radius: 8px; background: #343b48; overflow: hidden; }
    .prediction-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #6889c0, #91b5ff); }
    [data-testid="stProgress"] > div { background: #343b48; }
    [data-testid="stProgress"] [role="progressbar"] { background: #789bd4; }
    .skill-label {
        display: inline-block; padding: .38rem .68rem; margin: .2rem .22rem .2rem 0;
        border: 1px solid #3a465a; border-radius: 999px;
        background: var(--pale-blue); color: #c3d5f5; font-size: .82rem; font-weight: 600;
    }
    .skill-label.missing { background: #342b22; border-color: #58442c; color: #e5bd82; }
    .skill-label.matched { background: #22332d; border-color: #365344; color: #a8d5bc; }
    .result-heading { margin-bottom: .1rem; color: var(--ink); font-size: 1.45rem; font-weight: 750; }
    .result-subtitle { margin: 0 0 1rem; color: var(--muted); font-size: .9rem; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(110deg, #425f91, #5276b4);
        border: 1px solid #6787bd; border-radius: 12px;
        min-height: 3rem; font-weight: 700;
        color: #f4f6fa;
        box-shadow: 0 6px 16px rgba(0, 0, 0, .2);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 9px 20px rgba(0, 0, 0, .3);
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .85rem 2.5rem; }
        .hero { padding: 1.6rem 1.35rem; border-radius: 19px; }
        .dataset-strip { gap: .4rem; font-size: .8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(csv_path: str, pdf_path: str, csv_bytes: bytes | None = None):
    training_data, data_stats = load_training_data(
        csv_path or None,
        pdf_path or None,
        csv_bytes=csv_bytes,
    )
    model = train_classifier(training_data)
    return model, data_stats


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Career insights, made clear</div>
      <h1>ResumeLens AI</h1>
      <p>Understand your resume, discover matching career paths, and tailor it to a job.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Your dataset")
    csv_path = st.text_input("Local labeled resume CSV", str(DEFAULT_CSV_PATH))
    uploaded_dataset = st.file_uploader(
        "Or upload the labeled CSV",
        type=["csv"],
        max_upload_size=100,
        help="Required on Streamlit Cloud. The CSV must include Resume_str and Category columns.",
    )
    include_pdfs = st.checkbox(
        "Include category PDF corpus",
        value=False,
        help="Reading and extracting text from every PDF can add several minutes to the initial setup.",
    )
    pdf_path = ""
    if include_pdfs:
        pdf_path = st.text_input("Category PDF folder", str(DEFAULT_PDF_PATH))
    st.caption(
        "On Streamlit Cloud, upload the labeled CSV here; it is sent to the app host for training. "
        "Do not commit resume datasets to GitHub. PDF training remains optional because extracting "
        "every file can take several minutes."
    )
    st.markdown("---")
    st.markdown("**Next: analyze a resume**")
    st.caption(
        "After the training CSV loads, open the Analyze a resume tab in the main panel. "
        "Its resume uploader accepts PDF, TXT, and DOCX files."
    )

dataset_bytes = uploaded_dataset.getvalue() if uploaded_dataset is not None else None
local_csv_path = csv_path.strip()
if dataset_bytes is None and not Path(local_csv_path).expanduser().is_file():
    st.info(
        "**First, add the labeled training dataset:** upload the archive's CSV in the sidebar. "
        "That CSV trains the career-category model. After it loads, use the resume uploader "
        "here in the main panel to analyze an individual PDF, TXT, or DOCX resume."
    )
    st.stop()

try:
    with st.spinner("Learning from your resume dataset..."):
        model, data_stats = load_model(
            local_csv_path if dataset_bytes is None else "",
            pdf_path.strip(),
            dataset_bytes,
        )
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load the selected resume dataset: {error}")
    st.info("Upload a valid CSV with Resume_str and Category columns, or check the local CSV path.")
    st.stop()

st.markdown(
    f"""
    <div class="dataset-strip">
      <span><span class="dataset-dot"></span><strong>Model ready</strong></span>
      <span>·</span>
      <span><strong>{data_stats['csv_resumes']:,}</strong> CSV resumes</span>
      <span>·</span>
      <span><strong>{data_stats['pdf_resumes']:,}</strong> PDF resumes</span>
      <span>·</span>
      <span><strong>{data_stats['categories']}</strong> job families</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if data_stats["pdf_failures"]:
    st.warning(
        f"{data_stats['pdf_failures']} PDF file(s) could not be read and were skipped. "
        "The classifier trained on the remaining readable data."
    )

analyze_tab, dataset_tab = st.tabs(["Analyze a resume", "About the dataset"])
with analyze_tab:
    st.markdown('<h3 class="section-title">Start with your resume</h3>', unsafe_allow_html=True)
    st.caption("Upload a resume to explore likely career paths. Add a job description for a focused match.")
    with st.form("resume_analysis"):
        upload_column, job_column = st.columns([0.9, 1.1], gap="large")
        with upload_column:
            st.markdown("#### 1. Upload a resume")
            uploaded_resume = st.file_uploader(
                "PDF, TXT, or DOCX resume",
                type=["pdf", "txt", "docx"],
                max_upload_size=20,
                help=(
                    "This is for an individual resume to analyze. The labeled CSV in the sidebar "
                    "is the separate training dataset."
                ),
            )
            st.caption("Accepted: PDF, TXT, DOCX · up to 20 MB")
        with job_column:
            st.markdown("#### 2. Add a target role")
            job_description = st.text_area(
                "Job description (optional)",
                height=175,
                placeholder="Paste the role requirements here to see matching skills and useful keywords.",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("Analyze resume", type="primary", use_container_width=True)

    if submitted:
        if uploaded_resume is None:
            st.warning("Upload a resume file to continue.")
        else:
            contents = uploaded_resume.getvalue()
            if len(contents) > MAX_RESUME_BYTES:
                st.error("This resume is larger than the 20 MB upload limit.")
            else:
                try:
                    resume_text = extract_resume_text(uploaded_resume.name, contents)
                    results = analyze_resume(model, resume_text, job_description)
                except (OSError, ValueError) as error:
                    st.error(f"Could not analyze this resume: {error}")
                else:
                    recommendations = results["recommendations"]
                    best_match = recommendations[0]
                    st.markdown("---")
                    st.markdown('<div class="result-heading">Your career snapshot</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="result-subtitle">Resume analyzed: {html.escape(uploaded_resume.name)}</div>',
                        unsafe_allow_html=True,
                    )
                    metric_columns = st.columns([1, 1, 1])
                    with metric_columns[0]:
                        st.metric("Best-fit category", best_match["category"])
                    with metric_columns[1]:
                        st.metric("Model confidence", f"{best_match['confidence']:.0%}")
                    if "alignment" in results:
                        with metric_columns[2]:
                            st.metric("Job alignment", f"{results['alignment']:.0f}%")
                    st.caption(
                        "Category predictions reflect patterns in the supplied dataset; "
                        "they are not a hiring decision or a measure of candidate quality."
                    )

                    st.markdown("#### Best-matching job families")
                    prediction_markup = []
                    for rank, item in enumerate(recommendations, start=1):
                        category = html.escape(str(item["category"]).replace("-", " ").title())
                        confidence = max(0.0, min(1.0, float(item["confidence"])))
                        prediction_markup.append(
                            f"""
                            <div class="prediction-card">
                              <div class="prediction-top">
                                <div><span class="prediction-rank">0{rank}</span>
                                  <span class="prediction-name">{category}</span></div>
                                <span class="prediction-score">{confidence:.0%}</span>
                              </div>
                              <div class="prediction-track">
                                <div class="prediction-fill" style="width:{confidence:.2%}"></div>
                              </div>
                            </div>
                            """
                        )
                    st.markdown("".join(prediction_markup), unsafe_allow_html=True)

                    if "alignment" in results:
                        st.markdown("#### Target-role match")
                        if results["job_skills"]:
                            coverage = results["skill_coverage"]
                            st.write(f"Recognized skill coverage · **{coverage:.0%}**")
                            st.progress(int(coverage * 100))
                            matched, missing = st.columns(2)
                            with matched:
                                st.markdown("**Your matching skills**")
                                if results["matched_skills"]:
                                    st.markdown(
                                        "".join(
                                            f'<span class="skill-label matched">{html.escape(skill)}</span>'
                                            for skill in results["matched_skills"]
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.write("No recognized skills matched yet.")
                            with missing:
                                st.markdown("**Skills to consider highlighting**")
                                if results["missing_skills"]:
                                    st.markdown(
                                        "".join(
                                            f'<span class="skill-label missing">{html.escape(skill)}</span>'
                                            for skill in results["missing_skills"]
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.write("All recognized job skills were found.")
                        else:
                            st.info(
                                "No skills from the built-in vocabulary were recognized in the job description; "
                                "alignment is based on text similarity only."
                            )
                        if results["missing_keywords"]:
                            st.markdown("**Keywords to consider**")
                            st.markdown(
                                "".join(
                                    f'<span class="skill-label">{html.escape(keyword)}</span>'
                                    for keyword in results["missing_keywords"]
                                ),
                                unsafe_allow_html=True,
                            )
                        st.caption(
                            "Job alignment is a heuristic based on dataset vocabulary, recognized skill coverage, "
                            "and text similarity. Review suggestions for context before editing your resume."
                        )

                    if results["resume_skills"]:
                        st.markdown("#### Skills spotted in your resume")
                        st.markdown(
                            "".join(
                                f'<span class="skill-label">{html.escape(skill)}</span>'
                                for skill in results["resume_skills"]
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No skills from the built-in vocabulary were detected in this resume.")

with dataset_tab:
    st.markdown("### What the analyzer learns from")
    st.write(
        "The model trains on labeled `Resume_str` text from the CSV. Enable the optional PDF "
        "corpus in the sidebar to also include readable PDFs in category-named folders. It uses "
        "TF-IDF text features and logistic regression to rank the job categories in the selected data."
    )
    stats_columns = st.columns(3)
    stats_columns[0].metric("CSV resumes", f"{data_stats['csv_resumes']:,}")
    stats_columns[1].metric("Readable PDF resumes", f"{data_stats['pdf_resumes']:,}")
    stats_columns[2].metric("Job categories", f"{data_stats['categories']:,}")
    st.info(
        "All model fitting and resume analysis run locally in this app. This is a dataset-based "
        "text classifier, not a generative AI service, and results depend on the coverage and "
        "quality of the provided training data."
    )
