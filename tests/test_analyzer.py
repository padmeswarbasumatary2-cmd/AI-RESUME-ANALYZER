import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyzer import (
    analyze_resume,
    extract_resume_text,
    extract_skills,
    load_training_data,
    train_classifier,
)


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.training_data = pd.DataFrame(
            [
                {
                    "text": "Python SQL data analysis machine learning dashboards",
                    "category": "INFORMATION-TECHNOLOGY",
                },
                {
                    "text": "Python SQL software engineering data analysis",
                    "category": "INFORMATION-TECHNOLOGY",
                },
                {
                    "text": "Python data analytics and software development SQL",
                    "category": "INFORMATION-TECHNOLOGY",
                },
                {
                    "text": "patient care clinical nursing healthcare records",
                    "category": "HEALTHCARE",
                },
                {
                    "text": "healthcare nursing patient care medical records",
                    "category": "HEALTHCARE",
                },
                {
                    "text": "clinical healthcare patient support and nursing",
                    "category": "HEALTHCARE",
                },
            ]
        )
        self.model = train_classifier(self.training_data)

    def test_skill_matching_uses_whole_terms(self):
        skills = extract_skills("Python and SQL skills; experience with Pythonic tooling.")

        self.assertIn("Python", skills)
        self.assertIn("SQL", skills)

    def test_analyzer_ranks_related_category_and_job_skills(self):
        result = analyze_resume(
            self.model,
            "Python SQL data analysis software engineering",
            "We need Python, SQL, and project management experience.",
        )

        self.assertEqual(result["recommendations"][0]["category"], "INFORMATION-TECHNOLOGY")
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("SQL", result["matched_skills"])
        self.assertIn("Project Management", result["missing_skills"])
        self.assertGreaterEqual(result["alignment"], 0)
        self.assertLessEqual(result["alignment"], 100)

    def test_txt_resume_extraction_and_empty_document_error(self):
        text = extract_resume_text("resume.txt", b"  Python   developer ")

        self.assertEqual(text, "Python developer")
        with self.assertRaisesRegex(ValueError, "No readable text"):
            extract_resume_text("empty.txt", b" \n ")

    def test_invalid_docx_is_reported_as_a_readable_error(self):
        with self.assertRaisesRegex(ValueError, "DOCX file could not be read"):
            extract_resume_text("resume.docx", b"not a document")

    def test_csv_loader_uses_labeled_resume_text(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "resumes.csv"
            pd.DataFrame(
                {
                    "Resume_str": ["Python developer", "Nurse"],
                    "Category": ["INFORMATION-TECHNOLOGY", "HEALTHCARE"],
                }
            ).to_csv(csv_path, index=False)

            data, stats = load_training_data(csv_path)

        self.assertEqual(len(data), 2)
        self.assertEqual(stats["csv_resumes"], 2)
        self.assertEqual(stats["pdf_resumes"], 0)
        self.assertEqual(stats["categories"], 2)

    def test_csv_loader_accepts_in_memory_upload(self):
        csv_bytes = (
            b"Resume_str,Category\n"
            b'"Python data analysis",INFORMATION-TECHNOLOGY\n'
            b'"Clinical patient care",HEALTHCARE\n'
        )

        data, stats = load_training_data(csv_bytes=csv_bytes)

        self.assertEqual(len(data), 2)
        self.assertEqual(stats["csv_resumes"], 2)
        self.assertEqual(stats["categories"], 2)


if __name__ == "__main__":
    unittest.main()
