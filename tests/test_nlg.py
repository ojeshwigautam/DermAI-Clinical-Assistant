"""
test_nlg.py — Unit tests for the NLG report generation module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.nlp.nlg_module import PatientReportGenerator, compute_bleu
from src.utils.config import SEVERITY_LABELS, CLASS_LABELS


class TestPatientReportGenerator:
    @pytest.fixture
    def generator(self):
        return PatientReportGenerator()

    @pytest.fixture
    def sample_entities(self):
        return {
            "body_part":        ["left forearm"],
            "lesion_type":      ["mole"],
            "symptom_duration": ["3 months"],
        }

    def test_generate_returns_string(self, generator, sample_entities):
        report = generator.generate(
            cnn_class="mel",
            cnn_confidence=0.85,
            entities=sample_entities,
            severity="mild",
            report_id="TEST001",
        )
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_header(self, generator, sample_entities):
        report = generator.generate("mel", 0.85, sample_entities, "mild", "TEST001")
        assert "DERMATOLOGY" in report.upper()

    def test_report_contains_report_id(self, generator, sample_entities):
        report = generator.generate("mel", 0.85, sample_entities, "mild", "TEST001")
        assert "TEST001" in report

    def test_report_contains_entity_info(self, generator, sample_entities):
        report = generator.generate("mel", 0.85, sample_entities, "mild", "TEST001")
        assert "left forearm" in report.lower() or "forearm" in report.lower()

    @pytest.mark.parametrize("severity", SEVERITY_LABELS)
    def test_all_severity_levels(self, generator, sample_entities, severity):
        report = generator.generate("nv", 0.75, sample_entities, severity, "TEST002")
        assert isinstance(report, str)
        assert len(report) > 50

    def test_unknown_severity_defaults(self, generator, sample_entities):
        # Should not raise; defaults to 'mild'
        report = generator.generate("nv", 0.75, sample_entities, "unknown_sev", "TEST003")
        assert isinstance(report, str)

    def test_empty_entities(self, generator):
        report = generator.generate("bcc", 0.6, {}, "moderate", "TEST004")
        assert isinstance(report, str)
        assert "unspecified" in report.lower() or "skin lesion" in report.lower()

    def test_confidence_in_report(self, generator, sample_entities):
        report = generator.generate("mel", 0.812, sample_entities, "severe", "TEST005")
        # Confidence should appear as a percentage
        assert "81" in report or "0.812" in report

    @pytest.mark.parametrize("cls", CLASS_LABELS)
    def test_all_cnn_classes(self, generator, sample_entities, cls):
        report = generator.generate(cls, 0.8, sample_entities, "mild", "TEST006")
        assert cls.upper() in report


class TestBLEUEvaluation:
    def test_bleu_same_text(self):
        texts = ["the patient has a melanoma on the forearm"] * 5
        score = compute_bleu(texts, texts)
        assert score > 0.9

    def test_bleu_different_text(self):
        generated  = ["hello world this is a test"] * 5
        references = ["completely different clinical text here"] * 5
        score = compute_bleu(generated, references)
        assert 0.0 <= score <= 1.0

    def test_bleu_returns_float(self):
        gen = ["patient reports mild itching on the arm for 2 weeks"]
        ref = ["mild pruritus noted on the left arm present for 2 weeks"]
        score = compute_bleu(gen, ref)
        assert isinstance(score, float)

    def test_bleu_range(self):
        gen = ["This is a generated patient report about skin lesion severity."]
        ref = ["This is a reference patient report about dermatological severity."]
        score = compute_bleu(gen, ref)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
