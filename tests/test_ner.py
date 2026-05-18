"""
test_ner.py — Unit tests for the spaCy NER pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.nlp.ner_pipeline import build_ner_pipeline, extract_entities, build_patterns


class TestNERPatterns:
    def test_patterns_not_empty(self):
        patterns = build_patterns()
        assert len(patterns) > 0

    def test_patterns_have_required_keys(self):
        patterns = build_patterns()
        for p in patterns:
            assert "label" in p
            assert "pattern" in p

    def test_labels_valid(self):
        valid_labels = {"BODY_PART", "LESION_TYPE", "SYMPTOM_DURATION"}
        patterns = build_patterns()
        for p in patterns:
            assert p["label"] in valid_labels


class TestNERPipeline:
    @pytest.fixture(scope="class")
    def nlp(self):
        return build_ner_pipeline()

    def test_pipeline_loads(self, nlp):
        assert nlp is not None

    def test_extract_body_part(self, nlp):
        text = "I have a lesion on my left forearm."
        entities = extract_entities(text, nlp)
        assert "body_part" in entities
        # Should find "left forearm" or "forearm"
        combined = " ".join(entities.get("body_part", []))
        assert "forearm" in combined.lower()

    def test_extract_lesion_type(self, nlp):
        text = "The patient has a melanoma on the upper back."
        entities = extract_entities(text, nlp)
        assert "lesion_type" in entities
        combined = " ".join(entities.get("lesion_type", []))
        assert "melanoma" in combined.lower()

    def test_extract_duration(self, nlp):
        text = "This mole has been present for 3 months."
        entities = extract_entities(text, nlp)
        assert "symptom_duration" in entities
        combined = " ".join(entities.get("symptom_duration", []))
        assert "month" in combined.lower()

    def test_extract_all_entities(self, nlp):
        text = (
            "I've had a dark brown mole on my upper back for 6 months. "
            "The lesion has grown recently."
        )
        entities = extract_entities(text, nlp)
        # At least one of the three entity types should be found
        found = any([
            len(entities.get("body_part", [])) > 0,
            len(entities.get("lesion_type", [])) > 0,
            len(entities.get("symptom_duration", [])) > 0,
        ])
        assert found

    def test_empty_text(self, nlp):
        entities = extract_entities("", nlp)
        assert isinstance(entities, dict)
        # All lists should be empty
        for key in ["body_part", "lesion_type", "symptom_duration"]:
            assert entities.get(key, []) == []

    def test_returns_dict(self, nlp):
        entities = extract_entities("Some text about skin.", nlp)
        assert isinstance(entities, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
