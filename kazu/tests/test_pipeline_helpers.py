"""Tests for kazu.pipeline — helper functions and handlers."""

import logging

import pytest

from kazu.data import (
    CharSpan,
    Document,
    Entity,
    MentionConfidence,
    Section,
    PROCESSING_EXCEPTION,
)
from kazu.pipeline import (
    Pipeline,
    PipelineValueError,
    calc_doc_size,
    batch_metrics,
    FailedDocsLogHandler,
)


# ---------------------------------------------------------------------------
# calc_doc_size
# ---------------------------------------------------------------------------


class TestCalcDocSize:
    def test_single_section(self):
        doc = Document.create_simple_document("hello world")
        assert calc_doc_size(doc) == len("hello world")

    def test_multiple_sections(self):
        doc = Document(
            sections=[
                Section(text="abc", name="s1"),
                Section(text="defgh", name="s2"),
                Section(text="ij", name="s3"),
            ]
        )
        assert calc_doc_size(doc) == 3 + 5 + 2

    def test_empty_document(self):
        doc = Document(sections=[])
        assert calc_doc_size(doc) == 0

    def test_single_empty_section(self):
        doc = Document(sections=[Section(text="", name="empty")])
        assert calc_doc_size(doc) == 0


# ---------------------------------------------------------------------------
# batch_metrics
# ---------------------------------------------------------------------------


def _make_entity(match: str, start: int = 0) -> Entity:
    end = start + len(match)
    return Entity(
        match=match,
        entity_class="test",
        spans=frozenset([CharSpan(start=start, end=end)]),
        namespace="test_step",
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    )


class TestBatchMetrics:
    def test_basic_metrics(self):
        doc1 = Document(sections=[Section(text="abcde", name="s1")])
        doc2 = Document(sections=[Section(text="fg", name="s1")])
        metrics = batch_metrics([doc1, doc2])

        assert metrics["max_length"] == 5
        assert metrics["mean_length"] == 3.5
        assert metrics["max_ents"] == 0
        assert metrics["mean_ents"] == 0.0

    def test_with_entities(self):
        section = Section(text="EGFR is a gene", name="s1")
        section.entities.append(_make_entity("EGFR", start=0))
        doc = Document(sections=[section])

        metrics = batch_metrics([doc])
        assert metrics["max_ents"] == 1
        assert metrics["mean_ents"] == 1.0

    def test_multiple_docs_with_entities(self):
        s1 = Section(text="EGFR is a gene", name="s1")
        s1.entities.append(_make_entity("EGFR", start=0))

        s2 = Section(text="BRCA1 and TP53", name="s1")
        s2.entities.append(_make_entity("BRCA1", start=0))
        s2.entities.append(_make_entity("TP53", start=10))

        doc1 = Document(sections=[s1])
        doc2 = Document(sections=[s2])

        metrics = batch_metrics([doc1, doc2])
        assert metrics["max_ents"] == 2
        assert metrics["mean_ents"] == 1.5


# ---------------------------------------------------------------------------
# Pipeline.prefilter_docs
# ---------------------------------------------------------------------------


class _DummyStep:
    """A minimal Step implementation for testing Pipeline."""

    def __init__(self, name: str = "DummyStep"):
        self._name = name

    @classmethod
    def namespace(cls) -> str:
        return "DummyStep"

    def __call__(self, docs: list[Document]) -> tuple[list[Document], list[Document]]:
        return docs, []


class TestPrefilterDocs:
    def test_skips_long_documents(self):
        pipeline = Pipeline(steps=[_DummyStep()], skip_doc_len=10)
        short_doc = Document.create_simple_document("short")
        long_doc = Document.create_simple_document("a" * 20)

        result = pipeline.prefilter_docs([short_doc, long_doc])
        assert len(result) == 1
        assert result[0] is short_doc
        # long_doc should have PROCESSING_EXCEPTION set
        assert PROCESSING_EXCEPTION in long_doc.metadata

    def test_no_skip_when_none(self):
        pipeline = Pipeline(steps=[_DummyStep()], skip_doc_len=None)
        long_doc = Document.create_simple_document("a" * 999999)
        result = pipeline.prefilter_docs([long_doc])
        assert len(result) == 1

    def test_boundary_exactly_at_limit(self):
        pipeline = Pipeline(steps=[_DummyStep()], skip_doc_len=5)
        doc_at_limit = Document.create_simple_document(
            "abcde"
        )  # len == 5, which is >= skip_doc_len
        result = pipeline.prefilter_docs([doc_at_limit])
        assert len(result) == 0
        assert PROCESSING_EXCEPTION in doc_at_limit.metadata

    def test_multiple_docs_mixed(self):
        pipeline = Pipeline(steps=[_DummyStep()], skip_doc_len=10)
        docs = [
            Document.create_simple_document("ok"),
            Document.create_simple_document("a" * 15),
            Document.create_simple_document("fine"),
            Document.create_simple_document("b" * 10),  # exactly at limit
        ]
        result = pipeline.prefilter_docs(docs)
        assert len(result) == 2
        assert result[0].sections[0].text == "ok"
        assert result[1].sections[0].text == "fine"


# ---------------------------------------------------------------------------
# FailedDocsLogHandler
# ---------------------------------------------------------------------------


class TestFailedDocsLogHandler:
    def test_logs_warning_with_error_message(self, caplog):
        handler = FailedDocsLogHandler()
        doc = Document.create_simple_document("test")
        doc.metadata[PROCESSING_EXCEPTION] = "something went wrong"

        with caplog.at_level(logging.WARNING, logger="kazu.pipeline"):
            handler({"test_step": [doc]})

        assert any("something went wrong" in record.message for record in caplog.records)

    def test_logs_warning_without_error_message(self, caplog):
        handler = FailedDocsLogHandler()
        doc = Document.create_simple_document("test")
        # no PROCESSING_EXCEPTION in metadata

        with caplog.at_level(logging.WARNING, logger="kazu.pipeline"):
            handler({"test_step": [doc]})

        assert any("No error mesasge" in record.message for record in caplog.records)

    def test_multiple_steps_and_docs(self, caplog):
        handler = FailedDocsLogHandler()
        doc1 = Document.create_simple_document("test1")
        doc1.metadata[PROCESSING_EXCEPTION] = "error1"
        doc2 = Document.create_simple_document("test2")
        doc2.metadata[PROCESSING_EXCEPTION] = "error2"

        with caplog.at_level(logging.WARNING, logger="kazu.pipeline"):
            handler({"step_a": [doc1], "step_b": [doc2]})

        messages = [r.message for r in caplog.records]
        assert any("error1" in m for m in messages)
        assert any("error2" in m for m in messages)


# ---------------------------------------------------------------------------
# Pipeline.__call__ — step_namespaces + step_group conflict
# ---------------------------------------------------------------------------


class TestPipelineValueError:
    def test_raises_when_both_step_namespaces_and_step_group(self):
        pipeline = Pipeline(
            steps=[_DummyStep()],
            step_groups={"group1": ["DummyStep"]},
        )
        docs = [Document.create_simple_document("test")]

        with pytest.raises(PipelineValueError):
            pipeline(docs, step_namespaces=["DummyStep"], step_group="group1")

    def test_raises_for_nonexistent_step_namespace(self):
        pipeline = Pipeline(steps=[_DummyStep()])
        docs = [Document.create_simple_document("test")]

        with pytest.raises(PipelineValueError, match="do not exist"):
            pipeline(docs, step_namespaces=["NonExistentStep"])

    def test_raises_for_nonexistent_step_group(self):
        pipeline = Pipeline(
            steps=[_DummyStep()],
            step_groups={"group1": ["DummyStep"]},
        )
        docs = [Document.create_simple_document("test")]

        with pytest.raises(PipelineValueError):
            pipeline(docs, step_group="nonexistent_group")

    def test_raises_when_no_step_groups_configured(self):
        pipeline = Pipeline(steps=[_DummyStep()])
        docs = [Document.create_simple_document("test")]

        with pytest.raises(PipelineValueError, match="does not have any step groups"):
            pipeline(docs, step_group="any_group")
