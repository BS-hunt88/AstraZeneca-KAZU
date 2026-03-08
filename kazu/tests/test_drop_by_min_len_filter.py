"""Comprehensive unit tests for DropByMinLenFilter.

Tests that entities with match strings shorter than a given min_len
(specifically 3 characters) are correctly identified for filtering,
and that the filter integrates properly with EntityFilterCleanupAction
to remove short entities from documents.
"""

from kazu.data import Document, Entity
from kazu.steps.other.cleanup import (
    DropByMinLenFilter,
    EntityFilterCleanupAction,
)


def _make_entity(match: str, entity_class: str = "gene", namespace: str = "test") -> Entity:
    """Helper to create a contiguous entity with the given match string."""
    return Entity.load_contiguous_entity(
        start=0,
        end=len(match),
        match=match,
        entity_class=entity_class,
        namespace=namespace,
    )


class TestDropByMinLenFilterMinLen3:
    """Tests for DropByMinLenFilter with min_len=3."""

    def setup_method(self) -> None:
        self.filter_func = DropByMinLenFilter(min_len=3)

    # --- Direct filter call tests ---

    def test_single_char_entity_is_filtered(self) -> None:
        """A 1-character entity should be flagged for removal."""
        ent = _make_entity("A")
        assert self.filter_func(ent) is True

    def test_two_char_entity_is_filtered(self) -> None:
        """A 2-character entity should be flagged for removal."""
        ent = _make_entity("AB")
        assert self.filter_func(ent) is True

    def test_three_char_entity_is_kept(self) -> None:
        """A 3-character entity (at the boundary) should NOT be flagged."""
        ent = _make_entity("ABC")
        assert self.filter_func(ent) is False

    def test_long_entity_is_kept(self) -> None:
        """A long entity string should NOT be flagged."""
        ent = _make_entity("EGFR")
        assert self.filter_func(ent) is False

    def test_empty_string_entity_is_filtered(self) -> None:
        """An empty match string should be flagged for removal."""
        ent = _make_entity("")
        assert self.filter_func(ent) is True

    def test_whitespace_only_two_chars_is_filtered(self) -> None:
        """A match that is only whitespace but < min_len should be filtered."""
        ent = _make_entity("  ")
        assert self.filter_func(ent) is True

    def test_whitespace_three_chars_is_kept(self) -> None:
        """A 3-char whitespace string is at the boundary and should be kept by this filter."""
        ent = _make_entity("   ")
        assert self.filter_func(ent) is False

    def test_unicode_two_char_entity_is_filtered(self) -> None:
        """Unicode characters still count as individual chars; 2 should be filtered."""
        ent = _make_entity("\u03b1\u03b2")  # alpha beta
        assert self.filter_func(ent) is True

    def test_unicode_three_char_entity_is_kept(self) -> None:
        """Three unicode characters should not be filtered."""
        ent = _make_entity("\u03b1\u03b2\u03b3")  # alpha beta gamma
        assert self.filter_func(ent) is False

    def test_different_entity_classes_same_behaviour(self) -> None:
        """The filter is entity-class agnostic; short entities of any class are filtered."""
        for entity_class in ("gene", "disease", "drug", "cell_line"):
            short = _make_entity("XY", entity_class=entity_class)
            long = _make_entity("XYZ", entity_class=entity_class)
            assert self.filter_func(short) is True, f"Expected filter for class={entity_class}"
            assert self.filter_func(long) is False, f"Unexpected filter for class={entity_class}"

    # --- Integration with EntityFilterCleanupAction ---

    def test_integration_short_entities_removed_from_document(self) -> None:
        """When wired through EntityFilterCleanupAction, short entities are removed
        from a Document's sections."""
        action = EntityFilterCleanupAction(filter_fns=[self.filter_func])

        doc = Document.create_simple_document("AB and EGFR are mentioned in this text.")
        doc.sections[0].entities.extend(
            [
                _make_entity("AB"),      # len 2 -> should be removed
                _make_entity("EGFR"),    # len 4 -> should stay
                _make_entity("p53"),     # len 3 -> should stay
            ]
        )

        assert len(doc.sections[0].entities) == 3
        action.cleanup(doc)
        remaining = doc.sections[0].entities
        assert len(remaining) == 2
        remaining_matches = {e.match for e in remaining}
        assert remaining_matches == {"EGFR", "p53"}

    def test_integration_all_short_entities_removed(self) -> None:
        """If all entities are shorter than 3 chars, the section should have no entities."""
        action = EntityFilterCleanupAction(filter_fns=[self.filter_func])

        doc = Document.create_simple_document("A B C short entities only.")
        doc.sections[0].entities.extend(
            [
                _make_entity("A"),
                _make_entity("B"),
                _make_entity("XY"),
            ]
        )

        action.cleanup(doc)
        assert len(doc.sections[0].entities) == 0

    def test_integration_no_entities_removed_when_all_long(self) -> None:
        """If all entities are >= 3 chars, none should be removed."""
        action = EntityFilterCleanupAction(filter_fns=[self.filter_func])

        doc = Document.create_simple_document("BRCA1 and TP53 are genes.")
        doc.sections[0].entities.extend(
            [
                _make_entity("BRCA1"),
                _make_entity("TP53"),
                _make_entity("ABC"),
            ]
        )

        action.cleanup(doc)
        assert len(doc.sections[0].entities) == 3

    def test_integration_empty_entity_list_unchanged(self) -> None:
        """A document with no entities should remain unchanged after cleanup."""
        action = EntityFilterCleanupAction(filter_fns=[self.filter_func])

        doc = Document.create_simple_document("No entities here.")
        action.cleanup(doc)
        assert len(doc.sections[0].entities) == 0


class TestDropByMinLenFilterParameterized:
    """Tests for DropByMinLenFilter with various min_len values to ensure
    the parameterization works correctly."""

    def test_min_len_1_only_filters_empty(self) -> None:
        """With min_len=1, only an empty string should be filtered."""
        f = DropByMinLenFilter(min_len=1)
        assert f(_make_entity("")) is True
        assert f(_make_entity("X")) is False
        assert f(_make_entity("AB")) is False

    def test_min_len_5_filters_short_gene_names(self) -> None:
        """With min_len=5, common short gene symbols like p53 get filtered."""
        f = DropByMinLenFilter(min_len=5)
        assert f(_make_entity("p53")) is True    # len 3
        assert f(_make_entity("EGFR")) is True   # len 4
        assert f(_make_entity("BRCA1")) is False  # len 5
        assert f(_make_entity("BRCA12")) is False  # len 6

    def test_min_len_0_filters_nothing(self) -> None:
        """With min_len=0, no entity should ever be filtered."""
        f = DropByMinLenFilter(min_len=0)
        assert f(_make_entity("")) is False
        assert f(_make_entity("A")) is False
        assert f(_make_entity("long entity name")) is False
