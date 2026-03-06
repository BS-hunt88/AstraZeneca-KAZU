"""Tests for kazu.database.in_memory_db — SynonymDatabase and MetadataDatabase."""

import pytest

from kazu.data import (
    EquivalentIdAggregationStrategy,
    EquivalentIdSet,
    LinkingCandidate,
)
from kazu.database.in_memory_db import MetadataDatabase, SynonymDatabase
from kazu.utils.utils import Singleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linking_candidate(
    synonym_norm: str,
    raw_synonyms: frozenset[str],
    ids: list[str],
    source: str = "test_source",
    parser_name: str = "test_parser",
    agg_strategy: EquivalentIdAggregationStrategy = EquivalentIdAggregationStrategy.NO_STRATEGY,
) -> LinkingCandidate:
    """Build a minimal LinkingCandidate for database tests."""
    return LinkingCandidate(
        raw_synonyms=raw_synonyms,
        synonym_norm=synonym_norm,
        parser_name=parser_name,
        is_symbolic=False,
        associated_id_sets=frozenset(
            [EquivalentIdSet(ids_and_source=frozenset((id_, source) for id_ in ids))]
        ),
        aggregated_by=agg_strategy,
    )


# ---------------------------------------------------------------------------
# Fixtures — clear singletons before and after every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singletons():
    Singleton.clear_all()
    yield
    Singleton.clear_all()


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingletonBehaviour:
    def test_multiple_instantiations_return_same_object(self):
        db1 = MetadataDatabase()
        db2 = MetadataDatabase()
        assert db1 is db2

        syn1 = SynonymDatabase()
        syn2 = SynonymDatabase()
        assert syn1 is syn2

    def test_clear_all_resets_state(self):
        mdb = MetadataDatabase()
        mdb.add_parser("p", "drug", {"id1": {"label": "aspirin"}})
        assert "p" in mdb.loaded_parsers

        Singleton.clear_all()

        mdb_new = MetadataDatabase()
        assert mdb_new is not mdb
        assert "p" not in mdb_new.loaded_parsers


# ---------------------------------------------------------------------------
# MetadataDatabase
# ---------------------------------------------------------------------------


class TestMetadataDatabase:
    def test_add_parser_stores_metadata(self):
        mdb = MetadataDatabase()
        metadata = {"id1": {"label": "aspirin"}, "id2": {"label": "ibuprofen"}}
        mdb.add_parser("drug_parser", "drug", metadata)

        assert "drug_parser" in mdb.loaded_parsers
        assert mdb.parser_name_to_ent_class["drug_parser"] == "drug"
        assert mdb.get_by_idx("drug_parser", "id1") == {"label": "aspirin"}
        assert mdb.get_by_idx("drug_parser", "id2") == {"label": "ibuprofen"}

    def test_add_parser_overwrites_existing(self):
        mdb = MetadataDatabase()
        mdb.add_parser("p", "gene", {"g1": {"symbol": "EGFR"}})
        assert mdb.get_by_idx("p", "g1") == {"symbol": "EGFR"}

        # overwrite with new metadata
        mdb.add_parser("p", "gene", {"g1": {"symbol": "BRCA1"}, "g2": {"symbol": "TP53"}})
        assert mdb.get_by_idx("p", "g1") == {"symbol": "BRCA1"}
        assert mdb.get_by_idx("p", "g2") == {"symbol": "TP53"}

    def test_get_by_idx_returns_deep_copy(self):
        mdb = MetadataDatabase()
        mdb.add_parser("p", "drug", {"id1": {"label": "aspirin", "tags": ["nsaid"]}})

        result = mdb.get_by_idx("p", "id1")
        # mutate the returned copy
        result["label"] = "MUTATED"
        result["tags"].append("extra")

        # the database should be unaffected
        original = mdb.get_by_idx("p", "id1")
        assert original["label"] == "aspirin"
        assert original["tags"] == ["nsaid"]

    def test_get_by_idx_raises_on_missing_key(self):
        mdb = MetadataDatabase()
        mdb.add_parser("p", "drug", {"id1": {"label": "aspirin"}})
        with pytest.raises(KeyError):
            mdb.get_by_idx("p", "nonexistent_id")

    def test_get_all_returns_full_dict(self):
        mdb = MetadataDatabase()
        metadata = {"id1": {"label": "a"}, "id2": {"label": "b"}}
        mdb.add_parser("p", "drug", metadata)
        assert mdb.get_all("p") == metadata


# ---------------------------------------------------------------------------
# SynonymDatabase
# ---------------------------------------------------------------------------


class TestSynonymDatabase:
    def test_add_parser_indexes_by_normalized_string(self):
        sdb = SynonymDatabase()
        lc = _make_linking_candidate("aspirin", frozenset(["Aspirin", "aspirin"]), ["D001"])
        sdb.add_parser("drug_parser", [lc])

        assert "drug_parser" in sdb.loaded_parsers
        retrieved = sdb.get("drug_parser", "aspirin")
        assert retrieved is lc

    def test_add_parser_indexes_by_aggregation_strategy(self):
        sdb = SynonymDatabase()
        lc = _make_linking_candidate(
            "aspirin",
            frozenset(["Aspirin"]),
            ["D001"],
            agg_strategy=EquivalentIdAggregationStrategy.UNAMBIGUOUS,
        )
        sdb.add_parser("drug_parser", [lc])

        syns = sdb.get_syns_for_id(
            "drug_parser",
            "D001",
            strategy_filters={EquivalentIdAggregationStrategy.UNAMBIGUOUS},
        )
        assert "aspirin" in syns

    def test_add_parser_indexes_by_id(self):
        sdb = SynonymDatabase()
        lc1 = _make_linking_candidate("aspirin", frozenset(["Aspirin"]), ["D001"])
        lc2 = _make_linking_candidate("ibuprofen", frozenset(["Ibuprofen"]), ["D002"])
        sdb.add_parser("drug_parser", [lc1, lc2])

        # both IDs should be retrievable
        syns_d001 = sdb.get_syns_for_id("drug_parser", "D001")
        assert "aspirin" in syns_d001

        syns_d002 = sdb.get_syns_for_id("drug_parser", "D002")
        assert "ibuprofen" in syns_d002

    def test_get_returns_correct_linking_candidate(self):
        sdb = SynonymDatabase()
        lc1 = _make_linking_candidate("aspirin", frozenset(["Aspirin"]), ["D001"])
        lc2 = _make_linking_candidate("ibuprofen", frozenset(["Ibuprofen"]), ["D002"])
        sdb.add_parser("drug_parser", [lc1, lc2])

        assert sdb.get("drug_parser", "aspirin") is lc1
        assert sdb.get("drug_parser", "ibuprofen") is lc2

    def test_get_raises_on_missing_synonym(self):
        sdb = SynonymDatabase()
        lc = _make_linking_candidate("aspirin", frozenset(["Aspirin"]), ["D001"])
        sdb.add_parser("drug_parser", [lc])

        with pytest.raises(KeyError):
            sdb.get("drug_parser", "nonexistent_synonym")

    def test_get_all_returns_all_candidates(self):
        sdb = SynonymDatabase()
        lc1 = _make_linking_candidate("aspirin", frozenset(["Aspirin"]), ["D001"])
        lc2 = _make_linking_candidate("ibuprofen", frozenset(["Ibuprofen"]), ["D002"])
        sdb.add_parser("drug_parser", [lc1, lc2])

        all_candidates = sdb.get_all("drug_parser")
        assert len(all_candidates) == 2
        assert "aspirin" in all_candidates
        assert "ibuprofen" in all_candidates
        assert all_candidates["aspirin"] is lc1
        assert all_candidates["ibuprofen"] is lc2

    def test_multiple_ids_same_synonym(self):
        """A single synonym can map to multiple IDs via one EquivalentIdSet."""
        sdb = SynonymDatabase()
        lc = _make_linking_candidate("p27", frozenset(["p27"]), ["GENE1", "GENE2"])
        sdb.add_parser("gene_parser", [lc])

        retrieved = sdb.get("gene_parser", "p27")
        all_ids: set[str] = set()
        for equiv_id_set in retrieved.associated_id_sets:
            all_ids.update(equiv_id_set.ids)
        assert "GENE1" in all_ids
        assert "GENE2" in all_ids

    def test_get_syns_for_id_no_strategy_filter(self):
        sdb = SynonymDatabase()
        lc = _make_linking_candidate("aspirin", frozenset(["Aspirin"]), ["D001"])
        sdb.add_parser("drug_parser", [lc])

        # no filter — should still return the synonym
        syns = sdb.get_syns_for_id("drug_parser", "D001")
        assert "aspirin" in syns

    def test_get_syns_for_id_with_wrong_strategy_filter(self):
        sdb = SynonymDatabase()
        lc = _make_linking_candidate(
            "aspirin",
            frozenset(["Aspirin"]),
            ["D001"],
            agg_strategy=EquivalentIdAggregationStrategy.NO_STRATEGY,
        )
        sdb.add_parser("drug_parser", [lc])

        syns = sdb.get_syns_for_id(
            "drug_parser",
            "D001",
            strategy_filters={EquivalentIdAggregationStrategy.UNAMBIGUOUS},
        )
        assert "aspirin" not in syns
