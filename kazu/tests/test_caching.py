"""Tests for kazu.utils.caching — EntityLinkingLookupCache."""

import pytest

from kazu.data import (
    CharSpan,
    Entity,
    EquivalentIdAggregationStrategy,
    EquivalentIdSet,
    LinkingCandidate,
    LinkingMetrics,
    MentionConfidence,
    CandidatesToMetrics,
)
from kazu.utils.caching import EntityLinkingLookupCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(match: str, entity_class: str, start: int = 0) -> Entity:
    """Create a minimal Entity for cache tests."""
    end = start + len(match)
    return Entity(
        match=match,
        entity_class=entity_class,
        spans=frozenset([CharSpan(start=start, end=end)]),
        namespace="test_step",
        mention_confidence=MentionConfidence.HIGHLY_LIKELY,
    )


def _make_candidate_and_metrics(
    synonym_norm: str,
    ids: list[str],
    parser_name: str = "test_parser",
    search_score: float = 0.9,
) -> tuple[LinkingCandidate, LinkingMetrics]:
    """Create a LinkingCandidate + LinkingMetrics pair."""
    candidate = LinkingCandidate(
        raw_synonyms=frozenset([synonym_norm]),
        synonym_norm=synonym_norm,
        parser_name=parser_name,
        is_symbolic=False,
        associated_id_sets=frozenset(
            [EquivalentIdSet(ids_and_source=frozenset((id_, "test") for id_ in ids))]
        ),
        aggregated_by=EquivalentIdAggregationStrategy.NO_STRATEGY,
    )
    metrics = LinkingMetrics(search_score=search_score)
    return candidate, metrics


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntityLinkingLookupCache:
    def test_update_stores_candidates(self):
        cache = EntityLinkingLookupCache(lookup_cache_size=100)
        ent = _make_entity("aspirin", "drug")
        cand, met = _make_candidate_and_metrics("aspirin", ["D001"])
        candidates: CandidatesToMetrics = {cand: met}

        cache.update_candidates_lookup_cache(ent, candidates)

        # now check_lookup_cache should find it
        ent2 = _make_entity("aspirin", "drug")
        misses = cache.check_lookup_cache([ent2])
        assert len(misses) == 0
        # the entity should have been updated with the cached candidates
        assert len(ent2.linking_candidates) == 1

    def test_update_does_not_overwrite_existing(self):
        cache = EntityLinkingLookupCache(lookup_cache_size=100)
        ent = _make_entity("aspirin", "drug")
        cand1, met1 = _make_candidate_and_metrics("aspirin", ["D001"], search_score=0.9)
        candidates1: CandidatesToMetrics = {cand1: met1}
        cache.update_candidates_lookup_cache(ent, candidates1)

        # try to overwrite with different candidates
        cand2, met2 = _make_candidate_and_metrics("aspirin_alt", ["D999"], search_score=0.1)
        candidates2: CandidatesToMetrics = {cand2: met2}
        cache.update_candidates_lookup_cache(ent, candidates2)

        # the cache should still have the original candidates
        ent_check = _make_entity("aspirin", "drug")
        cache.check_lookup_cache([ent_check])
        # only the first candidate should be present
        assert cand1 in ent_check.linking_candidates
        assert cand2 not in ent_check.linking_candidates

    def test_check_lookup_cache_returns_misses(self):
        cache = EntityLinkingLookupCache(lookup_cache_size=100)
        ent_known = _make_entity("aspirin", "drug")
        cand, met = _make_candidate_and_metrics("aspirin", ["D001"])
        cache.update_candidates_lookup_cache(ent_known, {cand: met})

        # check with a known and an unknown entity
        ent_hit = _make_entity("aspirin", "drug")
        ent_miss = _make_entity("unknown_drug", "drug")
        misses = cache.check_lookup_cache([ent_hit, ent_miss])

        assert len(misses) == 1
        assert misses[0].match == "unknown_drug"
        # ent_hit should have candidates
        assert len(ent_hit.linking_candidates) == 1
        # ent_miss should have no candidates
        assert len(ent_miss.linking_candidates) == 0

    def test_lfu_eviction_when_cache_full(self):
        """LFU cache should evict least-frequently-used entries when full."""
        cache = EntityLinkingLookupCache(lookup_cache_size=2)

        # add first entity
        ent_a = _make_entity("drug_a", "drug")
        cand_a, met_a = _make_candidate_and_metrics("drug_a", ["A1"])
        cache.update_candidates_lookup_cache(ent_a, {cand_a: met_a})

        # add second entity
        ent_b = _make_entity("drug_b", "drug")
        cand_b, met_b = _make_candidate_and_metrics("drug_b", ["B1"])
        cache.update_candidates_lookup_cache(ent_b, {cand_b: met_b})

        # access ent_a to increase its frequency (via check_lookup_cache)
        ent_a_check = _make_entity("drug_a", "drug")
        cache.check_lookup_cache([ent_a_check])

        # add a third entity — this should evict ent_b (least frequently used)
        ent_c = _make_entity("drug_c", "drug")
        cand_c, met_c = _make_candidate_and_metrics("drug_c", ["C1"])
        cache.update_candidates_lookup_cache(ent_c, {cand_c: met_c})

        # ent_a should still be in cache
        ent_a_verify = _make_entity("drug_a", "drug")
        misses_a = cache.check_lookup_cache([ent_a_verify])
        assert len(misses_a) == 0

        # ent_b should have been evicted
        ent_b_verify = _make_entity("drug_b", "drug")
        misses_b = cache.check_lookup_cache([ent_b_verify])
        assert len(misses_b) == 1

    def test_different_entity_class_is_separate_cache_entry(self):
        """Entities with the same match text but different entity_class
        should be cached separately."""
        cache = EntityLinkingLookupCache(lookup_cache_size=100)

        ent_drug = _make_entity("aspirin", "drug")
        cand_drug, met_drug = _make_candidate_and_metrics("aspirin", ["D001"])
        cache.update_candidates_lookup_cache(ent_drug, {cand_drug: met_drug})

        # same match, different class — should be a miss
        ent_gene = _make_entity("aspirin", "gene")
        misses = cache.check_lookup_cache([ent_gene])
        assert len(misses) == 1

    def test_empty_entity_list(self):
        cache = EntityLinkingLookupCache(lookup_cache_size=100)
        misses = cache.check_lookup_cache([])
        assert misses == []
