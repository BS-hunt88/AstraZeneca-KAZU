"""Tests for kazu.utils.grouping — sort_then_group."""

from kazu.utils.grouping import sort_then_group


class TestSortThenGroup:
    def test_basic_sort_and_group(self):
        items = [3, 1, 2, 1, 3, 2]
        result = [(key, list(group)) for key, group in sort_then_group(items, key_func=lambda x: x)]

        assert result == [(1, [1, 1]), (2, [2, 2]), (3, [3, 3])]

    def test_with_reverse(self):
        items = [3, 1, 2, 1, 3, 2]
        result = [
            (key, list(group))
            for key, group in sort_then_group(items, key_func=lambda x: x, reverse=True)
        ]

        assert result == [(3, [3, 3]), (2, [2, 2]), (1, [1, 1])]

    def test_empty_iterable(self):
        result = list(sort_then_group([], key_func=lambda x: x))
        assert result == []

    def test_single_element(self):
        result = [(key, list(group)) for key, group in sort_then_group([42], key_func=lambda x: x)]
        assert result == [(42, [42])]

    def test_string_grouping(self):
        items = ["banana", "apple", "blueberry", "avocado", "cherry"]
        result = [
            (key, list(group)) for key, group in sort_then_group(items, key_func=lambda x: x[0])
        ]

        assert result == [
            ("a", ["apple", "avocado"]),
            ("b", ["banana", "blueberry"]),
            ("c", ["cherry"]),
        ]

    def test_tuple_key_func(self):
        """Group a list of (name, score) tuples by score."""
        items = [("alice", 90), ("bob", 80), ("charlie", 90), ("dave", 80)]
        result = [
            (key, list(group)) for key, group in sort_then_group(items, key_func=lambda x: x[1])
        ]

        assert result == [
            (80, [("bob", 80), ("dave", 80)]),
            (90, [("alice", 90), ("charlie", 90)]),
        ]

    def test_all_same_key(self):
        items = [5, 5, 5]
        result = [(key, list(group)) for key, group in sort_then_group(items, key_func=lambda x: x)]
        assert result == [(5, [5, 5, 5])]

    def test_reverse_with_string_keys(self):
        items = ["banana", "apple", "cherry"]
        result = [
            (key, list(group))
            for key, group in sort_then_group(items, key_func=lambda x: x[0], reverse=True)
        ]
        assert result == [
            ("c", ["cherry"]),
            ("b", ["banana"]),
            ("a", ["apple"]),
        ]
