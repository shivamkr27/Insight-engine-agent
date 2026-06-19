"""Unit tests for core/history.py — ConversationStore."""

import time
import pytest
from core.history import ConversationStore


class TestConversationStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ConversationStore(db_path=str(tmp_path / "history.db"))

    # ── upsert / list ────────────────────────────────────────────────────────

    def test_upsert_creates_conversation(self, store):
        store.upsert("t1", "alice")
        convs = store.list_user("alice")
        assert len(convs) == 1
        assert convs[0]["thread_id"] == "t1"
        assert convs[0]["title"] == "New conversation"

    def test_list_user_returns_only_own_conversations(self, store):
        store.upsert("t1", "alice")
        store.upsert("t2", "bob")
        alice_convs = store.list_user("alice")
        assert len(alice_convs) == 1
        assert alice_convs[0]["thread_id"] == "t1"

    def test_list_user_empty_for_new_user(self, store):
        store.upsert("t1", "alice")
        assert store.list_user("bob") == []

    def test_list_user_ordered_by_last_active_desc(self, store):
        # SQLite strftime has 1-second precision — sleep >1s to get distinct timestamps
        store.upsert("t1", "alice")
        time.sleep(1.05)
        store.upsert("t2", "alice")
        convs = store.list_user("alice")
        assert convs[0]["thread_id"] == "t2"

    def test_upsert_idempotent_does_not_duplicate(self, store):
        store.upsert("t1", "alice")
        store.upsert("t1", "alice")
        assert len(store.list_user("alice")) == 1

    def test_list_user_respects_limit(self, store):
        for i in range(10):
            store.upsert(f"t{i}", "alice")
        assert len(store.list_user("alice", limit=5)) == 5

    # ── set_title ────────────────────────────────────────────────────────────

    def test_set_title_updates_default_title(self, store):
        store.upsert("t1", "alice")
        store.set_title("t1", "What is PM-KISAN?")
        convs = store.list_user("alice")
        assert convs[0]["title"] == "What is PM-KISAN?"

    def test_set_title_truncates_to_80_chars(self, store):
        store.upsert("t1", "alice")
        long_title = "A" * 120
        store.set_title("t1", long_title)
        assert len(store.list_user("alice")[0]["title"]) == 80

    def test_set_title_does_not_overwrite_existing_custom_title(self, store):
        store.upsert("t1", "alice")
        store.set_title("t1", "First title")
        store.set_title("t1", "Second title")   # should be ignored
        assert store.list_user("alice")[0]["title"] == "First title"

    # ── touch ────────────────────────────────────────────────────────────────

    def test_touch_increments_message_count(self, store):
        store.upsert("t1", "alice")
        store.touch("t1")
        store.touch("t1")
        assert store.list_user("alice")[0]["message_count"] == 2

    def test_touch_updates_last_active(self, store):
        store.upsert("t1", "alice")
        store.upsert("t2", "alice")
        time.sleep(1.05)  # need >1s for SQLite second-precision timestamps
        store.touch("t1")
        # t1 should now be first (most recently active)
        convs = store.list_user("alice")
        assert convs[0]["thread_id"] == "t1"

    # ── delete ───────────────────────────────────────────────────────────────

    def test_delete_removes_own_conversation(self, store):
        store.upsert("t1", "alice")
        result = store.delete("t1", "alice")
        assert result is True
        assert store.list_user("alice") == []

    def test_delete_returns_false_for_wrong_user(self, store):
        store.upsert("t1", "alice")
        result = store.delete("t1", "bob")
        assert result is False
        assert len(store.list_user("alice")) == 1

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("nonexistent", "alice") is False
