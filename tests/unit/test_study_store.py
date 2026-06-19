"""Unit tests for core/study_store.py — StudyStore."""

import time
import pytest
from core.study_store import StudyStore


class TestStudyStore:
    @pytest.fixture
    def store(self, tmp_path):
        return StudyStore(db_path=str(tmp_path / "study.db"))

    # ── create_session / list_user ────────────────────────────────────────

    def test_create_session_and_list(self, store):
        store.create_session("alice", "study_alice_1", "rbi_report.pdf")
        sessions = store.list_user("alice")
        assert len(sessions) == 1
        assert sessions[0]["study_thread_id"] == "study_alice_1"
        assert sessions[0]["doc_name"] == "rbi_report.pdf"
        assert sessions[0]["status"] == "active"
        assert sessions[0]["score"] == 0.0
        assert sessions[0]["total"] == 0

    def test_list_user_isolation(self, store):
        store.create_session("alice", "study_alice_1", "doc_a.pdf")
        store.create_session("bob",   "study_bob_1",   "doc_b.pdf")
        assert len(store.list_user("alice")) == 1
        assert len(store.list_user("bob"))   == 1
        assert store.list_user("alice")[0]["study_thread_id"] == "study_alice_1"

    def test_list_user_empty_for_unknown(self, store):
        store.create_session("alice", "study_alice_1", "doc.pdf")
        assert store.list_user("bob") == []

    def test_create_session_idempotent(self, store):
        store.create_session("alice", "s1", "doc.pdf")
        store.create_session("alice", "s1", "doc.pdf")   # second call should be no-op
        assert len(store.list_user("alice")) == 1

    def test_list_user_respects_limit(self, store):
        for i in range(8):
            store.create_session("alice", f"s{i}", f"doc{i}.pdf")
        assert len(store.list_user("alice", limit=5)) == 5

    def test_list_user_multiple_sessions(self, store):
        store.create_session("alice", "s1", "doc_a.pdf")
        store.create_session("alice", "s2", "doc_b.pdf")
        sessions = store.list_user("alice")
        assert len(sessions) == 2
        doc_names = {s["doc_name"] for s in sessions}
        assert doc_names == {"doc_a.pdf", "doc_b.pdf"}

    # ── update_score ──────────────────────────────────────────────────────

    def test_update_score(self, store):
        store.create_session("alice", "s1", "doc.pdf")
        store.update_score("s1", 4.5, 6)
        session = store.list_user("alice")[0]
        assert session["score"] == 4.5
        assert session["total"] == 6

    def test_update_score_overwrite(self, store):
        store.create_session("alice", "s1", "doc.pdf")
        store.update_score("s1", 2.0, 3)
        store.update_score("s1", 5.0, 6)
        session = store.list_user("alice")[0]
        assert session["score"] == 5.0
        assert session["total"] == 6

    # ── complete_session ──────────────────────────────────────────────────

    def test_complete_session_changes_status(self, store):
        store.create_session("alice", "s1", "doc.pdf")
        store.complete_session("s1")
        session = store.list_user("alice")[0]
        assert session["status"] == "completed"

    def test_complete_session_does_not_affect_other_sessions(self, store):
        store.create_session("alice", "s1", "doc_a.pdf")
        store.create_session("alice", "s2", "doc_b.pdf")
        store.complete_session("s1")
        sessions = {s["study_thread_id"]: s for s in store.list_user("alice")}
        assert sessions["s1"]["status"] == "completed"
        assert sessions["s2"]["status"] == "active"

    # ── list_user ordering ────────────────────────────────────────────────

    def test_list_user_ordered_newest_first(self, store):
        store.create_session("alice", "s1", "doc_a.pdf")
        time.sleep(1.05)   # SQLite second-precision — need >1s gap
        store.create_session("alice", "s2", "doc_b.pdf")
        sessions = store.list_user("alice")
        assert sessions[0]["study_thread_id"] == "s2"
