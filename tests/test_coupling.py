"""Tests for co-change coupling analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gitlore.analyzers.coupling import analyze_coupling
from gitlore.config import AnalysisConfig
from gitlore.models import Commit, FileChange


def _make_commit(
    hash: str,
    files: list[FileChange],
    date: datetime | None = None,
) -> Commit:
    return Commit(
        hash=hash,
        author_name="Test",
        author_email="test@test.com",
        author_date=date or datetime(2026, 1, 1, tzinfo=timezone.utc),
        parents=["p1"],
        subject="some commit",
        body="",
        files=files,
    )


class TestAnalyzeCoupling:
    def test_basic_coupling(self):
        """Two files that always change together should be coupled."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = [
            _make_commit(
                f"c{i}",
                [
                    FileChange("src/api.py", added=10, deleted=5),
                    FileChange("src/models.py", added=5, deleted=3),
                ],
                date=base + timedelta(days=i),
            )
            for i in range(5)
        ]
        # Add some commits touching only api.py
        commits.append(
            _make_commit(
                "c5",
                [FileChange("src/api.py", added=3, deleted=1)],
                date=base + timedelta(days=5),
            )
        )

        config = AnalysisConfig(min_shared_commits=2, min_coupling_confidence=0.1, min_coupling_lift=1.0)
        pairs, modules, hubs = analyze_coupling(commits, config, reference_date=ref_date)

        assert len(pairs) > 0
        pair = pairs[0]
        assert {pair.file_a, pair.file_b} == {"src/api.py", "src/models.py"}
        assert pair.shared_commits > 0
        assert pair.confidence_a_to_b > 0
        assert pair.confidence_b_to_a > 0
        assert pair.lift > 0
        assert pair.strength > 0

    def test_no_coupling_different_files(self):
        """Files that never change together should not be coupled."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = [
            _make_commit(
                "c1",
                [FileChange("src/a.py", added=10, deleted=0)],
                date=base,
            ),
            _make_commit(
                "c2",
                [FileChange("src/b.py", added=10, deleted=0)],
                date=base + timedelta(days=1),
            ),
        ]
        config = AnalysisConfig(min_shared_commits=1)
        pairs, modules, hubs = analyze_coupling(commits, config, reference_date=ref_date)
        assert len(pairs) == 0

    def test_mega_commit_skipped(self):
        """Commits with too many files should be skipped."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # One mega commit with 60 files
        files = [FileChange(f"src/file{i}.py", added=1, deleted=0) for i in range(60)]
        commits = [_make_commit("c1", files, date=base)]

        config = AnalysisConfig(max_files_per_commit=50, min_shared_commits=1)
        pairs, modules, hubs = analyze_coupling(commits, config, reference_date=ref_date)
        assert len(pairs) == 0

    def test_confidence_asymmetry(self):
        """Confidence should be directional."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = []
        # a.py and b.py change together 5 times
        for i in range(5):
            commits.append(
                _make_commit(
                    f"ab{i}",
                    [
                        FileChange("a.py", added=5, deleted=2),
                        FileChange("b.py", added=3, deleted=1),
                    ],
                    date=base + timedelta(days=i),
                )
            )
        # b.py also changes alone 5 more times
        for i in range(5, 10):
            commits.append(
                _make_commit(
                    f"b{i}",
                    [FileChange("b.py", added=5, deleted=2)],
                    date=base + timedelta(days=i),
                )
            )

        config = AnalysisConfig(min_shared_commits=2, min_coupling_confidence=0.1, min_coupling_lift=1.0)
        pairs, _, _ = analyze_coupling(commits, config, reference_date=ref_date)

        assert len(pairs) > 0
        pair = pairs[0]
        # a->b confidence should be higher than b->a because a only changes with b
        if pair.file_a == "a.py":
            assert pair.confidence_a_to_b > pair.confidence_b_to_a
        else:
            assert pair.confidence_b_to_a > pair.confidence_a_to_b

    def test_rename_map_applied(self):
        """File renames should be resolved before counting."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = [
            _make_commit(
                "c1",
                [
                    FileChange("old_name.py", added=10, deleted=0),
                    FileChange("partner.py", added=5, deleted=0),
                ],
                date=base,
            ),
            _make_commit(
                "c2",
                [
                    FileChange("new_name.py", added=10, deleted=0),
                    FileChange("partner.py", added=5, deleted=0),
                ],
                date=base + timedelta(days=1),
            ),
            _make_commit(
                "c3",
                [
                    FileChange("new_name.py", added=10, deleted=0),
                    FileChange("partner.py", added=5, deleted=0),
                ],
                date=base + timedelta(days=2),
            ),
        ]
        rename_map = {"old_name.py": "new_name.py"}
        config = AnalysisConfig(min_shared_commits=2, min_coupling_confidence=0.1, min_coupling_lift=1.0)
        pairs, _, _ = analyze_coupling(
            commits, config, reference_date=ref_date, rename_map=rename_map
        )

        # All three should count as new_name.py + partner.py
        assert len(pairs) > 0
        pair = pairs[0]
        assert {pair.file_a, pair.file_b} == {"new_name.py", "partner.py"}

    def test_empty_input(self):
        pairs, modules, hubs = analyze_coupling([])
        assert pairs == []
        assert modules == []
        assert hubs == []

    def test_hub_file_detection(self):
        """A file coupled with many others should be detected as a hub."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = []
        # hub.py changes with a.py, b.py, c.py, d.py multiple times
        for i in range(5):
            for partner in ["a.py", "b.py", "c.py", "d.py"]:
                commits.append(
                    _make_commit(
                        f"c_{partner}_{i}",
                        [
                            FileChange("hub.py", added=5, deleted=2),
                            FileChange(partner, added=3, deleted=1),
                        ],
                        date=base + timedelta(days=i, hours=hash(partner) % 24),
                    )
                )

        config = AnalysisConfig(min_shared_commits=2, min_coupling_confidence=0.1, min_coupling_lift=1.0)
        pairs, modules, hubs = analyze_coupling(commits, config, reference_date=ref_date)

        assert len(hubs) > 0
        hub = next((h for h in hubs if h.path == "hub.py"), None)
        assert hub is not None
        assert hub.coupled_file_count >= 3

    def test_module_detection(self):
        """Groups of files that change together should form modules."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = []
        # Module 1: auth files change together
        for i in range(5):
            commits.append(
                _make_commit(
                    f"auth_{i}",
                    [
                        FileChange("src/auth/login.py", added=5, deleted=2),
                        FileChange("src/auth/session.py", added=3, deleted=1),
                        FileChange("src/auth/token.py", added=2, deleted=1),
                    ],
                    date=base + timedelta(days=i),
                )
            )
        # Module 2: payment files change together
        for i in range(5):
            commits.append(
                _make_commit(
                    f"pay_{i}",
                    [
                        FileChange("src/pay/checkout.py", added=5, deleted=2),
                        FileChange("src/pay/invoice.py", added=3, deleted=1),
                        FileChange("src/pay/stripe.py", added=2, deleted=1),
                    ],
                    date=base + timedelta(days=i + 5),
                )
            )

        config = AnalysisConfig(min_shared_commits=2, min_coupling_confidence=0.1, min_coupling_lift=1.0)
        pairs, modules, hubs = analyze_coupling(commits, config, reference_date=ref_date)

        # Should detect at least 2 modules
        assert len(modules) >= 2
        # Each module should have coherent files
        for mod in modules:
            assert len(mod.files) >= 2
