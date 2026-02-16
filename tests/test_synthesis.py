"""Tests for synthesis stage: XML conversion, synthesizer integration."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from gitlore.config import GitloreConfig
from gitlore.models import (
    AnalysisResult,
    ChurnHotspot,
    ClassifiedComment,
    CommentCategory,
    CommentCluster,
    CommitConvention,
    CouplingPair,
    FixAfterChain,
    FixAfterTier,
    HubFile,
    RevertChain,
    ReviewComment,
    SynthesisResult,
)
from gitlore.synthesis.synthesizer import (
    analysis_to_xml,
    synthesize,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_analysis() -> AnalysisResult:
    """A representative AnalysisResult for testing."""
    base = datetime(2026, 1, 15, tzinfo=timezone.utc)
    return AnalysisResult(
        hotspots=[
            ChurnHotspot(
                path="src/api/routes.ts",
                commit_count=47,
                weighted_commit_count=35.0,
                lines_added=1200,
                lines_deleted=800,
                churn_ratio=4.2,
                fix_ratio=0.35,
                score=8.7,
            ),
            ChurnHotspot(
                path="src/models/user.ts",
                commit_count=22,
                weighted_commit_count=18.0,
                lines_added=500,
                lines_deleted=300,
                churn_ratio=2.1,
                fix_ratio=0.18,
                score=4.3,
            ),
        ],
        revert_chains=[
            RevertChain(
                original_hash="abc1111",
                original_subject="feat(auth): add OAuth flow",
                revert_hashes=["def2222"],
                original_author="Alice",
                original_date=base,
                files=["src/auth/oauth.ts", "src/auth/callback.ts"],
            ),
        ],
        fix_after_chains=[
            FixAfterChain(
                original_hash="ghi3333",
                original_subject="feat(api): add pagination",
                original_author="Bob",
                original_date=base,
                fixup_hashes=["jkl4444", "mno5555"],
                fixup_subjects=["fix: pagination off-by-one", "fix: pagination header"],
                tier=FixAfterTier.IMMEDIATE,
                files=["src/api/pagination.ts", "src/api/routes.ts"],
                time_span=timedelta(hours=2),
            ),
        ],
        coupling_pairs=[
            CouplingPair(
                file_a="src/auth/login.ts",
                file_b="src/auth/session.ts",
                shared_commits=47.0,
                revisions_a=51.0,
                revisions_b=49.0,
                confidence_a_to_b=0.92,
                confidence_b_to_a=0.96,
                lift=3.5,
                strength=0.88,
            ),
            CouplingPair(
                file_a="src/api/routes.ts",
                file_b="src/api/middleware.ts",
                shared_commits=30.0,
                revisions_a=47.0,
                revisions_b=35.0,
                confidence_a_to_b=0.64,
                confidence_b_to_a=0.86,
                lift=2.1,
                strength=0.62,
            ),
        ],
        hub_files=[
            HubFile(
                path="src/api/routes.ts",
                coupled_file_count=12,
                total_coupling_weight=8.5,
            ),
        ],
        conventions=CommitConvention(
            primary_format="conventional_commits",
            format_adherence=0.92,
            types_used={"feat": 120, "fix": 80, "chore": 30, "refactor": 25},
            scopes_used={"auth": 45, "api": 60, "ui": 30},
            imperative_mood_rate=0.88,
            avg_subject_length=42.0,
            subject_under_72_rate=0.95,
            detected_rules=["Use conventional commits", "Include scope when applicable"],
        ),
        comment_clusters=[
            CommentCluster(
                cluster_id=0,
                label="Missing error handling in API routes",
                comments=[
                    ClassifiedComment(
                        comment=ReviewComment(
                            pr_number=101,
                            file_path="src/api/routes.ts",
                            line=42,
                            body="This endpoint doesn't handle the case where the user is not found.",
                            author="reviewer1",
                            created_at=base,
                        ),
                        categories=[CommentCategory.BUG],
                        confidence=0.9,
                    ),
                    ClassifiedComment(
                        comment=ReviewComment(
                            pr_number=105,
                            file_path="src/api/routes.ts",
                            line=88,
                            body="Add error handling for database timeout here.",
                            author="reviewer2",
                            created_at=base,
                        ),
                        categories=[CommentCategory.BUG],
                        confidence=0.85,
                    ),
                    ClassifiedComment(
                        comment=ReviewComment(
                            pr_number=110,
                            file_path="src/api/middleware.ts",
                            line=15,
                            body="Missing try/catch around external service call.",
                            author="reviewer1",
                            created_at=base,
                        ),
                        categories=[CommentCategory.BUG],
                        confidence=0.88,
                    ),
                ],
                coherence=0.82,
            ),
        ],
        total_commits_analyzed=500,
        analysis_date=base,
    )


# ── XML conversion tests ─────────────────────────────────────────────────────


class TestAnalysisToXml:
    def test_produces_valid_xml_structure(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert xml.startswith("<patterns>")
        assert xml.endswith("</patterns>")

    def test_includes_coupling_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="co-change-coupling"' in xml
        assert "src/auth/login.ts" in xml
        assert "src/auth/session.ts" in xml

    def test_includes_hotspot_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="churn-hotspots"' in xml
        assert "src/api/routes.ts" in xml
        assert "8.7" in xml

    def test_includes_revert_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="reverts"' in xml
        assert "OAuth" in xml

    def test_includes_fix_after_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="fix-after"' in xml
        assert "pagination" in xml

    def test_includes_conventions_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="commit-conventions"' in xml
        assert "conventional_commits" in xml

    def test_includes_hub_files_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="hub-files"' in xml

    def test_includes_review_patterns_category(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'category name="review-patterns"' in xml
        assert "error handling" in xml.lower()

    def test_review_patterns_first_in_xml(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        review_pos = xml.index('category name="review-patterns"')
        coupling_pos = xml.index('category name="co-change-coupling"')
        assert review_pos < coupling_pos

    def test_review_xml_includes_sample_metadata(self, sample_analysis: AnalysisResult):
        xml = analysis_to_xml(sample_analysis)
        assert 'author="reviewer1"' in xml
        assert 'pr="101"' in xml
        assert "<body>" in xml
        assert "<categories>" in xml

    def test_review_xml_includes_thread_comments(self):
        base = datetime(2026, 1, 15, tzinfo=timezone.utc)
        result = AnalysisResult(
            comment_clusters=[
                CommentCluster(
                    cluster_id=0,
                    label="Test cluster",
                    comments=[
                        ClassifiedComment(
                            comment=ReviewComment(
                                pr_number=42,
                                file_path="src/foo.py",
                                line=10,
                                body="Main comment body",
                                author="alice",
                                created_at=base,
                                is_resolved=True,
                                diff_context="+ some diff",
                                thread_comments=["Reply one", "Reply two"],
                            ),
                            categories=[CommentCategory.ARCHITECTURE],
                            confidence=0.9,
                        ),
                    ],
                    coherence=0.8,
                ),
            ],
        )
        xml = analysis_to_xml(result)
        assert 'resolved="true"' in xml
        assert "<reply>" in xml
        assert "Reply one" in xml
        assert "<diff_context>" in xml

    def test_notable_comments_xml(self):
        base = datetime(2026, 1, 15, tzinfo=timezone.utc)
        result = AnalysisResult(
            classified_comments=[
                ClassifiedComment(
                    comment=ReviewComment(
                        pr_number=50,
                        file_path="src/important.py",
                        line=5,
                        body="This is a security issue that needs fixing.",
                        author="securityreviewer",
                        created_at=base,
                        is_resolved=False,
                    ),
                    categories=[CommentCategory.SECURITY],
                    confidence=0.95,
                ),
                ClassifiedComment(
                    comment=ReviewComment(
                        pr_number=51,
                        file_path=None,
                        line=None,
                        body="Nice work!",
                        author="friendly",
                        created_at=base,
                    ),
                    categories=[CommentCategory.PRAISE],
                    confidence=0.99,
                ),
            ],
        )
        xml = analysis_to_xml(result)
        assert 'category name="notable-comments"' in xml
        assert "security issue" in xml.lower()
        # Praise-only comments should be filtered out
        assert "Nice work" not in xml

    def test_empty_analysis_produces_minimal_xml(self):
        empty = AnalysisResult()
        xml = analysis_to_xml(empty)
        assert "<patterns>" in xml
        assert "</patterns>" in xml

    def test_xml_escapes_special_characters(self):
        result = AnalysisResult(
            hotspots=[
                ChurnHotspot(
                    path="src/<special>&file.ts",
                    commit_count=10,
                    weighted_commit_count=8.0,
                    lines_added=100,
                    lines_deleted=50,
                    churn_ratio=1.5,
                    fix_ratio=0.2,
                    score=3.0,
                ),
            ],
        )
        xml = analysis_to_xml(result)
        assert "&lt;special&gt;" in xml
        assert "&amp;" in xml

    def test_xml_escapes_revert_files_in_description(self):
        result = AnalysisResult(
            revert_chains=[
                RevertChain(
                    original_hash="abc1234",
                    original_subject="feat: add parser",
                    revert_hashes=["def5678"],
                    files=["src/a&b.py"],
                )
            ]
        )
        xml = analysis_to_xml(result)
        ET.fromstring(xml)  # should be valid XML
        assert "src/a&amp;b.py" in xml


# ── Synthesizer integration test (mocked agent SDK) ──────────────────────────


_SAMPLE_LLM_OUTPUT = """\
<findings>
<finding category="fragile_area" severity="high">
<title>Missing error handling in API routes</title>
<files>
<file>src/api/routes.ts</file>
<file>src/api/middleware.ts</file>
</files>
<evidence>
<point source="reviews">Reviewers consistently flag missing try/catch around database and external service calls</point>
<point source="hotspots">src/api/routes.ts is the top churn hotspot with 35% fix ratio</point>
</evidence>
<insight>The API layer lacks consistent error handling. Database timeouts and user-not-found cases are the most common gaps flagged by reviewers.</insight>
</finding>
<finding category="architecture" severity="medium">
<title>Pipeline merges deterministic and LLM analysis</title>
<files>
<file>src/gitlore/pipeline.py</file>
<file>src/gitlore/synthesis/synthesizer.py</file>
</files>
<evidence>
<point source="code">pipeline.py orchestrates both branches, synthesizer.py consumes XML</point>
</evidence>
<insight>The pipeline merges deterministic git analysis with optional PR-comment analysis. The synthesizer turns analysis XML into structured findings.</insight>
</finding>
<finding category="convention" severity="low">
<title>Keep analyzers deterministic</title>
<evidence>
<point source="conventions">92% conventional commit adherence with scope</point>
<point source="code">Analyzers are pure Python with no LLM calls</point>
</evidence>
<insight>Keep analyzers deterministic and free of direct LLM calls. Use litellm.acompletion with bounded concurrency for async LLM work.</insight>
</finding>
</findings>
"""


def _mock_query_returning(text: str):
    """Create a mock for claude_agent_sdk.query that yields an AssistantMessage.

    Captures the user prompt text from the async iterable for later assertion.
    """
    captured: dict[str, object] = {}

    async def _fake_query(prompt, options=None):
        captured["options"] = options
        # Drain the async iterable prompt and capture the user message text
        if hasattr(prompt, "__aiter__"):
            async for msg in prompt:
                if isinstance(msg, dict) and "message" in msg:
                    captured["prompt"] = msg["message"].get("content", "")

        result_msg = MagicMock(spec=AssistantMessage)
        block = MagicMock(spec=TextBlock)
        block.text = text
        result_msg.content = [block]
        yield result_msg

    _fake_query.captured = captured
    return _fake_query


_PATCH_ENV = "gitlore.synthesis.synthesizer._configure_openrouter_env"


class TestSynthesizer:
    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_returns_findings(self, mock_query, _mock_env, sample_analysis: AnalysisResult):
        fake = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        mock_query.side_effect = fake
        config = GitloreConfig()

        result = synthesize(sample_analysis, config)

        mock_query.assert_called_once()
        assert "<patterns>" in fake.captured["prompt"]
        assert isinstance(result, SynthesisResult)
        assert len(result.findings) == 3
        assert result.findings[0].title == "Missing error handling in API routes"
        assert result.findings[0].category.value == "fragile_area"
        assert result.findings[0].severity.value == "high"
        assert "error handling" in result.findings[0].insight.lower()
        assert "src/api/routes.ts" in result.findings[0].files
        assert len(result.findings[0].evidence) == 2
        assert fake.captured["options"].max_turns == 50
        assert result.analysis is sample_analysis
        assert result.model_used == config.models.synthesizer

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_sets_has_review_data(self, mock_query, _mock_env, sample_analysis: AnalysisResult):
        mock_query.side_effect = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        config = GitloreConfig()

        result = synthesize(sample_analysis, config)

        assert result.has_review_data is True

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_no_review_data_flag(self, mock_query, _mock_env):
        mock_query.side_effect = _mock_query_returning("<findings></findings>")
        config = GitloreConfig()
        empty_analysis = AnalysisResult()

        result = synthesize(empty_analysis, config)

        assert result.has_review_data is False

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_with_empty_analysis(self, mock_query, _mock_env):
        mock_query.side_effect = _mock_query_returning("<findings></findings>")
        config = GitloreConfig()
        empty_analysis = AnalysisResult()

        result = synthesize(empty_analysis, config)

        assert isinstance(result, SynthesisResult)
        assert len(result.findings) == 0

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_prompt_mentions_review_data(self, mock_query, _mock_env, sample_analysis: AnalysisResult):
        fake = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        mock_query.side_effect = fake
        config = GitloreConfig()

        synthesize(sample_analysis, config)

        assert "review" in fake.captured["prompt"].lower()

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_prompt_git_only_note(self, mock_query, _mock_env):
        fake = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        mock_query.side_effect = fake
        config = GitloreConfig()
        analysis = AnalysisResult(total_commits_analyzed=100)

        synthesize(analysis, config)

        assert "git-only" in fake.captured["prompt"].lower()

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_pre_filters_low_confidence_coupling(
        self, mock_query, _mock_env, sample_analysis: AnalysisResult,
    ):
        sample_analysis.coupling_pairs.append(
            CouplingPair(
                file_a="src/utils/helpers.ts",
                file_b="src/utils/format.ts",
                shared_commits=3.0,
                revisions_a=40.0,
                revisions_b=35.0,
                confidence_a_to_b=0.08,
                confidence_b_to_a=0.09,
                lift=0.5,
                strength=0.1,
            ),
        )
        fake = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        mock_query.side_effect = fake
        config = GitloreConfig()

        synthesize(sample_analysis, config)

        assert "src/utils/helpers.ts" not in fake.captured["prompt"]

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_synthesize_stores_raw_xml(self, mock_query, _mock_env, sample_analysis: AnalysisResult):
        mock_query.side_effect = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        config = GitloreConfig()

        result = synthesize(sample_analysis, config)

        assert "<findings>" in result.raw_xml

    @patch(_PATCH_ENV)
    @patch("gitlore.synthesis.synthesizer.query")
    def test_content_property_renders_markdown(self, mock_query, _mock_env, sample_analysis: AnalysisResult):
        mock_query.side_effect = _mock_query_returning(_SAMPLE_LLM_OUTPUT)
        config = GitloreConfig()

        result = synthesize(sample_analysis, config)

        assert "## Missing error handling in API routes" in result.content
        assert "src/api/routes.ts" in result.content
