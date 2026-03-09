"""All shared data models for gitlore."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# ── Git extraction ──────────────────────────────────────────────────────────


@dataclass
class FileChange:
    """A single file changed in a commit."""

    path: str
    added: int | None = None  # None for binary
    deleted: int | None = None  # None for binary
    old_path: str | None = None  # set if rename


@dataclass
class Commit:
    """Raw commit from git log."""

    hash: str
    author_name: str
    author_email: str
    author_date: datetime
    parents: list[str]
    subject: str
    body: str
    files: list[FileChange] = field(default_factory=list)

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1

    @property
    def file_paths(self) -> list[str]:
        return [f.path for f in self.files]


# ── Commit classification ───────────────────────────────────────────────────


class CommitType(Enum):
    FEAT = "feat"
    FIX = "fix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CI = "ci"
    BUILD = "build"
    CHORE = "chore"
    STYLE = "style"
    PERF = "perf"
    REVERT = "revert"
    MERGE = "merge"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedCommit:
    """A commit with its classified type."""

    commit: Commit
    commit_type: CommitType
    scope: str | None = None
    is_breaking: bool = False
    ticket: str | None = None
    is_conventional: bool = False


# ── Churn / hotspot ─────────────────────────────────────────────────────────


@dataclass
class ChurnHotspot:
    """A file identified as a churn hotspot."""

    path: str
    commit_count: int
    weighted_commit_count: float
    lines_added: int
    lines_deleted: int
    churn_ratio: float  # (added + deleted) / file changes
    fix_ratio: float  # fraction of commits that are fixes
    score: float  # final weighted hotspot score


# ── Revert chains ───────────────────────────────────────────────────────────


@dataclass
class RevertChain:
    """A chain of revert commits."""

    original_hash: str
    original_subject: str
    revert_hashes: list[str] = field(default_factory=list)
    original_author: str = ""
    original_date: datetime | None = None
    files: list[str] = field(default_factory=list)

    @property
    def is_effectively_reverted(self) -> bool:
        return len(self.revert_hashes) % 2 == 1

    @property
    def depth(self) -> int:
        return len(self.revert_hashes)


# ── Fix-after chains ────────────────────────────────────────────────────────


class FixAfterTier(Enum):
    IMMEDIATE = "immediate"  # < 30 min, same author
    FOLLOWUP = "followup"  # < 4 hours, same author, fix keywords
    DELAYED = "delayed"  # < 7 days, any author, strong fix signals


@dataclass
class FixAfterChain:
    """A commit followed by one or more fix-up commits."""

    original_hash: str
    original_subject: str
    original_author: str
    original_date: datetime
    fixup_hashes: list[str] = field(default_factory=list)
    fixup_subjects: list[str] = field(default_factory=list)
    tier: FixAfterTier = FixAfterTier.IMMEDIATE
    files: list[str] = field(default_factory=list)
    time_span: timedelta = field(default_factory=timedelta)


# ── Co-change coupling ──────────────────────────────────────────────────────


@dataclass
class CouplingPair:
    """Two files that change together frequently."""

    file_a: str
    file_b: str
    shared_commits: float  # weighted count
    revisions_a: float
    revisions_b: float
    confidence_a_to_b: float
    confidence_b_to_a: float
    lift: float
    strength: float


@dataclass
class ImplicitModule:
    """A group of files detected as a logical module via graph community detection."""

    module_id: int
    files: list[str]
    internal_coupling_avg: float = 0.0


@dataclass
class HubFile:
    """A file that couples with many others (high centrality)."""

    path: str
    coupled_file_count: int
    total_coupling_weight: float


# ── Commit conventions ──────────────────────────────────────────────────────


@dataclass
class CommitConvention:
    """Detected commit convention patterns."""

    primary_format: str  # "conventional_commits", "ticket_prefix", "freeform", etc.
    format_adherence: float  # 0.0-1.0
    types_used: dict[str, int] = field(default_factory=dict)
    scopes_used: dict[str, int] = field(default_factory=dict)
    ticket_format: str | None = None
    ticket_adherence: float = 0.0
    imperative_mood_rate: float = 0.0
    avg_subject_length: float = 0.0
    subject_under_72_rate: float = 0.0
    has_body_rate: float = 0.0
    ends_with_period_rate: float = 0.0
    starts_lowercase_rate: float = 0.0
    detected_rules: list[str] = field(default_factory=list)


# ── PR review comments ──────────────────────────────────────────────────────


@dataclass
class ReviewComment:
    """A review comment from a PR."""

    pr_number: int
    file_path: str | None
    line: int | None
    body: str
    author: str
    created_at: datetime
    is_resolved: bool | None = None
    diff_context: str | None = None
    thread_comments: list[str] = field(default_factory=list)
    review_state: str | None = None  # APPROVED, CHANGES_REQUESTED, etc.


class CommentCategory(Enum):
    BUG = "bug"
    ARCHITECTURE = "architecture"
    CONVENTION = "convention"
    SECURITY = "security"
    PERFORMANCE = "performance"
    NITPICK = "nitpick"
    QUESTION = "question"
    PRAISE = "praise"


@dataclass
class ClassifiedComment:
    """A review comment with LLM-assigned categories."""

    comment: ReviewComment
    categories: list[CommentCategory]
    confidence: float = 0.0


@dataclass
class CommentCluster:
    """A group of semantically similar comments."""

    cluster_id: int
    label: str
    comments: list[ClassifiedComment]
    centroid: list[float] | None = None
    coherence: float = 0.0


# ── Pipeline aggregates ─────────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """Aggregates all analysis outputs for synthesis input."""

    hotspots: list[ChurnHotspot] = field(default_factory=list)
    revert_chains: list[RevertChain] = field(default_factory=list)
    fix_after_chains: list[FixAfterChain] = field(default_factory=list)
    coupling_pairs: list[CouplingPair] = field(default_factory=list)
    implicit_modules: list[ImplicitModule] = field(default_factory=list)
    hub_files: list[HubFile] = field(default_factory=list)
    conventions: CommitConvention | None = None
    comment_clusters: list[CommentCluster] = field(default_factory=list)
    classified_comments: list[ClassifiedComment] = field(default_factory=list)
    total_commits_analyzed: int = 0
    analysis_date: datetime | None = None


# ── Knowledge notes ─────────────────────────────────────────────────────────


_VALID_CONFIDENCE = {"high", "medium", "low"}

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


@dataclass
class KnowledgeNote:
    """A piece of tribal knowledge extracted from repository investigation."""

    id: str
    text: str
    anchors: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    confidence: str = "medium"
    created_at: datetime | None = None
    search_text: str = ""
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if self.confidence not in _VALID_CONFIDENCE:
            self.confidence = "medium"


# ── Index / retrieval ───────────────────────────────────────────────────────


@dataclass
class SourceCoverage:
    """Describes which data sources were available during the last build."""

    git: bool = True
    github: bool = False
    docs: bool = False
    semantic: bool = False
    classified_reviews: bool = False


@dataclass
class BuildMetadata:
    """Metadata about the latest built context index."""

    repo_path: str
    built_at: datetime
    total_commits_analyzed: int = 0
    note_count: int = 0
    source_coverage: SourceCoverage = field(default_factory=SourceCoverage)
    head_commit: str = ""


@dataclass
class FileEdge:
    """A structural relationship used to widen planning scope."""

    src: str
    dst: str
    edge_type: str
    score: float
    reason: str


@dataclass
class RelatedFile:
    """A file suggested because it is structurally related to the query scope."""

    path: str
    reason: str
    score: float


# ── Query / retrieval ───────────────────────────────────────────────────────


@dataclass
class PlanningQuery:
    """The retrieval inputs for a planning-time lookup."""

    task: str
    files: list[str] = field(default_factory=list)
    diff_text: str = ""
    diff_path: str | None = None
    tentative_plan: str = ""
    question: str = ""
    max_notes: int = 5


@dataclass
class PlanningNote:
    """A minimal planning-time note returned to agents and humans."""

    text: str
    refs: list[str] = field(default_factory=list)
    confidence: str = "medium"


@dataclass
class PlanningBrief:
    """A bounded planning-time retrieval result."""

    task: str
    summary: str = ""
    notes: list[PlanningNote] = field(default_factory=list)
    related_files: list[RelatedFile] = field(default_factory=list)
    source_coverage: SourceCoverage = field(default_factory=SourceCoverage)
    build_metadata: BuildMetadata | None = None


@dataclass
class ExportBundle:
    """Knowledge notes rendered into export formats."""

    notes: list[KnowledgeNote] = field(default_factory=list)
    build_metadata: BuildMetadata | None = None
