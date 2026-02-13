# Git Log Extraction: Library Comparison & Strategy

Research for gitlore's extraction layer -- pulling structured data from git history for analysis.

## 1. Library Comparison

### Raw `git log` via subprocess

**How it works:** Shell out to the `git` CLI using `subprocess.Popen`/`run` with `--pretty=format:` and `--numstat`/`--stat` flags. Parse the structured text output.

**Pros:**
- Fastest option by far -- git's C implementation is heavily optimized
- Full access to every `git log` feature (format placeholders, `--diff-filter`, `-M` rename detection, `--follow`, revision ranges)
- Zero dependency beyond git itself (always available in target environments)
- Streaming output via `Popen` -- constant memory regardless of repo size
- `--pretty=format:` with custom delimiters makes parsing deterministic
- Incremental via `FROM_HASH..TO_HASH` revision range syntax

**Cons:**
- Must write and maintain a parser (moderate effort, one-time)
- Platform-specific edge cases (encoding, path separators on Windows)
- Spawning processes has overhead per invocation (mitigated by batching into single `git log` calls)

**Performance:** Processes 50k+ commits in seconds when using `--pretty=format:` with `--numstat`. A single `git log` invocation with streaming stdout is the fastest possible path.

### PyDriller (v2.9, actively maintained)

**How it works:** High-level Python framework built on top of GitPython. Provides `Repository.traverse_commits()` iterator with rich commit/file/metric objects.

**Pros:**
- Excellent API ergonomics -- 50% less code than GitPython for equivalent tasks
- Rich data model: `Commit` objects with `modified_files`, `diff_parsed`, change types (ADD/MODIFY/DELETE/RENAME/COPY), method-level analysis for supported languages
- Built-in filtering: `since`/`to` dates, `from_commit`/`to_commit` hashes, `only_in_branch`, `only_no_merge`, `only_modifications_with_file_types`, `only_authors`
- Process metrics: DMM unit complexity/size/interfacing, code churn
- Lazy loading for expensive operations (diffs, source code)
- `num_workers` parameter for multithreaded traversal
- Active maintenance: v2.7 (Oct 2024), v2.8 (Jul 2025), v2.9 (Sep 2025)

**Cons:**
- Built on GitPython, which spawns subprocess per command -- ~55 commits/sec throughput
- Processing 50k commits: ~15 minutes (vs seconds for raw `git log`)
- Memory issues with multiprocessing due to pickle serialization of Commit objects
- GitPython leaks resources (file handles, processes), requires manual cache clearing
- `num_workers > 1` breaks commit ordering
- No true streaming -- objects stay in memory if you accumulate them
- Heavy for our use case: we don't need method-level analysis or DMM metrics

**Key API:**
```python
from pydriller import Repository

for commit in Repository(
    path_to_repo='/path/to/repo',
    from_commit='abc123',       # incremental start
    to_commit='def456',         # incremental end
    only_no_merge=True,         # skip merge commits
    order='reverse',            # newest first
).traverse_commits():
    print(commit.hash, commit.msg, commit.author.name)
    for f in commit.modified_files:
        print(f.filename, f.change_type.name, f.added_lines, f.deleted_lines)
        print(f.old_path, '->', f.new_path)  # rename detection
```

### GitPython (v3.1.x, stable maintenance)

**How it works:** Python library wrapping git CLI commands via subprocess. Provides ORM-like objects for repos, commits, trees, blobs, diffs.

**Pros:**
- Mature, widely used (basis for PyDriller, DVC, etc.)
- Good object model for repos/commits/trees
- Two database backends: GitDB (pure Python, lower memory) and GitCmdObjectDB (faster, uses git-cat-file)
- Can access low-level git plumbing

**Cons:**
- Still subprocess-based -- each operation spawns a git process
- 2-5x slower than git CLI for bulk extraction from packed repos (GitDB backend)
- Resource leaks on Windows, manual cleanup required
- More verbose than PyDriller for common operations
- Limited diff parsing compared to PyDriller

**Verdict:** If you're going to use a subprocess wrapper, use PyDriller (cleaner API) or go directly to raw subprocess (faster).

### dulwich (v0.22+)

**How it works:** Pure Python git implementation. Reads git object store directly without shelling out.

**Pros:**
- No git installation required (pure Python)
- Direct access to git internals (pack files, objects, refs)
- Optional Rust extensions for performance
- No subprocess overhead

**Cons:**
- Significantly slower than native git for most operations
- Low-level API -- you build everything yourself (commit traversal, diff generation)
- Rust extensions improve speed but add build complexity
- Smaller ecosystem, less documentation for common patterns
- Re-implements git algorithms in Python -- risk of subtle behavioral differences

**Verdict:** Wrong tool for this job. We can assume git is installed in target environments.

### pygit2 (libgit2 bindings)

**How it works:** Python bindings to libgit2, a C library reimplementing git.

**Pros:**
- Fast C-level operations without subprocess overhead
- Fine-grained control over git internals
- Good performance for most operations

**Cons:**
- Pins to specific libgit2 versions -- dependency management pain
- libgit2 installation can be difficult (system library)
- Some operations (large blob reads) are paradoxically slower than GitPython
- API is lower-level, more code to write
- Build/install friction is a dealbreaker for a CLI tool users install via pip/pipx

**Verdict:** Dependency burden too high for a CLI tool. Not worth it when raw `git log` is faster and simpler.

## 2. Performance on Large Repos (50k+ commits)

| Approach | 50k commits | Memory | Incremental |
|----------|------------|--------|-------------|
| `git log` subprocess (streaming) | 5-15 seconds | ~constant (streaming stdout) | Native (`hash..HEAD`) |
| PyDriller | ~15 minutes | Grows with accumulation; leaks possible | `from_commit`/`to_commit` |
| GitPython | ~10 minutes | 2-5x more with GitDB | Manual revision range |
| dulwich | ~20+ minutes | High (Python object overhead) | Manual walk |

**Recommendation for 50k+ commits:** Raw `git log` with streaming parsing is the only approach that stays under 30 seconds. All Python-native approaches are orders of magnitude slower.

### Streaming Pattern (subprocess)
```python
import subprocess

proc = subprocess.Popen(
    ['git', '-C', repo_path, 'log',
     '--pretty=format:%H%x00%an%x00%ae%x00%aI%x00%P%x00%s%x00%b%x1e',
     '--numstat'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    encoding='utf-8',
    errors='replace',
)

# Stream line-by-line -- constant memory
for line in proc.stdout:
    # parse each line
    pass
```

The key insight: `%x00` (NUL) as field separator and `%x1e` (record separator) as commit delimiter makes parsing unambiguous. `--numstat` appends tab-separated `added\tdeleted\tfilename` lines after each commit.

## 3. What Data to Extract

### Commit-level fields
| Field | git format | Notes |
|-------|-----------|-------|
| Hash | `%H` | Full 40-char SHA |
| Author name | `%an` | |
| Author email | `%ae` | For identity deduplication |
| Author date | `%aI` | ISO 8601 format |
| Committer name | `%cn` | Usually same as author |
| Committer date | `%cI` | Differs from author date on rebase/cherry-pick |
| Parent hashes | `%P` | Space-separated; >1 parent = merge commit |
| Subject | `%s` | First line of message |
| Body | `%b` | Rest of message |

### File-level fields (via `--numstat`)
| Field | Source | Notes |
|-------|--------|-------|
| Lines added | numstat col 1 | `-` for binary files |
| Lines deleted | numstat col 2 | `-` for binary files |
| File path | numstat col 3 | |
| Rename detection | `--numstat -M` | Shows `{old => new}` path format |
| Change type | `--diff-filter` or separate `--name-status` | A/M/D/R/C |

### Merge commit handling
- Detect via `len(parents) > 1`
- For gitlore, we generally want to **skip merge commits** (`--no-merges`) since they don't represent authored work -- the real changes are in the merged branch commits
- Exception: squash-merged PRs appear as single commits, which we do want
- Merge commits from `git merge --no-ff` contain no diff (or a combined diff of conflicts), rarely useful

### Branch info
- `--all` flag includes all branches, not just HEAD
- `--decorate=short` adds branch/tag labels to commits
- For gitlore, analyzing the default branch (main/master) is the primary use case
- Branch filtering: `git log main` or `git log --first-parent` for linear history

## 4. Incremental Analysis

### Strategy: Checkpoint by commit hash

Store the last-processed commit hash. On next run, use revision range:

```python
def get_new_commits(repo_path: str, since_hash: str | None = None) -> subprocess.Popen:
    cmd = ['git', '-C', repo_path, 'log', '--pretty=format:...', '--numstat']
    if since_hash:
        cmd.append(f'{since_hash}..HEAD')  # only new commits
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, encoding='utf-8')
```

### Checkpoint storage
- Store in a simple JSON file alongside the output: `{"last_commit": "abc123", "timestamp": "2025-01-01T00:00:00Z"}`
- Location: `.gitlore/checkpoint.json` in the repo root or in XDG state dir
- On first run: process all commits, save HEAD hash
- On subsequent runs: process `last_commit..HEAD`, append results

### Edge cases
- **Force pushes / rebases:** `last_commit` may no longer exist in history. Detect with `git cat-file -t <hash>` and fall back to full re-scan
- **Multiple branches:** checkpoint per branch if analyzing multiple
- **Shallow clones:** `git log` will stop at the shallow boundary; detect with `git rev-parse --is-shallow-repository`

### Alternative: Date-based checkpointing
```bash
git log --since="2025-01-15T00:00:00Z" --pretty=format:...
```
Simpler but can miss commits with backdated author dates. Hash-based is more reliable.

## 5. Practical Code Patterns

### Pattern 1: Structured git log parser (recommended)

```python
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

COMMIT_SEP = '\x1e'  # record separator
FIELD_SEP = '\x00'    # field separator

@dataclass
class FileChange:
    path: str
    added: int | None    # None for binary
    deleted: int | None  # None for binary
    old_path: str | None = None  # set if rename

@dataclass
class CommitInfo:
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

GIT_LOG_FORMAT = FIELD_SEP.join([
    '%H',   # hash
    '%an',  # author name
    '%ae',  # author email
    '%aI',  # author date ISO
    '%P',   # parent hashes
    '%s',   # subject
    '%b',   # body
]) + COMMIT_SEP


def iter_commits(
    repo_path: str,
    since_hash: str | None = None,
    no_merges: bool = True,
) -> Iterator[CommitInfo]:
    """Stream commits from git log with constant memory usage."""
    cmd = [
        'git', '-C', repo_path, 'log',
        f'--pretty=format:{GIT_LOG_FORMAT}',
        '--numstat',
        '-M',  # rename detection
    ]
    if no_merges:
        cmd.append('--no-merges')
    if since_hash:
        cmd.append(f'{since_hash}..HEAD')

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        errors='replace',
    )

    buffer = ''
    for chunk in iter(lambda: proc.stdout.read(8192), ''):
        buffer += chunk
        while COMMIT_SEP in buffer:
            raw_commit, buffer = buffer.split(COMMIT_SEP, 1)
            raw_commit = raw_commit.strip()
            if not raw_commit:
                continue
            commit = _parse_commit_block(raw_commit)
            if commit:
                yield commit

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        raise RuntimeError(f'git log failed: {stderr}')


def _parse_commit_block(raw: str) -> CommitInfo | None:
    """Parse a single commit block (header + numstat lines)."""
    lines = raw.split('\n')

    # First line contains the formatted commit header
    header_line = lines[0]
    parts = header_line.split(FIELD_SEP)
    if len(parts) < 7:
        return None

    hash_, author_name, author_email, date_str, parents_str, subject, body = (
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
        FIELD_SEP.join(parts[6:]),  # body may contain NUL in theory
    )

    commit = CommitInfo(
        hash=hash_,
        author_name=author_name,
        author_email=author_email,
        author_date=datetime.fromisoformat(date_str),
        parents=parents_str.split() if parents_str else [],
        subject=subject,
        body=body.strip(),
    )

    # Remaining lines are numstat: "added\tdeleted\tpath"
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        numstat_parts = line.split('\t', 2)
        if len(numstat_parts) != 3:
            continue
        added_str, deleted_str, path = numstat_parts
        added = int(added_str) if added_str != '-' else None
        deleted = int(deleted_str) if deleted_str != '-' else None

        # Rename detection: path may be "{old => new}" or "dir/{old => new}/file"
        old_path = None
        if ' => ' in path and '{' in path:
            old_path = _expand_rename_path(path, use_old=True)
            path = _expand_rename_path(path, use_old=False)

        commit.files.append(FileChange(
            path=path,
            added=added,
            deleted=deleted,
            old_path=old_path,
        ))

    return commit


def _expand_rename_path(path: str, use_old: bool) -> str:
    """Expand git's rename format: 'dir/{old.py => new.py}/sub' -> full path."""
    import re
    def replace(m):
        old, new = m.group(1), m.group(2)
        return old if use_old else new
    return re.sub(r'\{(.+?) => (.+?)\}', replace, path)
```

### Pattern 2: Using PyDriller for quick prototyping

```python
from pydriller import Repository
from datetime import datetime

def extract_commits_pydriller(repo_path: str, since: datetime | None = None):
    """Slower but useful for prototyping or small repos."""
    kwargs = {
        'path_to_repo': repo_path,
        'only_no_merge': True,
        'order': 'reverse',  # newest first
    }
    if since:
        kwargs['since'] = since

    results = []
    for commit in Repository(**kwargs).traverse_commits():
        results.append({
            'hash': commit.hash,
            'author': commit.author.name,
            'email': commit.author.email,
            'date': commit.author_date,
            'message': commit.msg,
            'is_merge': commit.merge,
            'parents': commit.parents,
            'files': [{
                'path': f.new_path or f.old_path,
                'old_path': f.old_path if f.change_type.name == 'RENAME' else None,
                'change_type': f.change_type.name,
                'added': f.added_lines,
                'deleted': f.deleted_lines,
            } for f in commit.modified_files],
        })
    return results
```

### Pattern 3: Batch extraction with `git log --format=json`-like output

```python
import json
import subprocess

def extract_commits_json(repo_path: str, since_hash: str | None = None) -> list[dict]:
    """Use git log with a JSON-friendly format for simple extraction."""
    fmt = (
        '{"hash":"%H","author":"%an","email":"%ae",'
        '"date":"%aI","parents":"%P","subject":"%s"}'
    )
    cmd = ['git', '-C', repo_path, 'log', f'--pretty=format:{fmt}']
    if since_hash:
        cmd.append(f'{since_hash}..HEAD')

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    commits = []
    for line in result.stdout.strip().split('\n'):
        if line:
            commits.append(json.loads(line))
    return commits
```

**Warning:** Pattern 3 breaks if commit subjects contain quotes or special JSON characters. Use Pattern 1 with NUL separators for production.

## 6. Recommendations

### Primary approach: Raw `git log` subprocess with streaming parser

**Why:**
1. **Performance (quality):** Orders of magnitude faster than any Python library. 50k commits in seconds, not minutes.
2. **Cost:** Zero dependencies. git is always available. Parser is ~100 lines of code (Pattern 1 above).
3. **Latency:** Streaming via `Popen` means we can start processing immediately, no buffering entire history.
4. **Incremental:** Native support via `hash..HEAD` revision range. Trivial checkpoint system.
5. **Completeness:** Full access to git's format placeholders, rename detection, filtering.

### When to use PyDriller
- **Prototyping and testing:** Quick iteration on what data we need before building the parser.
- **Small repos (<1k commits):** The 55 commits/sec throughput is fine, and the API is nicer.
- **If we need source code content:** PyDriller gives `source_code` and `source_code_before` on modified files; doing this with raw git requires `git show <hash>:<path>` calls.
- **Process metrics:** If we ever need DMM complexity metrics, PyDriller has them built in.

### Architecture suggestion

```
git log subprocess (fast, streaming)
    |
    v
CommitInfo dataclasses (structured data)
    |
    v
Analysis layer (churn, coupling, patterns)
```

Use the streaming `iter_commits()` from Pattern 1 as the primary extraction layer. Keep PyDriller as an optional dependency for when we need source-code-level analysis (e.g., reading file contents at specific commits for deeper pattern detection).

### What NOT to use
- **GitPython:** No advantage over raw subprocess; slower, leaky, more code.
- **dulwich:** Pure Python reimplementation of git is the wrong abstraction for a tool that can assume git is installed.
- **pygit2:** libgit2 system dependency makes pip installation painful for end users.
