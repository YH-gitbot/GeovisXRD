# validation_scripts/_fmt.py
"""
Shared console-formatting helpers for validate_*.py scripts.

Tags used across all scripts
─────────────────────────────
  [INPUT]    — artifact or dataset this script consumes
  [OUTPUT]   — artifact this script writes to disk
  [OK]       — assertion / check passed
  [WARN]     — non-fatal issue; execution continues
  [SKIP]     — step omitted (missing dep, failed assertion, etc.)
  [REUSE]    — upstream artifact loaded without rebuilding
  [BUILD]    — prerequisite built from scratch as a data-prep step
  [FALLBACK] — synthetic / generated replacement used (no real input found)
"""

W = 64   # console line width


# ─────────────────────────────────────────────────────────────────
# Header / section dividers
# ─────────────────────────────────────────────────────────────────

def _multiline(label, items):
    """Yield formatted lines for a header block field (1+ items)."""
    if isinstance(items, str):
        items = [items]
    for i, item in enumerate(items):
        prefix = f"  {label:<9}" if i == 0 else "           "
        yield f"{prefix}: {item}"


def header(script, purpose, upstream, downstream):
    """Top-of-script banner.

    Parameters
    ----------
    script     : str   — module / file being validated
    purpose    : str   — one-line description
    upstream   : str | list[str]  — inputs consumed
    downstream : str | list[str]  — outputs produced (for next scripts)
    """
    print("═" * W)
    print(f"  VALIDATE : {script}")
    print(f"  Purpose  : {purpose}")
    print("─" * W)
    for line in _multiline("Upstream", upstream):
        print(line)
    for line in _multiline("Produces", downstream):
        print(line)
    print("═" * W)


def section(n, title):
    """Numbered section divider."""
    pad = "─" * max(0, W - len(str(n)) - len(title) - 14)
    print(f"\n── Section {n}: {title} {pad}")


def part(title):
    """Bold part divider (within a script, above section level)."""
    print(f"\n{'━' * W}")
    print(f"  {title}")
    print("━" * W)


def rule():
    """Thin horizontal rule."""
    print("─" * W)


# ─────────────────────────────────────────────────────────────────
# Status-tagged print lines
# ─────────────────────────────────────────────────────────────────

def _tag(tag, msg, path=None):
    t = f"[{tag}]"
    print(f"  {t:<11} {msg}")
    if path is not None:
        print(f"  {'':11} → {path}")


def ok(msg, path=None):        _tag("OK",       msg, path)
def warn(msg, path=None):      _tag("WARN",     msg, path)
def skip(msg, path=None):      _tag("SKIP",     msg, path)
def inp(msg, path=None):       _tag("INPUT",    msg, path)
def out(msg, path=None):       _tag("OUTPUT",   msg, path)
def reuse(msg, path=None):     _tag("REUSE",    msg, path)
def build(msg, path=None):     _tag("BUILD",    msg, path)
def fallback(msg, path=None):  _tag("FALLBACK", msg, path)


# ─────────────────────────────────────────────────────────────────
# Closing artifact summary
# ─────────────────────────────────────────────────────────────────

def artifact_summary(inputs_used, outputs_written, reusable_by=None):
    """Print the structured closing artifact summary.

    Parameters
    ----------
    inputs_used    : list[str]  — what was consumed (dataset names, paths)
    outputs_written: list[str]  — what was written (path + description)
    reusable_by    : list[str] or None  — which downstream scripts can reuse outputs
    """
    print("\n" + "═" * W)
    print("  ARTIFACT SUMMARY")
    print("─" * W)
    print("  [INPUTS USED]")
    for item in inputs_used:
        print(f"    {item}")
    print()
    print("  [OUTPUTS WRITTEN]")
    for item in outputs_written:
        print(f"    {item}")
    if reusable_by:
        print()
        print("  [REUSABLE BY]")
        for item in reusable_by:
            print(f"    {item}")
    print("═" * W)
    print("  DONE")
    print("═" * W)
