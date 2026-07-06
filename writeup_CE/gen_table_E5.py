r"""Generate the LaTeX results table for the E5 architecture sweep.

Reads the E5 experiment-log ``.jsonl`` files listed in LOG_FILES below (one
"Max Facts" column per file) and writes a LaTeX ``tabular`` to OUT_FILE, which
``writup.tex`` pulls in with ``\input``.

Just run:
    python writeup_CE/gen_table_E5.py

To change which logs are included (or their order / how many), edit the
LOG_FILES list below. Columns appear in the order listed; consecutive files
sharing a criterion (any / most / all, detected from the filename) are grouped
under a shared header, and within each group the row-wise maximum is bolded.
"""

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent     # writeup_CE/
E5_DIR = HERE.parent / "E5"                 # sibling folder holding the logs

# ── Edit this list to choose which logs become columns (in order) ────────────
LOG_FILES = [
    E5_DIR / "experiment_log_CE_any_(1).jsonl",
    E5_DIR / "experiment_log_CE_any_(2).jsonl",
    E5_DIR / "experiment_log_CE_any_(3).jsonl",
    E5_DIR / "experiment_log_CE_any_(4).jsonl",
    E5_DIR / "experiment_log_CE_most_(1).jsonl",
    E5_DIR / "experiment_log_CE_most_(2).jsonl",
    E5_DIR / "experiment_log_CE_most_(3).jsonl",
    E5_DIR / "experiment_log_CE_most_(4).jsonl",
    E5_DIR / "experiment_log_CE_all_(1).jsonl",
    E5_DIR / "experiment_log_CE_all_(2).jsonl",
    E5_DIR / "experiment_log_CE_all_(3).jsonl",
    E5_DIR / "experiment_log_CE_all_(4).jsonl",
]
OUT_FILE = HERE / "table_E5.tex"           # written next to writup.tex

# A value is boxed as an outlier if it differs from the median of its criterion
# group's repeats (in the same row) by more than this fraction.
OUTLIER_FRAC = 0.20

CM = r"\cmark"
XM = r"\xmark"
ATTLABEL = {"none": "2Emb", "uni": "Unif Attn", "full": "Lrn Attn"}
CRIT_LABEL = {"any": "Any of 11", "most": "Most of 11", "all": "All of 11"}
ATT_ORDER = ["none", "uni", "full"]


def att_of(settings):
    if settings["attention"]:
        return "full" if not settings["qk_is_one"] else "uni"
    return "none"


def key_of(settings):
    return (att_of(settings), settings["ff"], settings["norms"],
            settings["ff_residual"], settings["bias"], settings["ff_activation_type"])


def parse_column_label(path):
    """Return (criterion, sub_label) parsed from a log filename.

    e.g. experiment_log_CE_any_(1).jsonl -> ("any", "(1)").
    criterion is None if no any/most/all token is found; sub_label is "" if no
    trailing "(k)" or number is present.
    """
    stem = Path(path).stem
    crit = next((c for c in ("any", "most", "all") if re.search(rf"_{c}(?:_|\b)", stem)), None)
    m = re.search(r"(\([^)]*\)|\d+)\s*$", stem)
    sub = m.group(1) if m else ""
    return crit, sub


def read_log(path):
    """Map architecture key -> max_facts for one .jsonl log."""
    vals = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        vals[key_of(d["settings"])] = d["max_facts"]
    return vals


def build_groups(col_labels):
    """Group consecutive columns sharing a criterion.

    Returns a list of (title, [col_indices], [sub_labels]). For grouped columns
    a missing sub-label falls back to a 1-based "(k)" index within the group.
    """
    groups = []
    for i, (crit, sub) in enumerate(col_labels):
        if groups and groups[-1][0] == crit and crit is not None:
            groups[-1][1].append(i)
            groups[-1][2].append(sub)
        else:
            groups.append((crit, [i], [sub]))
    titled = []
    for crit, idxs, subs in groups:
        title = CRIT_LABEL.get(crit, "Max Facts" if crit is None else crit)
        subs = [s if s else f"({k})" for k, s in enumerate(subs, start=1)]
        titled.append((title, idxs, subs))
    return titled


def fmt_row_values(key, col_values, groups):
    """Render the data cells for one architecture row.

    Within each criterion group the row-wise maximum is bolded, and any value
    more than OUTLIER_FRAC from the group's median is boxed (so an outlier that
    is also the maximum renders as \\fbox{\\textbf{...}}).
    """
    cells = [""] * len(col_values)
    for _title, idxs, _subs in groups:
        present = [col_values[i].get(key) for i in idxs]
        nums = [v for v in present if v is not None]
        gmax = max(nums) if nums else None
        med = statistics.median(nums) if len(nums) >= 2 else None
        for i, v in zip(idxs, present):
            if v is None:
                cells[i] = "--"
                continue
            s = str(v)
            if v == gmax:
                s = r"\textbf{" + s + "}"
            if med is not None and med > 0 and abs(v - med) > OUTLIER_FRAC * med:
                s = r"\fbox{" + s + "}"
            cells[i] = s
    return cells


def cm(flag):
    return CM if flag else XM


def generate(col_values, groups):
    n_data = len(col_values)
    n_fixed = 6  # Mixing, MLP, Norms, Res, Bias, Act

    lines = []
    # A solid vertical rule before each data group separates Act from the first
    # group and each criterion group from the next.
    colspec = "@{}cccccc" + "".join("|" + "c" * len(idxs) for _t, idxs, _s in groups) + "@{}"
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")

    # Grouped header row + cmidrules.
    header1 = [""] * n_fixed
    cmidrules = []
    for title, idxs, _subs in groups:
        span = len(idxs)
        header1.append(r"\multicolumn{%d}{|c}{\textbf{%s}}" % (span, title))
        first = n_fixed + idxs[0] + 1
        last = n_fixed + idxs[-1] + 1
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (first, last))
    lines.append(" & ".join(header1) + r" \\")
    lines.append("".join(cmidrules))

    # Sub-header row (fixed column names + per-column labels).
    sub_by_col = [""] * n_data
    for _title, idxs, subs in groups:
        for i, s in zip(idxs, subs):
            sub_by_col[i] = s
    header2 = [r"\textbf{Mixing}", r"\textbf{MLP}", r"\textbf{Norms}",
               r"\textbf{Res}", r"\textbf{Bias}", r"\textbf{Act}"] + sub_by_col
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # Body: ff=True block then ff=False block, per attention variant.
    for ai, att in enumerate(ATT_ORDER):
        # Each (norms, res) pair is a 4-row group (bias x activation); a dashed
        # line separates the groups, i.e. wherever Res changes value.
        ff_true_blocks = [(norms, res) for norms in [False, True] for res in [False, True]]
        for bi, (norms, res) in enumerate(ff_true_blocks):
            for bias in [False, True]:
                for act in ["GELU", "ReLU"]:
                    key = (att, True, norms, res, bias, act)
                    cells = fmt_row_values(key, col_values, groups)
                    lines.append(f"{ATTLABEL[att]} & {CM} & {cm(norms)} & {cm(res)} & "
                                 f"{cm(bias)} & {act} & " + " & ".join(cells) + r" \\")
            if bi < len(ff_true_blocks) - 1:
                lines.append(r"\hdashline[1pt/3pt]")
        lines.append(r"\hdashline[1pt/3pt]")
        for norms in [False, True]:
            key = (att, False, norms, True, True, "ReLU")
            cells = fmt_row_values(key, col_values, groups)
            lines.append(f"{ATTLABEL[att]} & {XM} & {cm(norms)} & N/A & N/A & N/A & "
                         + " & ".join(cells) + r" \\")
        if ai < len(ATT_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    log_paths = [Path(p) for p in LOG_FILES]
    missing = [p for p in log_paths if not p.exists()]
    if missing:
        raise SystemExit("log file(s) not found:\n" + "\n".join(f"  {p}" for p in missing))

    col_values = [read_log(p) for p in log_paths]
    col_labels = [parse_column_label(p) for p in log_paths]
    groups = build_groups(col_labels)

    table = generate(col_values, groups)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(table, encoding="utf-8")

    print(f"Wrote {OUT_FILE}  ({len(log_paths)} column(s)):")
    for p, (crit, sub) in zip(log_paths, col_labels):
        print(f"  {p.name:42}  ->  {CRIT_LABEL.get(crit, crit or '?')} {sub}")


if __name__ == "__main__":
    main()
