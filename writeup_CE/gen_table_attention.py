r"""Generate the attention-variant pairwise comparison table for E5.

Token mixing has three values (2Emb / Unif Attn / Lrn Attn). This table
compares them pairwise. One row per architecture that has a mixing choice; the
ff=True block has 16 rows (2 norm x 2 res x 2 bias x 2 act) and the ff=False
block adds 2 rows (norm only, since the no-FF runs vary only norm). For each
criterion (any / most / all) and each pair (2E>U, U>L, 2E>L) the row gives:

  * the number of repeats (out of 4) in which the left variant beats the right
    (paired by repeat index),
  * the mean signed percentage difference (left - right) / right, and
  * the mean signed absolute difference left - right (in facts).

Output is three vertically stacked ``tabular`` blocks (one per criterion,
wrapped in an outer one-column tabular) written to OUT_FILE, pulled into
``writup.tex`` with ``\input``. Data helpers come from gen_table_E5.py and the
shared formatting from gen_table_activation.py.

Just run:
    python writeup_CE/gen_table_attention.py
"""

from pathlib import Path

from gen_table_E5 import (LOG_FILES, read_log, parse_column_label,
                          CRIT_LABEL, ATT_ORDER, CM, XM, ATTLABEL, cm)
from gen_table_activation import CRIT_ORDER, fmt_pct, fmt_abs, fmt_count

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "table_attention.tex"

# Pairwise comparisons (left vs right), following the 2Emb > Unif Attn >
# Lrn Attn order.
PAIRS = [("none", "uni"), ("uni", "full"), ("none", "full")]
ABBR = {"none": "2E", "uni": "U", "full": "L"}  # compact header labels

# The ff=False runs in the logs carry these (leftover) res/bias/act settings.
FFFALSE_RES, FFFALSE_BIAS, FFFALSE_ACT = True, True, "ReLU"


def compare_pair(files, a, b, ff, norms, res, bias, act):
    """Repeats where mixing variant a beats b, mean %diff (a-b)/b, and mean
    absolute diff a-b (in facts)."""
    wins = 0
    n = 0
    diffs = []
    adiffs = []
    for f in files:
        va = f.get((a, ff, norms, res, bias, act))
        vb = f.get((b, ff, norms, res, bias, act))
        if va is None or vb is None:
            continue
        n += 1
        if va > vb:
            wins += 1
        adiffs.append(va - vb)
        if vb:
            diffs.append((va - vb) / vb * 100.0)
    mean_diff = sum(diffs) / len(diffs) if diffs else None
    mean_abs = sum(adiffs) / len(adiffs) if adiffs else None
    return wins, n, mean_diff, mean_abs


def one_table(crit, files):
    """Build the tabular for a single criterion; returns its LaTeX source."""
    n_fixed = 5                # Norms, MLP, Res, Bias, Act
    per_crit = 3 * len(PAIRS)  # (count, %diff, abs diff) per pair
    colspec = "@{}ccccc|" + "c" * per_crit + "@{}"

    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]

    header1 = [""] * n_fixed + \
        [r"\multicolumn{%d}{|c}{\textbf{%s}}" % (per_crit, CRIT_LABEL[crit])]
    lines.append(" & ".join(header1) + r" \\")
    lines.append(r"\cmidrule(lr){%d-%d}" % (n_fixed + 1, n_fixed + per_crit))

    header2 = [r"\textbf{Norms}", r"\textbf{MLP}", r"\textbf{Res}", r"\textbf{Bias}", r"\textbf{Act}"]
    for a, b in PAIRS:
        header2 += [r"\textbf{%s$>$%s}" % (ABBR[a], ABBR[b]),
                    r"\textbf{\%$\Delta$}", r"\textbf{$\Delta$}"]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    def data_cells(ff, norms, res, bias, act):
        out = []
        for a, b in PAIRS:
            wins, n, mean_diff, mean_abs = compare_pair(
                files, a, b, ff, norms, res, bias, act)
            out += [fmt_count(wins, n), fmt_pct(mean_diff), fmt_abs(mean_abs)]
        return out

    # ff=True block: 16 rows over (norms, res, bias, act).
    block = [(norms, res, bias, act) for norms in [False, True]
             for res in [False, True] for bias in [False, True]
             for act in ["GELU", "ReLU"]]
    for ri, (norms, res, bias, act) in enumerate(block):
        if ri > 0 and ri % 8 == 0:        # solid rule when Norm changes (8 rows)
            lines.append(r"\midrule")
        cells = [cm(norms), CM, cm(res), cm(bias), act] + data_cells(True, norms, res, bias, act)
        lines.append(" & ".join(cells) + r" \\")
        if ri % 2 == 1 and ri < len(block) - 1 and (ri + 1) % 8 != 0:
            lines.append(r"\hdashline[1pt/3pt]")

    # ff=False block: only norm varies there.
    lines.append(r"\midrule")
    for norms in [False, True]:
        cells = [cm(norms), XM, "N/A", "N/A", "N/A"] + \
            data_cells(False, norms, FFFALSE_RES, FFFALSE_BIAS, FFFALSE_ACT)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate(cols):
    files_by_crit = {crit: [cv for c, cv in cols if c == crit] for crit in CRIT_ORDER}
    crits = [c for c in CRIT_ORDER if files_by_crit[c]]

    # One table per criterion, stacked vertically. Nesting them in a
    # one-column tabular keeps the block a single unit for \input.
    lines = [r"\begin{tabular}{@{}c@{}}"]
    for i, crit in enumerate(crits):
        lines.append(one_table(crit, files_by_crit[crit]))
        if i < len(crits) - 1:
            lines.append(r"\\[3ex]")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    log_paths = [Path(p) for p in LOG_FILES]
    missing = [p for p in log_paths if not p.exists()]
    if missing:
        raise SystemExit("log file(s) not found:\n" + "\n".join(f"  {p}" for p in missing))

    cols = [(parse_column_label(p)[0], read_log(p)) for p in log_paths]
    OUT_FILE.write_text(generate(cols), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
