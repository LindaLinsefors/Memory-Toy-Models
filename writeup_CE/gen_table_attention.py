r"""Generate the attention-variant pairwise comparison table for E5.

Attention has three values (None / Unif / Full). This table compares them
pairwise. One row per architecture that has an attention choice; the ff=True
block has 16 rows (2 norm x 2 res x 2 bias x 2 act) and the ff=False block adds
2 rows (norm only, since the no-FF runs vary only norm). For each criterion
(any / most / all) and each pair (None>Unif, Unif>Full, None>Full) the row gives:

  * the number of repeats (out of 4) in which the left variant beats the right
    (paired by repeat index), and
  * the mean signed percentage difference (left - right) / right.

Output is a LaTeX ``tabular`` written to OUT_FILE, pulled into ``writup.tex``
with ``\input``. Data helpers come from gen_table_E5.py and the shared
formatting from gen_table_activation.py.

Just run:
    python writeup_CE/gen_table_attention.py
"""

from pathlib import Path

from gen_table_E5 import (LOG_FILES, read_log, parse_column_label,
                          CRIT_LABEL, ATT_ORDER, CM, XM, ATTLABEL, cm)
from gen_table_activation import CRIT_ORDER, fmt_pct, fmt_count

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "table_attention.tex"

# Pairwise comparisons (left vs right), following the None > Unif > Full order.
PAIRS = [("none", "uni"), ("uni", "full"), ("none", "full")]
ABBR = {"none": "N", "uni": "U", "full": "F"}  # compact header labels

# The ff=False runs in the logs carry these (leftover) res/bias/act settings.
FFFALSE_RES, FFFALSE_BIAS, FFFALSE_ACT = True, True, "ReLU"


def compare_pair(files, a, b, ff, norms, res, bias, act):
    """Repeats where attention variant a beats b, and mean %diff (a-b)/b."""
    wins = 0
    n = 0
    diffs = []
    for f in files:
        va = f.get((a, ff, norms, res, bias, act))
        vb = f.get((b, ff, norms, res, bias, act))
        if va is None or vb is None:
            continue
        n += 1
        if va > vb:
            wins += 1
        if vb:
            diffs.append((va - vb) / vb * 100.0)
    mean_diff = sum(diffs) / len(diffs) if diffs else None
    return wins, n, mean_diff


def generate(cols):
    files_by_crit = {crit: [cv for c, cv in cols if c == crit] for crit in CRIT_ORDER}
    crits = [c for c in CRIT_ORDER if files_by_crit[c]]

    n_fixed = 5            # Norm, FF, Res, Bias, Act
    per_crit = 2 * len(PAIRS)  # (count, %diff) per pair
    colspec = "@{}ccccc" + ("|" + "c" * per_crit) * len(crits) + "@{}"

    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]

    header1 = [""] * n_fixed
    cmidrules = []
    for j, crit in enumerate(crits):
        header1.append(r"\multicolumn{%d}{|c}{\textbf{%s}}" % (per_crit, CRIT_LABEL[crit]))
        first = n_fixed + per_crit * j + 1
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (first, first + per_crit - 1))
    lines.append(" & ".join(header1) + r" \\")
    lines.append("".join(cmidrules))

    header2 = [r"\textbf{Norm}", r"\textbf{FF}", r"\textbf{Res}", r"\textbf{Bias}", r"\textbf{Act}"]
    for _crit in crits:
        for a, b in PAIRS:
            header2 += [r"\textbf{%s$>$%s}" % (ABBR[a], ABBR[b]), r"\textbf{$\Delta$}"]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    def data_cells(ff, norms, res, bias, act):
        out = []
        for crit in crits:
            for a, b in PAIRS:
                wins, n, mean_diff = compare_pair(
                    files_by_crit[crit], a, b, ff, norms, res, bias, act)
                out += [fmt_count(wins, n), fmt_pct(mean_diff)]
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
