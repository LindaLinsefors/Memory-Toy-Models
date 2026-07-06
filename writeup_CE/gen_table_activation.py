r"""Generate the GELU-vs-ReLU per-architecture table for the E5 sweep.

One row per architecture that has an activation choice (ff=True only:
3 attn x 2 norm x 2 res x 2 bias = 24 rows). For each criterion (any / most /
all) the row reports, across that criterion's repeats:

  * how often GELU beats ReLU, as "wins / n_repeats" (paired by repeat index,
    since each log file measures both GELU and ReLU under identical settings),
  * the mean signed percentage difference (GELU - ReLU) / ReLU over the repeats,
  * the mean signed absolute difference GELU - ReLU (in facts) over the repeats.

Output is a LaTeX ``tabular`` written to OUT_FILE, pulled into ``writup.tex``
with ``\input``. Input logs and parsing are reused from gen_table_E5.py, so
editing LOG_FILES there is enough.

Just run:
    python writeup_CE/gen_table_activation.py
"""

from pathlib import Path

from gen_table_E5 import (LOG_FILES, read_log, parse_column_label,
                          CRIT_LABEL, ATT_ORDER, CM, XM, ATTLABEL, cm)

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "table_activation.tex"

CRIT_ORDER = ["any", "most", "all"]


def architecture_rows():
    """Yield (attn, norms, res, bias) for every ff=True architecture (24 of them)."""
    for att in ATT_ORDER:
        for norms in [False, True]:
            for res in [False, True]:
                for bias in [False, True]:
                    yield att, norms, res, bias


# Colour for the Mean %diff cell, by sign (xcolor is loaded in writup.tex).
POS_COLOR = "green!55!black"
NEG_COLOR = "red"


def compare(files, att, norms, res, bias):
    """For one architecture and one criterion's repeat files, return
    (gelu_better, equal, relu_better, mean_pct_diff, mean_abs_diff), comparing
    GELU vs ReLU per repeat (paired by repeat index)."""
    g_better = equal = r_better = 0
    diffs = []
    adiffs = []
    for f in files:
        g = f.get((att, True, norms, res, bias, "GELU"))
        r = f.get((att, True, norms, res, bias, "ReLU"))
        if g is None or r is None:
            continue
        if g > r:
            g_better += 1
        elif r > g:
            r_better += 1
        else:
            equal += 1
        adiffs.append(g - r)
        if r:
            diffs.append((g - r) / r * 100.0)
    mean_diff = sum(diffs) / len(diffs) if diffs else None
    mean_abs = sum(adiffs) / len(adiffs) if adiffs else None
    return g_better, equal, r_better, mean_diff, mean_abs


def fmt_pct(mean_diff):
    """Colour the mean %diff by sign: green if GELU ahead, red if ReLU ahead."""
    if mean_diff is None:
        return "--"
    s = f"${mean_diff:+.1f}\\%$"
    if mean_diff > 0:
        return r"\textcolor{%s}{%s}" % (POS_COLOR, s)
    if mean_diff < 0:
        return r"\textcolor{%s}{%s}" % (NEG_COLOR, s)
    return s


def fmt_abs(mean_diff):
    """Colour the mean absolute difference (in facts) by sign, like fmt_pct."""
    if mean_diff is None:
        return "--"
    if abs(mean_diff - round(mean_diff)) < 1e-9:
        s = f"${round(mean_diff):+d}$"
    else:
        s = f"${mean_diff:+.1f}$"
    if mean_diff > 0:
        return r"\textcolor{%s}{%s}" % (POS_COLOR, s)
    if mean_diff < 0:
        return r"\textcolor{%s}{%s}" % (NEG_COLOR, s)
    return s


def fmt_count(count, total):
    """Blank for zero; bold a count when every repeat agrees (count == total)."""
    if count == 0:
        return ""
    return r"\textbf{%d}" % count if total and count == total else str(count)


def generate(cols):
    files_by_crit = {crit: [cv for c, cv in cols if c == crit] for crit in CRIT_ORDER}
    crits = [c for c in CRIT_ORDER if files_by_crit[c]]

    n_fixed = 4    # Mixing, Norms, Res, Bias
    per_crit = 5   # GELU better, equal, ReLU better, Mean %diff, Mean abs diff
    colspec = "@{}lccc" + ("|" + "c" * per_crit) * len(crits) + "@{}"

    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]

    # Grouped header: one criterion spanning its sub-columns.
    header1 = [""] * n_fixed
    cmidrules = []
    for j, crit in enumerate(crits):
        header1.append(r"\multicolumn{%d}{|c}{\textbf{%s}}" % (per_crit, CRIT_LABEL[crit]))
        first = n_fixed + per_crit * j + 1
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (first, first + per_crit - 1))
    lines.append(" & ".join(header1) + r" \\")
    lines.append("".join(cmidrules))

    header2 = [r"\textbf{Mixing}", r"\textbf{Norms}", r"\textbf{Res}", r"\textbf{Bias}"]
    for _crit in crits:
        header2 += [r"\textbf{G$>$R}", r"\textbf{$=$}", r"\textbf{R$>$G}",
                    r"\textbf{Mean \%$\Delta$}", r"\textbf{$\Delta$}"]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for ai, att in enumerate(ATT_ORDER):
        if ai > 0:
            lines.append(r"\midrule")
        block = [(norms, res, bias) for norms in [False, True]
                 for res in [False, True] for bias in [False, True]]
        for ri, (norms, res, bias) in enumerate(block):
            cells = [ATTLABEL[att], cm(norms), cm(res), cm(bias)]
            for crit in crits:
                g_better, equal, r_better, mean_diff, mean_abs = compare(
                    files_by_crit[crit], att, norms, res, bias)
                total = g_better + equal + r_better
                cells += [fmt_count(g_better, total), fmt_count(equal, total),
                          fmt_count(r_better, total), fmt_pct(mean_diff),
                          fmt_abs(mean_abs)]
            lines.append(" & ".join(cells) + r" \\")
            # Dotted line after every second row (not before the block's \midrule).
            if ri % 2 == 1 and ri < len(block) - 1:
                lines.append(r"\hdashline[1pt/3pt]")

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
    print(f"Wrote {OUT_FILE}  ({sum(1 for _ in architecture_rows())} architecture rows)")


if __name__ == "__main__":
    main()
