r"""Generate the Residual-on vs Residual-off per-architecture table for E5.

One row per architecture that has a residual choice (ff=True only:
3 attn x 2 norm x 2 bias x 2 act = 24 rows). For each criterion (any / most /
all) the row reports, across that criterion's repeats (paired by repeat index):

  * how often residual-on beats / equals / trails residual-off, and
  * the mean signed percentage difference (on - off) / off.

Output is a LaTeX ``tabular`` written to OUT_FILE, pulled into ``writup.tex``
with ``\input``. Data helpers come from gen_table_E5.py and the shared
formatting (coloured %diff) from gen_table_activation.py, so editing LOG_FILES
in gen_table_E5.py is enough to change the inputs.

Just run:
    python writeup_CE/gen_table_residual.py
"""

from pathlib import Path

from gen_table_E5 import (LOG_FILES, read_log, parse_column_label,
                          CRIT_LABEL, ATT_ORDER, CM, XM, ATTLABEL, cm)
from gen_table_activation import CRIT_ORDER, fmt_pct, fmt_count

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "table_residual.tex"

# The compared axis: residual-on ("a") versus residual-off ("b").
LABEL_A, LABEL_B = "On", "Off"


def architecture_rows():
    """Yield (attn, norms, bias, act) for every ff=True architecture (24 of them)."""
    for att in ATT_ORDER:
        for norms in [False, True]:
            for bias in [False, True]:
                for act in ["GELU", "ReLU"]:
                    yield att, norms, bias, act


def compare(files, att, norms, bias, act):
    """For one architecture and one criterion's repeat files, return
    (on_better, equal, off_better, mean_pct_diff), comparing residual on vs off
    per repeat (paired by repeat index)."""
    on_better = equal = off_better = 0
    diffs = []
    for f in files:
        on = f.get((att, True, norms, True, bias, act))    # ff_residual = True
        off = f.get((att, True, norms, False, bias, act))  # ff_residual = False
        if on is None or off is None:
            continue
        if on > off:
            on_better += 1
        elif off > on:
            off_better += 1
        else:
            equal += 1
        if off:
            diffs.append((on - off) / off * 100.0)
    mean_diff = sum(diffs) / len(diffs) if diffs else None
    return on_better, equal, off_better, mean_diff


def generate(cols):
    files_by_crit = {crit: [cv for c, cv in cols if c == crit] for crit in CRIT_ORDER}
    crits = [c for c in CRIT_ORDER if files_by_crit[c]]

    n_fixed = 4    # Attn, Norm, Bias, Act
    per_crit = 4   # On better, equal, Off better, Mean %diff
    colspec = "@{}lccc" + ("|" + "c" * per_crit) * len(crits) + "@{}"

    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]

    header1 = [""] * n_fixed
    cmidrules = []
    for j, crit in enumerate(crits):
        header1.append(r"\multicolumn{%d}{|c}{\textbf{%s}}" % (per_crit, CRIT_LABEL[crit]))
        first = n_fixed + per_crit * j + 1
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (first, first + per_crit - 1))
    lines.append(" & ".join(header1) + r" \\")
    lines.append("".join(cmidrules))

    header2 = [r"\textbf{Attn}", r"\textbf{Norm}", r"\textbf{Bias}", r"\textbf{Act}"]
    for _crit in crits:
        header2 += [r"\textbf{%s}" % LABEL_A, r"\textbf{$=$}",
                    r"\textbf{%s}" % LABEL_B, r"\textbf{Mean \%$\Delta$}"]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for ai, att in enumerate(ATT_ORDER):
        if ai > 0:
            lines.append(r"\midrule")
        block = [(norms, bias, act) for norms in [False, True]
                 for bias in [False, True] for act in ["GELU", "ReLU"]]
        for ri, (norms, bias, act) in enumerate(block):
            cells = [ATTLABEL[att], cm(norms), cm(bias), act]
            for crit in crits:
                on_better, equal, off_better, mean_diff = compare(
                    files_by_crit[crit], att, norms, bias, act)
                total = on_better + equal + off_better
                cells += [fmt_count(on_better, total), fmt_count(equal, total),
                          fmt_count(off_better, total), fmt_pct(mean_diff)]
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
