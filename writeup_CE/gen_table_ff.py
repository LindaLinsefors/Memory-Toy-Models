r"""Generate the feed-forward on vs off table for E5.

The ff=False runs only exist for the 6 (attention x norm) combinations (with
res/bias/act fixed at the leftover on/on/ReLU values), so the FF on/off
comparison is over exactly those 6 matched architectures. For each criterion
(any / most / all) the row reports, across that criterion's repeats (paired by
repeat index):

  * how often FF-on beats / equals / trails FF-off, and
  * the mean signed percentage difference (on - off) / off.

Output is a LaTeX ``tabular`` written to OUT_FILE, pulled into ``writup.tex``
with ``\input``.

Just run:
    python writeup_CE/gen_table_ff.py
"""

from pathlib import Path

from gen_table_E5 import (LOG_FILES, read_log, parse_column_label,
                          CRIT_LABEL, ATT_ORDER, ATTLABEL, cm)
from gen_table_activation import CRIT_ORDER, fmt_pct, fmt_count

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "table_ff.tex"

# Settings held fixed across the FF on/off comparison (those present on the
# ff=False runs in the logs).
FIX_RES, FIX_BIAS, FIX_ACT = True, True, "ReLU"


def compare(files, att, norms):
    """Repeats where FF-on beats / equals / trails FF-off, and mean %diff."""
    on_better = equal = off_better = 0
    diffs = []
    for f in files:
        on = f.get((att, True, norms, FIX_RES, FIX_BIAS, FIX_ACT))    # ff = True
        off = f.get((att, False, norms, FIX_RES, FIX_BIAS, FIX_ACT))  # ff = False
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

    n_fixed = 2    # Attn, Norm
    per_crit = 4   # On better, equal, Off better, Mean %diff
    colspec = "@{}lc" + ("|" + "c" * per_crit) * len(crits) + "@{}"

    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]

    header1 = [""] * n_fixed
    cmidrules = []
    for j, crit in enumerate(crits):
        header1.append(r"\multicolumn{%d}{|c}{\textbf{%s}}" % (per_crit, CRIT_LABEL[crit]))
        first = n_fixed + per_crit * j + 1
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (first, first + per_crit - 1))
    lines.append(" & ".join(header1) + r" \\")
    lines.append("".join(cmidrules))

    header2 = [r"\textbf{Attn}", r"\textbf{Norm}"]
    for _crit in crits:
        header2 += [r"\textbf{On}", r"\textbf{$=$}", r"\textbf{Off}", r"\textbf{Mean \%$\Delta$}"]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for ai, att in enumerate(ATT_ORDER):
        for norms in [False, True]:
            cells = [ATTLABEL[att], cm(norms)]
            for crit in crits:
                on_better, equal, off_better, mean_diff = compare(
                    files_by_crit[crit], att, norms)
                total = on_better + equal + off_better
                cells += [fmt_count(on_better, total), fmt_count(equal, total),
                          fmt_count(off_better, total), fmt_pct(mean_diff)]
            lines.append(" & ".join(cells) + r" \\")
        if ai < len(ATT_ORDER) - 1:
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
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
