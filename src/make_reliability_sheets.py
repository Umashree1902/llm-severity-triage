"""Create the blind labelling workbooks for the human label reliability check (152 defects, 38 per severity).

Usage:  python src/make_reliability_sheets.py
Output: data/reliability/Severity_Labelling_Reviewer_A.xlsx, ..._Reviewer_B.xlsx, key.csv

Each reviewer gets the same 152 defects under neutral item codes (R001-R152), in a different random order, with
only the information available at filing time (no Bugzilla id, no severity, no priority, no status, no model
output). key.csv maps item codes to Bugzilla ids and the human labels; it is git-ignored and is regenerated
identically by this script (fixed seed), so it never needs to be shared with reviewers.
"""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from common import PROCESSED, ROOT, SEED

OUT = ROOT / "data" / "reliability"
N = 152
FONT = "Arial"
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEAD = PatternFill("solid", fgColor="DDE6F2")
THIN = Side(style="thin", color="BFC5CE")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

DEFINITIONS = [
    ("S1", "Catastrophic", "Blocks development or testing, may affect more than 25% of users, causes data loss, "
           "potential chemspill or security risk, and has no workaround."),
    ("S2", "Serious", "Major functionality or product severely impaired, and no satisfactory workaround exists."),
    ("S3", "Normal", "Blocks non-critical functionality, and a workaround exists."),
    ("S4", "Small/Trivial", "Minor significance, cosmetic issue, or low or no impact to users."),
]


def instructions(ws, reviewer):
    ws.title = "Instructions"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 95
    r = 2
    ws.cell(r, 2, f"Defect severity labelling: Reviewer {reviewer}").font = Font(name=FONT, size=16, bold=True)
    r += 2
    lines = [
        "Thank you for helping. You will assign a severity to 152 software defect reports from Mozilla's public bug "
        "tracker (Firefox, Thunderbird and related products), filed between January 2025 and June 2026.",
        "Your labels are compared with a second reviewer's and with the severity Mozilla's triagers assigned, to "
        "measure how consistently people label severity. There are no right or wrong answers for this check: label "
        "each report as you would in a real triage meeting.",
        "",
        "How to label",
        "1. Open the Labels tab. Each row is one defect report: summary, description and filing details.",
        "2. Read the summary and description, then choose a severity (S1-S4) from the drop-down in the yellow "
        "Severity column, using the definitions below.",
        "3. Choose how confident you are (1 = unsure, 2 = fairly sure, 3 = sure) in the yellow Confidence column.",
        "4. Optional: add a short note if a report was hard to judge (for example, 'unclear if a workaround exists').",
        "",
        "Rules (please follow them, they are what makes the check valid)",
        "- Work alone. Do not discuss the reports or your labels with the other reviewer until both of you have "
        "returned your sheet.",
        "- Do not look the reports up on bugzilla.mozilla.org or search for them online; the real severity is shown "
        "there.",
        "- Do not use an AI assistant or any tool to suggest labels.",
        "- Judge only from what is in the sheet. If information is missing, decide as you would at triage.",
        "- Take breaks; most people need 2-3 hours in total (about one minute per report). Save as you go.",
        "",
        "When you are done: save the file without renaming it and send it back.",
    ]
    for t in lines:
        c = ws.cell(r, 2, t)
        c.font = Font(name=FONT, size=11, bold=t in ("How to label", "Rules (please follow them, they are what "
                                                     "makes the check valid)"))
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 15 if len(t) < 110 else 30
        r += 1
    r += 1
    ws.cell(r, 2, "Severity definitions (Mozilla)").font = Font(name=FONT, size=12, bold=True)
    r += 1
    for j, h in enumerate(["Severity", "Name", "Definition"]):
        c = ws.cell(r, 2 + j, h); c.font = Font(name=FONT, bold=True); c.fill = HEAD; c.border = BOX
    for code, name, d in DEFINITIONS:
        r += 1
        for j, v in enumerate([code, name, d]):
            c = ws.cell(r, 2 + j, v); c.font = Font(name=FONT); c.border = BOX
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 30
    r += 2
    ws.cell(r, 2, "Legend").font = Font(name=FONT, size=12, bold=True)
    r += 1
    c = ws.cell(r, 2, ""); c.fill = YELLOW; c.border = BOX
    ws.cell(r, 3, "Yellow cells: fill these in (Labels tab, columns I-K). Do not change any other cell.").font = Font(name=FONT)
    ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=4)
    r += 2
    ws.cell(r, 2, "Example (made-up report, not part of the 152)").font = Font(name=FONT, size=12, bold=True)
    r += 1
    ex = [("Summary", "Bookmarks toolbar icons blurry on high-DPI screens"),
          ("Description", "Steps: set display scaling to 200%, show the bookmarks toolbar. Actual: favicons look "
                          "blurry. Expected: sharp icons. Everything still works."),
          ("Severity", "S4"), ("Confidence", "3"), ("Notes", "Cosmetic only")]
    for k, v in ex:
        a = ws.cell(r, 2, k); a.font = Font(name=FONT, bold=True); a.border = BOX
        b = ws.cell(r, 3, v); b.font = Font(name=FONT); b.border = BOX
        b.alignment = Alignment(wrap_text=True, vertical="top")
        if k in ("Severity", "Confidence", "Notes"):
            b.fill = YELLOW
        ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 30 if k == "Description" else 15
        r += 1
    r += 1
    ws.cell(r, 2, "Progress").font = Font(name=FONT, size=12, bold=True)
    c = ws.cell(r, 3, f'=COUNTIF(Labels!I3:I{N + 2},"S*")&" of {N} labelled"')
    c.font = Font(name=FONT, bold=True)


def labels(wb, items, reviewer):
    ws = wb.create_sheet("Labels")
    ws.freeze_panes = "C3"
    heads = ["Item", "Product", "Component", "Platform", "Operating system", "Keywords", "Summary", "Description",
             "Severity (S1-S4)", "Confidence (1-3)", "Notes (optional)"]
    widths = [8, 14, 16, 10, 12, 14, 38, 80, 12, 12, 26]
    ws.cell(1, 1, f"Reviewer {reviewer}: fill in the yellow columns only. Definitions are on the Instructions "
                  f"tab.").font = Font(name=FONT, italic=True, color="5B6475")
    for j, (h, w) in enumerate(zip(heads, widths), 1):
        c = ws.cell(2, j, h); c.font = Font(name=FONT, bold=True); c.fill = HEAD; c.border = BOX
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[c.column_letter].width = w
    sev = DataValidation(type="list", formula1='"S1,S2,S3,S4"', allow_blank=True,
                         error="Choose S1, S2, S3 or S4", errorTitle="Severity")
    conf = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True,
                          error="Choose 1, 2 or 3", errorTitle="Confidence")
    ws.add_data_validation(sev); ws.add_data_validation(conf)
    for i, row in enumerate(items.itertuples(), start=3):
        vals = [row.item, row.product, row.component, row.platform, row.op_sys, row.keywords, row.summary,
                row.description_llm]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, "" if pd.isna(v) else str(v))
            c.font = Font(name=FONT, size=10); c.border = BOX
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for j in (9, 10, 11):
            c = ws.cell(i, j); c.fill = YELLOW; c.border = BOX
            c.font = Font(name=FONT, size=11, bold=j == 9)
            c.alignment = Alignment(horizontal="center" if j < 11 else "left", vertical="top", wrap_text=True)
        sev.add(f"I{i}"); conf.add(f"J{i}")
        ws.row_dimensions[i].height = min(400, 15 * max(2, len(str(row.description_llm)) // 95 + 1))
    ws.auto_filter.ref = f"A2:K{len(items) + 2}"


def main():
    s = pd.read_csv(PROCESSED / "sample.csv")
    r = s[s.in_reliability.astype(bool)].copy()
    assert len(r) == N, len(r)
    r = r.sample(frac=1, random_state=SEED).reset_index(drop=True)
    r["item"] = [f"R{i:03d}" for i in range(1, N + 1)]
    for c in ("platform", "op_sys", "keywords"):
        r[c] = r[c].replace({"Unspecified": "", "unspecified": "", "All": "All"}).fillna("")
    r["keywords"] = r.keywords.str.replace(";", ", ")
    # a reporter occasionally states the severity in the text; hide it so the reviewer judges independently
    leak = r"\b[SP][1-4]\b"
    for c in ("summary", "description_llm"):
        r[c] = r[c].str.replace(leak, "[redacted]", regex=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, reviewer in enumerate(("A", "B")):
        order = r.sample(frac=1, random_state=SEED + 1 + k)       # a different order for each reviewer
        wb = Workbook()
        instructions(wb.active, reviewer)
        labels(wb, order, reviewer)
        wb.active = 0
        wb.save(OUT / f"Severity_Labelling_Reviewer_{reviewer}.xlsx")
    r[["item", "bug_id", "final_severity", "first_severity", "sample_split"]].to_csv(OUT / "key.csv", index=False)
    print(f"Wrote {N} items for reviewers A and B to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
