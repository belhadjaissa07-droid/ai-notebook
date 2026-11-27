"""STEP 0 - a sample PDF that a text-only RAG cannot fully answer.

The quarterly numbers live ONLY in a raster chart (Figure 1), uptime ONLY in a vector chart
(Figure 2), capacity in a table (Table 1). The text just says 'see Figure 1'.

Run: python src/rag/multimodal/0_make_sample_pdf.py   ->  data/greengrid_report_2025.pdf
"""
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parents[3] / "data" / "greengrid_report_2025.pdf"
W, H, M = 595, 842, 56
SOLAR, WIND, GREEN = (0.94, 0.67, 0.12), (0.16, 0.47, 0.78), (0.2, 0.7, 0.45)


def para(page, y, text, h=90):
    page.insert_textbox(pymupdf.Rect(M, y, W - M, y + h), text, fontsize=11)
    return y + h + 10


def bars(page, x, y, w, h, title, cats, series, lo, hi):
    """Grouped bar chart drawn with shapes: a VECTOR chart (no image inside the PDF)."""
    page.insert_text((x, y - 12), title, fontsize=11, fontname="hebo")
    page.draw_line((x, y + h), (x + w, y + h))
    gw = w / len(cats)
    bw = gw / (len(series) + 1)
    for i, cat in enumerate(cats):
        page.insert_text((x + i * gw + 8, y + h + 14), cat, fontsize=9)
        for j, (vals, color) in enumerate(series):
            bh = h * (vals[i] - lo) / (hi - lo)
            bx = x + i * gw + 8 + j * bw
            page.draw_rect(pymupdf.Rect(bx, y + h - bh, bx + bw - 4, y + h), color=color, fill=color)
            page.insert_text((bx, y + h - bh - 4), str(vals[i]), fontsize=8)


def build() -> pymupdf.Document:
    doc = pymupdf.open()

    p = doc.new_page(width=W, height=H)
    p.insert_text((M, 80), "GreenGrid Annual Report 2025", fontsize=22, fontname="hebo")
    y = para(p, 110, "GreenGrid operates solar farms and onshore wind parks across North Africa and Southern "
                     "Europe. In 2025 we focused on grid reliability, storage and operating cost reduction.")
    y = para(p, y, "Production followed seasonal patterns: solar rises in the warm months while wind is "
                   "stronger in winter. Detailed quarterly figures are shown in Figure 1 on the next page. "
                   "We do not restate the individual numbers in the text.")
    para(p, y, "Outlook: we plan to add battery storage at three sites in 2026 and extend predictive "
               "maintenance to every turbine. There were no lost-time incidents during the year.")

    # Figure 1: draw the chart on a scratch page, rasterise it, embed the PNG (numbers become pixels)
    scratch = pymupdf.open().new_page(width=480, height=300)
    bars(scratch, 40, 50, 400, 200, "Quarterly renewable output 2025 (GWh)", ["Q1", "Q2", "Q3", "Q4"],
         [([120, 310, 380, 140], SOLAR), ([210, 150, 90, 260], WIND)], 0, 400)
    p = doc.new_page(width=W, height=H)
    p.insert_text((M, 80), "1. Energy output", fontsize=16, fontname="hebo")
    p.insert_image(pymupdf.Rect(M, 120, W - M, 120 + (W - 2 * M) * 300 / 480), stream=scratch.get_pixmap(dpi=150).tobytes("png"))
    p.insert_text((M, 120 + (W - 2 * M) * 300 / 480 + 22), "Figure 1: Quarterly renewable output by technology, 2025 (GWh).", fontsize=10)

    p = doc.new_page(width=W, height=H)
    p.insert_text((M, 80), "2. Capacity and reliability", fontsize=16, fontname="hebo")
    p.insert_text((M, 108), "Table 1: Installed capacity by region (MW).", fontsize=10)
    rows = [["Region", "Solar (MW)", "Wind (MW)", "Storage (MWh)"], ["Tunisia", "420", "180", "60"],
            ["Morocco", "510", "340", "120"], ["Spain", "380", "260", "90"], ["Italy", "150", "310", "40"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            box = pymupdf.Rect(M + c * 120, 120 + r * 24, M + (c + 1) * 120, 144 + r * 24)
            p.draw_rect(box, color=(0, 0, 0), width=0.8)
            p.insert_textbox(box + (6, 6, -2, 0), cell, fontsize=10, fontname="hebo" if r == 0 else "helv")
    bars(p, M, 300, 380, 150, "Grid uptime by region, 2025 (%)", ["Tunisia", "Morocco", "Spain", "Italy"],
         [([97.1, 98.4, 99.2, 95.6], GREEN)], 90, 100)
    p.insert_text((M, 490), "Figure 2: Grid uptime by region, 2025 (%).", fontsize=10)
    return doc


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build().save(OUT)
    print(f"Sample PDF written to {OUT}")
