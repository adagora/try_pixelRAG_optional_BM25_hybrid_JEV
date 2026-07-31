#!/usr/bin/env python
"""Generate a stand-in manufacturing manual PDF.

Mimics the visual character of gate/door technical documentation: dimension
callouts in small type, a parts table with SKUs, and a balloon-numbered
exploded view. Used to validate the pipeline and measure how much small-text
detail survives rendering + embedding, before committing real documents.

Throwaway: delete once real PDFs are in ./pdfs.
"""

import fitz  # PyMuPDF, already a pixelrag core dependency

A4 = fitz.paper_rect("a4")  # 595 x 842 pt


def _label(page, x, y, text, size=6):
    """Dimension callout — the smallest text that must survive the pipeline."""
    page.insert_text((x, y), text, fontsize=size, fontname="helv")


def page_elevation(doc):
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 60), "SG-2400 Sliding Gate — Elevation", fontsize=14, fontname="hebo")
    page.insert_text((50, 78), "Drawing No. DWG-SG2400-01  Rev. C", fontsize=8)

    # Gate frame
    frame = fitz.Rect(90, 140, 470, 380)
    page.draw_rect(frame, color=(0, 0, 0), width=1.2)
    # Infill bars
    for i in range(1, 12):
        x = 90 + i * (380 / 12)
        page.draw_line(fitz.Point(x, 140), fitz.Point(x, 380), color=(0.3, 0.3, 0.3), width=0.6)

    # Horizontal dimension line with small callout
    page.draw_line(fitz.Point(90, 405), fitz.Point(470, 405), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(90, 400), fitz.Point(90, 410), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(470, 400), fitz.Point(470, 410), color=(0, 0, 0), width=0.5)
    _label(page, 260, 402, "2400 mm")

    # Vertical dimension
    page.draw_line(fitz.Point(490, 140), fitz.Point(490, 380), color=(0, 0, 0), width=0.5)
    _label(page, 495, 262, "1800 mm")

    # Fine annotations — the hardest text to preserve
    _label(page, 100, 135, "Ø48 post", size=5.5)
    _label(page, 380, 135, "top rail 40x40x2", size=5.5)
    _label(page, 100, 392, "ground clearance 45 mm", size=5.5)
    _label(page, 300, 392, "infill pitch 110 mm o/c", size=5.5)

    page.insert_text((50, 450), "Note: verify site opening before fabrication. Tolerance ±3 mm.", fontsize=8)
    return page


def page_parts_table(doc):
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 60), "SG-2400 — Bill of Materials", fontsize=14, fontname="hebo")

    rows = [
        ("Item", "Part No.", "Description", "Qty", "Material"),
        ("1", "SG-FRM-2400", "Main frame weldment 2400x1800", "1", "S235JR"),
        ("2", "SG-RLR-080", "Bottom roller assembly 80 mm", "2", "Nylon/Steel"),
        ("3", "SG-GDE-U40", "Upper guide bracket U-40", "2", "Galv. steel"),
        ("4", "SG-INF-110", "Infill bar 40x20x1.5, pitch 110", "12", "S235JR"),
        ("5", "SG-END-CAP", "End cap 40x40 black", "4", "PA6"),
        ("6", "SG-LCK-M12", "Locking pin M12x90", "1", "A2-70"),
        ("7", "SG-MTR-370", "Drive motor 370 W 230 V", "1", "—"),
        ("8", "SG-RCK-M4", "Drive rack module 4, 1 m", "3", "Nylon 6"),
    ]

    y = 100
    col_x = [55, 100, 175, 380, 420]
    for r_i, row in enumerate(rows):
        bold = r_i == 0
        size = 8 if bold else 7.5
        font = "hebo" if bold else "helv"
        for c_i, cell in enumerate(row):
            page.insert_text((col_x[c_i], y), cell, fontsize=size, fontname=font)
        page.draw_line(fitz.Point(50, y + 4), fitz.Point(545, y + 4),
                       color=(0.7, 0.7, 0.7), width=0.4)
        y += 22

    page.insert_text((50, y + 20),
                     "Torque spec: roller bolts 45 Nm. Motor mount bolts 25 Nm.", fontsize=7.5)
    return page


def page_exploded(doc):
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 60), "SG-2400 — Roller Assembly, Exploded View", fontsize=14, fontname="hebo")

    # Stacked components with balloon numbers
    parts = [
        (150, "Housing SG-RLR-H"),
        (210, "Bearing 6204-2RS"),
        (270, "Nylon wheel Ø80"),
        (330, "Axle M12x110"),
        (390, "Circlip DIN 471"),
    ]
    for i, (y, name) in enumerate(parts, start=1):
        page.draw_rect(fitz.Rect(180, y, 340, y + 34), color=(0, 0, 0), width=0.8)
        page.insert_text((190, y + 22), name, fontsize=7)
        # Balloon
        page.draw_circle(fitz.Point(150, y + 17), 11, color=(0, 0, 0), width=0.8)
        page.insert_text((146, y + 21), str(i), fontsize=8, fontname="hebo")
        page.draw_line(fitz.Point(161, y + 17), fitz.Point(180, y + 17),
                       color=(0, 0, 0), width=0.5)

    _label(page, 360, 170, "press fit, do not lubricate", size=5.5)
    _label(page, 360, 290, "replace every 5000 cycles", size=5.5)
    return page


def main():
    doc = fitz.open()
    page_elevation(doc)
    page_parts_table(doc)
    page_exploded(doc)
    out = "pdfs/_TEST_SG2400_manual.pdf"
    doc.save(out)
    doc.close()
    print(f"wrote {out} (3 pages)")


if __name__ == "__main__":
    main()
