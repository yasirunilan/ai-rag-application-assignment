"""Deterministic Technical Manual & Evaluation Corpus Generator (Person 1 · Story E0.1 & E0.2).

Creates:
1. `amt_handbook.pdf` (4 pages):
   - Page 1: Prose + Structured Torque Table (Table 4-1: AN3-4A, AN4-12A, MS20006-8, NAS1304-15) + tiny decorative icon (filtered by E2.1).
   - Page 2: Prose + Vector Schematic Figure 9-12 (Generator Control Relay & Bus Tie Schematic — Model 737-400).
   - Page 3: Prose + Vector Schematic Figure 9-14 (Starter Relay Control Wiring Diagram — Model 737-800).
   - Page 4: Image-only degraded scanned page (triggers E1.2 Scanned Detector + OCR Quality Gate -> review_queue.jsonl).
2. `dude_hydraulic_manual.pdf` (2 pages — DUDE eval slice):
   - Page 1: Prose + Table 2-3 (Hydraulic Fault Codes: FAULT-E409, FAULT-H204, FAULT-P118).
   - Page 2: Prose + Figure 2-1 (Hydraulic Actuator Assembly & Seal Arrangement — Model A320).
3. `gold_eval.jsonl`:
   - 12 page-labeled evaluation questions balanced across `text`, `figure`, and `table` modalities.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from mmrag.schemas import GoldQuestion


def _draw_table(
    page: fitz.Page,
    x0: float,
    y0: float,
    col_widths: list[float],
    row_height: float,
    headers: list[str],
    rows: list[list[str]],
) -> fitz.Rect:
    """Draws a ruled lattice table on a PDF page so pdfplumber/PyMuPDF extract structured cells."""
    all_rows = [headers] + rows
    total_w = sum(col_widths)
    total_h = row_height * len(all_rows)
    shape = page.new_shape()

    for r_idx, row in enumerate(all_rows):
        cy = y0 + r_idx * row_height
        cx = x0
        for c_idx, cell_text in enumerate(row):
            cw = col_widths[c_idx]
            rect = fitz.Rect(cx, cy, cx + cw, cy + row_height)
            shape.draw_rect(rect)
            page.insert_text(
                fitz.Point(cx + 5, cy + row_height - 7),
                cell_text,
                fontsize=9.5 if r_idx > 0 else 10.0,
                fontname="helv",
            )
            cx += cw

    shape.finish(color=(0.1, 0.1, 0.1), width=1.0)
    shape.commit()
    return fitz.Rect(x0, y0, x0 + total_w, y0 + total_h)


def _make_schematic_png(
    title_inside_box: str,
    left_box: str,
    center_box: str,
    right_box: str,
    terminal_code: str,
    seed_variant: int = 0,
) -> bytes:
    """Creates an embedded technical schematic PNG for realistic figure crop extraction."""
    img = Image.new("RGB", (680, 320), color=(252, 253, 255))
    draw = ImageDraw.Draw(img)

    # Outer schematic border
    draw.rectangle([8, 8, 672, 312], outline=(20, 40, 80), width=3)
    draw.text((24, 18), title_inside_box, fill=(20, 40, 80))

    # Three component blocks connected by wiring buses
    draw.rectangle([35, 85, 195, 225], outline=(30, 60, 120), width=2)
    draw.text((50, 140), left_box, fill=(10, 20, 40))

    draw.rectangle([255, 75, 425, 235], outline=(140, 30, 30), width=2)
    draw.text((275, 130), center_box, fill=(100, 10, 10))
    draw.text((275, 165), f"TERM: {terminal_code}", fill=(60, 60, 60))

    draw.rectangle([485, 85, 645, 225], outline=(20, 110, 50), width=2)
    draw.text((500, 140), right_box, fill=(10, 70, 30))

    # Wiring lines + relay coil symbols
    draw.line([(195, 130), (255, 130)], fill=(0, 0, 0), width=3)
    draw.line([(195, 180), (255, 180)], fill=(0, 0, 0), width=2)
    draw.line([(425, 155), (485, 155)], fill=(0, 0, 0), width=3)

    if seed_variant == 1:
        draw.line([(115, 225), (115, 275), (565, 275), (565, 225)], fill=(40, 40, 140), width=2)
        draw.text((285, 282), "28V DC CONTROL FEEDBACK LOOP", fill=(40, 40, 140))
    else:
        draw.line([(115, 225), (115, 275), (340, 275), (340, 235)], fill=(120, 60, 20), width=2)
        draw.text((145, 282), "SENSING LINE BUS", fill=(120, 60, 20))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_degraded_scan_png() -> bytes:
    """Creates a full-page noisy, unreadable microfiche scan image (Story E1.2 test page)."""
    img = Image.new("RGB", (800, 1000), color=(215, 212, 205))
    draw = ImageDraw.Draw(img)
    # Draw corrupted pseudo-glyphs and heavy scan streaks that fail OCR quality gate
    for y in range(40, 960, 22):
        draw.line([(30, y), (770, y + (y % 7) - 3)], fill=(95, 95, 95), width=2)
        draw.text((45, y - 10), "~|1lI| %%% ^&* @#$ ~|1lI| ??? === /// |||", fill=(70, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_amt_handbook_pdf(output_path: str | Path) -> Path:
    """Generates `amt_handbook.pdf` (4 pages: torque table, 2 relay wiring diagrams, 1 bad scan)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # -------------------------------------------------------------------------
    # Page 1: Prose + Structured Torque Table (Table 4-1) + Tiny Icon Rule
    # -------------------------------------------------------------------------
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(
        fitz.Point(50, 50),
        "Chapter 4: Airframe Structural Fasteners & Torque Limits (Model 737-800)",
        fontsize=14,
        fontname="helv",
    )
    p1.insert_text(
        fitz.Point(50, 80),
        "4.1 Standard Maintenance Practices and Hydraulic System Ratings",
        fontsize=11.5,
        fontname="helv",
    )
    prose_p1 = (
        "All primary structural fasteners on the Model 737-800 airframe must be tightened "
        "using a calibrated torque wrench inspected within the last 90 days. The primary "
        "hydraulic flight control system operates at a nominal pressure of 3,000 PSI with "
        "a thermal relief valve cracking pressure of 3,750 PSI. Never reuse self-locking "
        "nuts once removal torque drops below the minimum prevailing torque threshold."
    )
    p1.insert_textbox(fitz.Rect(50, 95, 560, 175), prose_p1, fontsize=10, fontname="helv")

    # Tiny decorative rule/icon (28x10 pt) — Story E2.1 must filter this out
    tiny_img = Image.new("RGB", (28, 10), color=(40, 80, 160))
    tiny_buf = io.BytesIO()
    tiny_img.save(tiny_buf, format="PNG")
    p1.insert_image(fitz.Rect(50, 180, 78, 190), stream=tiny_buf.getvalue())

    p1.insert_text(
        fitz.Point(50, 215),
        "Table 4-1. Standard Airframe Bolt Torque Specifications (Model 737-800)",
        fontsize=11,
        fontname="helv",
    )
    headers_t1 = ["Part Number", "Thread Size", "Application", "Dry Torque (in-lbs)", "Lubed Torque (in-lbs)"]
    rows_t1 = [
        ["AN3-4A", "10-32", "Aileron Trim Bracket", "20-25", "12-15"],
        ["AN4-12A", "1/4-28", "Elevator Bellcrank Mount", "50-70", "35-45"],
        ["MS20006-8", "3/8-24", "Main Wing Spar Fitting", "160-190", "110-130"],
        ["NAS1304-15", "1/4-28", "Spoiler Actuator Clevis", "65-85", "45-55"],
    ]
    _draw_table(
        p1,
        x0=50,
        y0=230,
        col_widths=[90, 75, 140, 100, 105],
        row_height=26,
        headers=headers_t1,
        rows=rows_t1,
    )

    # -------------------------------------------------------------------------
    # Page 2: Vector/Raster Schematic — Figure 9-12 (Generator Control Relay, Model 737-400)
    # -------------------------------------------------------------------------
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(
        fitz.Point(50, 50),
        "Chapter 9: Electrical Power & Relay Distribution (Model 737-400)",
        fontsize=14,
        fontname="helv",
    )
    p2.insert_text(
        fitz.Point(50, 80),
        "9.2 AC Generator Control Relay & Bus Tie Circuitry",
        fontsize=11.5,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(50, 95, 560, 155),
        "The generator control relay isolates the IDG exciter field whenever an overvoltage "
        "or differential current fault is latched by the Generator Control Unit (GCU). "
        "Refer to the schematic below for relay coil pinouts.",
        fontsize=10,
        fontname="helv",
    )
    fig_9_12_bytes = _make_schematic_png(
        title_inside_box="RELAY CONTROL WIRING SCHEMATIC",
        left_box="28V DC BUS\nCB-14 (5A)",
        center_box="RELAY COIL K1\n4PDT CONTACTS",
        right_box="LOAD CONTACTS\nOUT TERMINAL",
        terminal_code="PIN-A4 / K1",
        seed_variant=0,
    )
    p2.insert_image(fitz.Rect(60, 170, 550, 410), stream=fig_9_12_bytes)
    p2.insert_text(
        fitz.Point(60, 430),
        "Figure 9-12. Generator Control Relay & Bus Tie Schematic (Model 737-400)",
        fontsize=10.5,
        fontname="helv",
    )

    # -------------------------------------------------------------------------
    # Page 3: Vector/Raster Schematic — Figure 9-14 (Starter Relay Control Wiring, Model 737-800)
    # Note: Inside the image box it also says "RELAY CONTROL WIRING SCHEMATIC",
    # so without nearby caption/heading context (Failure Mode 1), vision captions drift!
    # -------------------------------------------------------------------------
    p3 = doc.new_page(width=612, height=792)
    p3.insert_text(
        fitz.Point(50, 50),
        "Chapter 9: Engine Starting & Ignition Control Systems (Model 737-800)",
        fontsize=14,
        fontname="helv",
    )
    p3.insert_text(
        fitz.Point(50, 80),
        "9.4 Engine Starter Relay Control Wiring and Valve Interlock",
        fontsize=11.5,
        fontname="helv",
    )
    p3.insert_textbox(
        fitz.Rect(50, 95, 560, 155),
        "When the start switch is placed in the GND position on the Model 737-800 overhead "
        "panel, current flows through the N2 RPM cutout switch to energize the starter "
        "relay coil and open the pneumatic starter shutoff valve.",
        fontsize=10,
        fontname="helv",
    )
    fig_9_14_bytes = _make_schematic_png(
        title_inside_box="RELAY CONTROL WIRING SCHEMATIC",
        left_box="28V DC BUS\nSTART SW",
        center_box="RELAY COIL K2\n4PDT CONTACTS",
        right_box="VALVE SOLENOID\nOUT TERMINAL",
        terminal_code="PIN-B7 / K2",
        seed_variant=1,
    )
    p3.insert_image(fitz.Rect(60, 170, 550, 410), stream=fig_9_14_bytes)
    p3.insert_text(
        fitz.Point(60, 430),
        "Figure 9-14. Engine Starter Relay Control Wiring Diagram (Model 737-800)",
        fontsize=10.5,
        fontname="helv",
    )

    # -------------------------------------------------------------------------
    # Page 4: Degraded Scanned Page (no text layer + full-page noisy scan)
    # -------------------------------------------------------------------------
    p4 = doc.new_page(width=612, height=792)
    scan_bytes = _make_degraded_scan_png()
    p4.insert_image(fitz.Rect(15, 15, 597, 777), stream=scan_bytes)

    pdf_bytes = doc.tobytes()
    doc.close()
    try:
        out.write_bytes(pdf_bytes)
    except PermissionError:
        if not out.exists():
            raise
    return out


def build_dude_hydraulic_pdf(output_path: str | Path) -> Path:
    """Generates `dude_hydraulic_manual.pdf` (2 pages: fault code table + actuator diagram)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # Page 1: Fault Code Table (Table 2-3) + Prose
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(
        fitz.Point(50, 50),
        "Section 2: Hydraulic Landing Gear Troubleshooting & Fault Codes (Model A320)",
        fontsize=13.5,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(50, 75, 560, 145),
        "Before servicing the yellow or green hydraulic manifold on the Model A320, "
        "depressurize the accumulator to 0 PSI and verify that nitrogen pre-charge "
        "reads 1,450 PSI at 20 degrees Celsius ambient temperature.",
        fontsize=10,
        fontname="helv",
    )
    p1.insert_text(
        fitz.Point(50, 165),
        "Table 2-3. Hydraulic Electronic Monitoring Fault Codes & Corrective Actions",
        fontsize=11,
        fontname="helv",
    )
    headers_t2 = ["Fault Code", "Subsystem", "Trigger Condition", "Mandatory Action"]
    rows_t2 = [
        ["FAULT-E409", "Green Reservoir", "Fluid level < 2.1 Liters", "Inspect return line & refill Skydrol LD-4"],
        ["FAULT-H204", "PTU Manifold", "Delta pressure > 650 PSI", "Replace PTU transfer control valve V-12"],
        ["FAULT-P118", "EDP Pump #1", "Case drain temp > 125 C", "Isolate engine pump & replace stator seal"],
    ]
    _draw_table(
        p1,
        x0=50,
        y0=180,
        col_widths=[90, 105, 140, 175],
        row_height=26,
        headers=headers_t2,
        rows=rows_t2,
    )

    # Page 2: Hydraulic Actuator Exploded View Schematic (Figure 2-1)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(
        fitz.Point(50, 50),
        "Section 2.5: Main Landing Gear Retraction Actuator Assembly (Model A320)",
        fontsize=13.5,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(50, 75, 560, 135),
        "The main landing gear retraction actuator incorporates a dual-tandem bronze "
        "gland seal and a hydraulic cushion damping sleeve at the up-lock stroke end.",
        fontsize=10,
        fontname="helv",
    )
    fig_2_1_bytes = _make_schematic_png(
        title_inside_box="HYDRAULIC ACTUATOR SEAL & PORT DIAGRAM",
        left_box="PORT A (EXTEND)\n3000 PSI INLET",
        center_box="PISTON GLAND\nSEAL KIT SK-90",
        right_box="PORT B (RETRACT)\nDAMPING ORIFICE",
        terminal_code="SK-90 / ORIFICE-3",
        seed_variant=1,
    )
    p2.insert_image(fitz.Rect(60, 155, 550, 395), stream=fig_2_1_bytes)
    p2.insert_text(
        fitz.Point(60, 415),
        "Figure 2-1. Main Landing Gear Retraction Actuator Seal & Port Diagram (Model A320)",
        fontsize=10.5,
        fontname="helv",
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    try:
        out.write_bytes(pdf_bytes)
    except PermissionError:
        if not out.exists():
            raise
    return out


def build_default_gold_questions() -> list[GoldQuestion]:
    """Returns 12 gold evaluation queries covering text, figure, and table modalities."""
    return [
        # --- FIGURE QUERIES (4) ---
        GoldQuestion(
            query_id="Q_FIG_01",
            query="Show the engine starter relay control wiring diagram for Model 737-800",
            gold_doc_id="amt_handbook",
            gold_pages=[3],
            expected_type="figure",
            gold_artefact_hint="Figure 9-14",
            expected_keywords=["Figure 9-14", "Starter Relay", "737-800"],
        ),
        GoldQuestion(
            query_id="Q_FIG_02",
            query="Show the AC generator control relay and bus tie schematic for Model 737-400",
            gold_doc_id="amt_handbook",
            gold_pages=[2],
            expected_type="figure",
            gold_artefact_hint="Figure 9-12",
            expected_keywords=["Figure 9-12", "Generator Control Relay", "737-400"],
        ),
        GoldQuestion(
            query_id="Q_FIG_03",
            query="Show the main landing gear retraction actuator seal and port diagram for Model A320",
            gold_doc_id="dude_hydraulic_manual",
            gold_pages=[2],
            expected_type="figure",
            gold_artefact_hint="Figure 2-1",
            expected_keywords=["Figure 2-1", "Actuator", "A320"],
        ),
        GoldQuestion(
            query_id="Q_FIG_04",
            query="Diagram showing valve solenoid PIN-B7 and starter relay coil K2 wiring",
            gold_doc_id="amt_handbook",
            gold_pages=[3],
            expected_type="figure",
            gold_artefact_hint="Figure 9-14",
            expected_keywords=["Figure 9-14", "Starter"],
        ),
        # --- TABLE QUERIES (4) ---
        GoldQuestion(
            query_id="Q_TAB_01",
            query="What is the dry and lubed torque spec table for bolt part number AN4-12A?",
            gold_doc_id="amt_handbook",
            gold_pages=[1],
            expected_type="table",
            gold_artefact_hint="Table 4-1",
            expected_keywords=["AN4-12A", "50-70", "35-45"],
        ),
        GoldQuestion(
            query_id="Q_TAB_02",
            query="Lookup table entry and specifications for part number NAS1304-15",
            gold_doc_id="amt_handbook",
            gold_pages=[1],
            expected_type="table",
            gold_artefact_hint="Table 4-1",
            expected_keywords=["NAS1304-15", "65-85", "45-55"],
        ),
        GoldQuestion(
            query_id="Q_TAB_03",
            query="Lookup table entry and mandatory action for code FAULT-E409",
            gold_doc_id="dude_hydraulic_manual",
            gold_pages=[1],
            expected_type="table",
            gold_artefact_hint="Table 2-3",
            expected_keywords=["FAULT-E409", "Green Reservoir", "Skydrol"],
        ),
        GoldQuestion(
            query_id="Q_TAB_04",
            query="Lookup fault code FAULT-H204 PTU manifold delta pressure action in troubleshooting table",
            gold_doc_id="dude_hydraulic_manual",
            gold_pages=[1],
            expected_type="table",
            gold_artefact_hint="Table 2-3",
            expected_keywords=["FAULT-H204", "PTU Manifold", "V-12"],
        ),
        # --- TEXT QUERIES (4) ---
        GoldQuestion(
            query_id="Q_TXT_01",
            query="What is the nominal operating pressure and thermal relief valve cracking pressure of the 737-800 hydraulic flight control system?",
            gold_doc_id="amt_handbook",
            gold_pages=[1],
            expected_type="text",
            gold_artefact_hint="3,000 PSI",
            expected_keywords=["3,000 PSI", "3,750 PSI"],
        ),
        GoldQuestion(
            query_id="Q_TXT_02",
            query="What nitrogen pre-charge pressure is required on the Model A320 hydraulic accumulator at 20 degrees Celsius?",
            gold_doc_id="dude_hydraulic_manual",
            gold_pages=[1],
            expected_type="text",
            gold_artefact_hint="1,450 PSI",
            expected_keywords=["1,450 PSI"],
        ),
        GoldQuestion(
            query_id="Q_TXT_03",
            query="How does current flow when the start switch is placed in the GND position on the 737-800 overhead panel?",
            gold_doc_id="amt_handbook",
            gold_pages=[3],
            expected_type="text",
            gold_artefact_hint="N2 RPM cutout switch",
            expected_keywords=["N2 RPM cutout switch", "shutoff valve"],
        ),
        GoldQuestion(
            query_id="Q_TXT_04",
            query="When does the generator control relay isolate the IDG exciter field?",
            gold_doc_id="amt_handbook",
            gold_pages=[2],
            expected_type="text",
            gold_artefact_hint="overvoltage or differential current",
            expected_keywords=["overvoltage", "Generator Control Unit"],
        ),
    ]


def ensure_default_corpus(manuals_dir: str | Path, gold_eval_path: str | Path) -> dict[str, Any]:
    """Ensures the corpus directory contains manuals + page-labeled gold evaluation set (Story E0.2)."""
    m_dir = Path(manuals_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    amt_pdf = m_dir / "amt_handbook.pdf"
    dude_pdf = m_dir / "dude_hydraulic_manual.pdf"

    build_amt_handbook_pdf(amt_pdf)
    build_dude_hydraulic_pdf(dude_pdf)

    gold_path = Path(gold_eval_path)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    questions = build_default_gold_questions()
    with gold_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump(), ensure_ascii=False) + "\n")

    return {
        "manuals": [str(amt_pdf), str(dude_pdf)],
        "gold_eval_path": str(gold_path),
        "num_gold_questions": len(questions),
    }
