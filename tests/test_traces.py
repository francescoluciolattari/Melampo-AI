"""Traces read as numbers: vector curves of a digital PDF, the vectorised picture of a scan, the same analysis
(16 October 2026). Synthetic electrophoresis curves: sums of Gaussians with known areas."""

import numpy as np
import pytest

from melampo.data import traces as tr

ZONES = ((0.15, 0.035, 1.0), (0.33, 0.025, 0.12), (0.48, 0.03, 0.22), (0.63, 0.035, 0.2), (0.85, 0.07, 0.18))
TRUE = {"albumin": 54.5, "alpha1": 4.7, "alpha2": 10.3, "beta": 10.9, "gamma": 19.6}


def spe(spike=False, n=300):
    x = np.linspace(0, 1, n)
    y = sum(h * np.exp(-0.5 * ((x - c) / w) ** 2) for c, w, h in ZONES)
    if spike:
        y = y + 0.35 * np.exp(-0.5 * ((x - 0.82) / 0.008) ** 2)
    return x, y


def pdf_with_curve(x, y, lines=()):
    pts = [(72 + x[i] * 400, 300 + y[i] * 200) for i in range(len(x))]
    ops = "1 w 0 0 0 RG\n" + f"{pts[0][0]:.2f} {pts[0][1]:.2f} m\n" + "".join(f"{a:.2f} {b:.2f} l\n" for a, b in pts[1:])
    ops += "S\n72 290 m 472 290 l S\n72 290 m 72 520 l S\n"
    for i, line in enumerate(lines):
        ops += f"BT /F1 10 Tf 72 {740 - 14 * i} Td ({line}) Tj ET\n"
    content = ops.encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


def picture(x, y, h=200, w=500):
    img = np.full((h, w), 255.0)
    xs = (x * (w - 20) + 10).astype(int)
    ys = (h - 10 - y / y.max() * 150).astype(int)
    for i in range(len(xs) - 1):
        for t in np.linspace(0, 1, 40):
            img[int(ys[i] + (ys[i + 1] - ys[i]) * t), int(xs[i] + (xs[i + 1] - xs[i]) * t)] = 0
    img[h - 10, :] = 0
    img[:, 8] = 0
    img[::25, ::2] = 120          # dotted grid
    img[5:12, 300:340] = 0        # a label
    return img


needs_poppler = pytest.mark.skipif(not __import__("shutil").which("pdftocairo"), reason="poppler not installed")


@needs_poppler
def test_the_vector_curve_of_a_digital_pdf_gives_the_zones():
    curves = tr.vector_curves(pdf_with_curve(*spe()))
    assert len(curves) == 1           # the axes are straight lines, not traces
    r = tr.analyse_electrophoresis(curves[0], source="vector")
    assert all(abs(r.zones[k] - v) <= 1.5 for k, v in TRUE.items()) and r.narrow_peaks == []


def test_the_vectorised_picture_gives_the_same_zones():
    r = tr.analyse_electrophoresis(tr.raster_curve(picture(*spe())), source="raster")
    assert all(abs(r.zones[k] - v) <= 2.0 for k, v in TRUE.items())


@pytest.mark.parametrize("make", ["vector", "raster"])
def test_a_narrow_peak_in_the_gamma_zone_is_described(make):
    x, y = spe(spike=True)
    if make == "vector":
        if not __import__("shutil").which("pdftocairo"):
            pytest.skip("poppler not installed")
        curve = tr.vector_curves(pdf_with_curve(x, y))[0]
    else:
        curve = tr.raster_curve(picture(x, y))
    r = tr.analyse_electrophoresis(curve)
    assert [p.zone for p in r.narrow_peaks] == ["gamma"]


def test_the_curve_is_checked_against_the_table():
    r = tr.analyse_electrophoresis(tr.raster_curve(picture(*spe())), table=TRUE)
    assert r.table_check["consistent"]
    wrong = dict(TRUE, albumin=40.0)
    assert not tr.analyse_electrophoresis(tr.raster_curve(picture(*spe())), table=wrong).table_check["consistent"]


def test_a_reversed_axis_is_turned():
    x, y = spe()
    r = tr.analyse_electrophoresis(np.column_stack([x, y[::-1]]))
    assert "axis_reversed_albumin_right" in r.notes and abs(r.zones["albumin"] - TRUE["albumin"]) < 1.5


@needs_poppler
def test_an_electrophoresis_pdf_attachment_carries_its_trace():
    from melampo.data.case_attachments import CaseAttachment, process_case_attachments
    from melampo.data.document_processing import ClinicalDocumentProcessor

    lines = ["ELETTROFORESI SIEROPROTEICA", "Albumina      54,5   %   55,8 - 66,1", "Alfa 1         4,7   %   2,9 - 4,9",
             "Alfa 2        10,3   %   7,1 - 11,8", "Beta          10,9   %   8,4 - 13,1",
             "Gamma         19,6   %   11,1 - 18,8"]
    item = CaseAttachment.from_payload_item({"data": pdf_with_curve(*spe(), lines=lines), "filename": "spe.pdf"})
    attachment = process_case_attachments([item], ClinicalDocumentProcessor()).attachments[0]
    s = attachment.summary()
    assert "electrophoresis_trace" in s["routing"]["interpreters"]
    reading = attachment.traces["items"][0]["reading"]
    assert reading["table_check"]["consistent"] and s["traces"] == {
        "count": 1, "sources": ["vector"], "interpreted": 1, "narrow_peaks": 0, "note": None}
