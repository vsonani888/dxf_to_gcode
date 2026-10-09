"""Load closed contours from SVG / DXF files and normalise them so that the
bottom-left corner of the drawing's bounding box is (0, 0)."""
import os
import numpy as np

CURVE_TOL = 0.05  # max distance (drawing units, usually mm) between a curve and its straight-line approximation


def _clean(pts):
    """Drop consecutive duplicates and make sure the contour is usable."""
    out = []
    for p in pts:
        if not out or (abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9):
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out if len(out) >= 3 else None


def _segments_needed(seg, tol):
    """How many straight pieces a curve needs to stay within `tol` of the true curve
    (Wang's formula for Beziers, sagitta for arcs)."""
    from svgelements import Arc, CubicBezier, QuadraticBezier

    def p(q):
        return np.array([q.x, q.y])

    if isinstance(seg, CubicBezier):
        a, b, c, d = p(seg.start), p(seg.control1), p(seg.control2), p(seg.end)
        m = max(np.hypot(*(a - 2 * b + c)), np.hypot(*(b - 2 * c + d)))
        n = np.sqrt(0.75 * m / tol)
    elif isinstance(seg, QuadraticBezier):
        a, b, c = p(seg.start), p(seg.control), p(seg.end)
        n = np.sqrt(0.25 * np.hypot(*(a - 2 * b + c)) / tol)
    elif isinstance(seg, Arc):
        r = max(abs(seg.rx), abs(seg.ry), 1e-9)
        step = 2 * np.arccos(max(-1.0, 1 - tol / r))
        n = abs(seg.sweep) / max(step, 1e-6)
    else:
        n = 16
    return int(min(1000, max(1, np.ceil(n))))


def load_svg(path):
    from svgelements import SVG, Shape, Path, Move, Line, Close, Text

    svg = SVG.parse(path)  # applies all transforms
    # Work in the document's own user units (the viewBox), which Inkscape / CorelDRAW write in
    # the document unit, normally mm. svgelements scales them to the width/height attribute.
    scale = 1.0
    if svg.viewbox is not None and svg.viewbox.width and svg.width:
        scale = svg.viewbox.width / svg.width
    contours = []
    for el in svg.elements():
        if isinstance(el, Text):
            print("WARNING: SVG contains <text>. Convert text to paths first "
                  "(Inkscape: Path > Object to Path).")
            continue
        if not isinstance(el, Shape):
            continue
        p = Path(el)
        if len(p) == 0:
            continue
        for sub in p.as_subpaths():
            pts = []
            for seg in Path(sub):
                if isinstance(seg, Move):
                    pts.append((seg.end.x, seg.end.y))
                elif isinstance(seg, (Line, Close)):
                    if seg.end is not None:
                        pts.append((seg.end.x, seg.end.y))
                else:  # curves / arcs
                    n = _segments_needed(seg, CURVE_TOL / scale)
                    pts.extend(map(tuple, seg.npoint(np.linspace(0, 1, n + 1)[1:])))
            pts = [(x * scale, -y * scale) for x, y in pts]  # SVG y points down; flip so y points up
            c = _clean(pts)
            if c:
                contours.append(c)
    return contours


def load_dxf(path):
    import ezdxf
    from ezdxf import path as ezpath

    doc = ezdxf.readfile(path)
    contours = []
    for e in doc.modelspace():
        try:
            p = ezpath.make_path(e)
        except (TypeError, ValueError):
            continue
        for sub in p.sub_paths():
            pts = [(v.x, v.y) for v in sub.flattening(distance=CURVE_TOL / 4)]
            if len(pts) < 3 or np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) > 1e-3:
                continue  # open path - not a cuttable shape
            c = _clean(pts)
            if c:
                contours.append(c)
    return contours


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        contours = load_svg(path)
    elif ext == ".dxf":
        contours = load_dxf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (use .svg or .dxf)")
    if not contours:
        raise ValueError("No closed shapes found in file.")
    return normalise(contours)


def normalise(contours):
    """Shift everything so the bounding box's bottom-left corner sits at (0, 0)."""
    allpts = np.vstack([np.array(c) for c in contours])
    minx, miny = allpts.min(axis=0)
    return [[(x - minx, y - miny) for x, y in c] for c in contours]
