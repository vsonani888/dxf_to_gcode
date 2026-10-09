"""Load closed contours from SVG / DXF files and normalise them so that the
bottom-left corner of the drawing's bounding box is (0, 0)."""
import os
import numpy as np

CURVE_TOL = 0.25  # max chord length (drawing units, usually mm) when flattening curves


def _clean(pts):
    """Drop consecutive duplicates and make sure the contour is usable."""
    out = []
    for p in pts:
        if not out or (abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9):
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out if len(out) >= 3 else None


def load_svg(path):
    from svgelements import SVG, Shape, Path, Move, Line, Close, Text

    svg = SVG.parse(path, ppi=25.4)  # applies all transforms; ppi=25.4 -> 1 unit = 1 mm
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
                    n = max(4, int(np.ceil(seg.length() / CURVE_TOL)))
                    for t in np.linspace(0, 1, n + 1)[1:]:
                        q = seg.point(t)
                        pts.append((q.x, q.y))
            pts = [(x, -y) for x, y in pts]  # SVG y points down; flip so y points up
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
