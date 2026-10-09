"""Toolpath planning for a 2D hot-wire foam cutter.

The wire can never lift, so every move cuts. Rules:
  * travel  - moves between shapes go through free foam only, never through a part
  * cut     - each shape's outline is cut as one closed loop
  * bridge  - the only time the wire enters a part: to reach a hole inside it
              (it goes in and comes back out along the same line)

Nesting is handled with a contour tree:
  depth 0 = part outline, depth 1 = hole in that part,
  depth 2 = a separate part sitting inside that hole, and so on.
"""
import heapq
import math

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, substring, unary_union
from shapely.prepared import prep


class Contour:
    def __init__(self, coords):
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        self.poly = Polygon(poly.exterior.coords)  # filled outline, no holes
        self.children = []
        self.depth = 0

    @property
    def is_hole(self):
        return self.depth % 2 == 1


def build_tree(contours, eps):
    nodes = sorted((Contour(c) for c in contours), key=lambda n: -n.poly.area)
    roots = []
    for i, node in enumerate(nodes):
        parent = None
        for big in reversed(nodes[:i]):  # ascending area -> first hit is the tightest container
            if big.poly.area > node.poly.area and big.poly.buffer(eps).contains(node.poly):
                parent = big
                break
        if parent is None:
            roots.append(node)
        else:
            node.depth = parent.depth + 1
            parent.children.append(node)
    return roots, nodes


def ring_part(ring, a, b):
    """Coordinates along a closed ring from distance a to distance b (b may wrap past the end)."""
    L = ring.length
    if b - a < 1e-12:
        return []
    while a >= L:
        a -= L
        b -= L
    if b <= L + 1e-12:
        return list(substring(ring, a, min(b, L)).coords)
    return list(substring(ring, a, L).coords) + list(substring(ring, 0, b - L).coords)[1:]


class Router:
    """Shortest travel path inside `container` that avoids `obstacles` (visibility graph + A*)."""

    def __init__(self, container, obstacles, clearance, eps):
        obst = unary_union(obstacles)
        self.block = prep(obst.buffer(-eps))      # shrunk a hair so sliding along an outline is allowed
        self.inside = prep(container.buffer(eps))
        free = container.buffer(-clearance).difference(
            obst.buffer(clearance, join_style="mitre")).simplify(clearance * 0.3)
        self.nodes = []
        for g in getattr(free, "geoms", [free]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            for ring in [g.exterior, *g.interiors]:
                self.nodes.extend(tuple(p) for p in list(ring.coords)[:-1])
        self._vis = {}
        self.eps = eps

    def visible(self, p, q):
        seg = LineString([p, q])
        return self.inside.contains(seg) and not self.block.intersects(seg)

    def _vis_cached(self, i, j, pts):
        key = (min(i, j), max(i, j))
        if key not in self._vis:
            self._vis[key] = self.visible(pts[i], pts[j])
        return self._vis[key]

    def route(self, a, b):
        a, b = tuple(a), tuple(b)
        if math.dist(a, b) < self.eps or self.visible(a, b):
            return [a, b]
        pts = self.nodes + [a, b]
        n = len(self.nodes)
        start, goal = n, n + 1
        g = {start: 0.0}
        prev = {}
        heap = [(math.dist(a, b), start)]
        closed = set()
        while heap:
            _, u = heapq.heappop(heap)
            if u == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return [pts[k] for k in reversed(path)]
            if u in closed:
                continue
            closed.add(u)
            for v in range(len(pts)):
                if v == u or v in closed or v == start:
                    continue
                # edges touching a/b change per query, don't cache those
                ok = (self._vis_cached(u, v, pts) if u < n and v < n
                      else self.visible(pts[u], pts[v]))
                if not ok:
                    continue
                nd = g[u] + math.dist(pts[u], pts[v])
                if nd < g.get(v, math.inf):
                    g[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd + math.dist(pts[v], b), v))
        print(f"WARNING: no clean travel path found {a} -> {b}, using straight line")
        return [a, b]


def order_parts(parts, start):
    """Visiting order: nearest-neighbour tour from `start` and back, improved with 2-opt."""
    if len(parts) <= 1:
        return list(parts)
    geoms = [Point(start)] + [p.poly for p in parts]
    n = len(geoms)
    D = [[geoms[i].distance(geoms[j]) for j in range(n)] for i in range(n)]

    tour, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[tour[-1]][j])
        tour.append(nxt)
        left.remove(nxt)
    tour.append(0)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                a, b, c, d = tour[i - 1], tour[i], tour[j], tour[j + 1]
                if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return [parts[k - 1] for k in tour[1:-1]]


class Planner:
    def __init__(self, contours, origin=(0.0, 0.0), clearance=None):
        allpts = [p for c in contours for p in c] + [origin]
        xs, ys = zip(*allpts)
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        diag = math.hypot(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1])
        self.eps = diag * 1e-6
        self.clearance = clearance if clearance is not None else diag * 0.005
        self.origin = tuple(origin)
        self.roots, self.contours = build_tree(contours, self.eps * 10)
        self.moves = []  # list of (kind, [(x, y), ...]) with kind in travel / cut / bridge

    def plan(self):
        minx, miny, maxx, maxy = self.bounds
        m = max(maxx - minx, maxy - miny)
        world = box(minx - m, miny - m, maxx + m, maxy + m)
        self.moves = []
        self._visit_region(world, self.roots, self.origin)
        return self.moves

    def _emit(self, kind, coords):
        coords = [tuple(c) for c in coords]
        if len(coords) < 2:
            return
        if self.moves and self.moves[-1][0] == kind and math.dist(self.moves[-1][1][-1], coords[0]) < 1e-9:
            self.moves[-1][1].extend(coords[1:])
        else:
            self.moves.append((kind, coords))

    def _visit_region(self, container, parts, start):
        """Cut every part lying in `container` (free foam), starting and ending at `start`."""
        if not parts:
            return
        router = Router(container, [p.poly for p in parts], self.clearance, self.eps)
        pos = start
        for part in order_parts(parts, start):
            entry = nearest_points(part.poly.exterior, Point(pos))[0].coords[0]
            self._emit("travel", router.route(pos, entry))
            self._cut_part(part, entry)
            pos = entry
        self._emit("travel", router.route(pos, start))

    def _cut_part(self, part, entry):
        # Choose bridges (cut-ins) so every hole is reachable: grow from the outline, always
        # taking the shortest bridge that stays inside the part material.
        holes = part.children
        material = Polygon(part.poly.exterior.coords, [h.poly.exterior.coords for h in holes])
        material_ok = prep(material.buffer(self.eps * 10))
        bridges = {}
        connected, remaining = [part], list(holes)
        while remaining:
            best = None
            for c in connected:
                for h in remaining:
                    a, b = nearest_points(c.poly.exterior, h.poly.exterior)
                    seg = LineString([a, b])
                    if seg.length > self.eps and not material_ok.contains(seg):
                        continue
                    if best is None or seg.length < best[0]:
                        best = (seg.length, c, h, a.coords[0], b.coords[0])
            if best is None:  # weird geometry - just go straight in from the outline
                h = remaining[0]
                a, b = nearest_points(part.poly.exterior, h.poly.exterior)
                best = (0, part, h, a.coords[0], b.coords[0])
            _, c, h, a, b = best
            bridges.setdefault(id(c), []).append((a, h, b))
            connected.append(h)
            remaining.remove(h)
        self._cut_contour(part, entry, bridges)

    def _cut_contour(self, contour, start, bridges):
        ring = LineString(contour.poly.exterior.coords)
        L = ring.length
        t0 = ring.project(Point(start))
        excursions = sorted((((ring.project(Point(a)) - t0) % L, a, h, b)
                             for a, h, b in bridges.get(id(contour), [])), key=lambda e: e[0])
        cur = 0.0
        for r, a, h, b in excursions:
            self._emit("cut", ring_part(ring, t0 + cur, t0 + r))
            self._emit("bridge", [a, b])
            self._cut_contour(h, b, bridges)              # cut the hole outline
            if h.children:                                # parts sitting inside the hole
                self._visit_region(h.poly, h.children, b)
            self._emit("bridge", [b, a])
            cur = r
        self._emit("cut", ring_part(ring, t0 + cur, t0 + L))
