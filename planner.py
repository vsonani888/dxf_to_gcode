"""Toolpath planning for a 2D hot-wire foam cutter.

The wire can never lift, so every move cuts. Rules:
  * travel  - moves between shapes go through free foam only, never through a part
  * cut     - each shape's outline is cut as one closed loop
  * bridge  - the only time the wire enters a part: to reach a hole inside it
              (it goes in and comes back out along the same line)

Nesting is handled with a contour tree:
  depth 0 = part outline, depth 1 = hole in that part,
  depth 2 = a separate part sitting inside that hole, and so on.
An outline drawn twice (a hole's edge plus the piece that fills that hole, as CAD exports
often do) is merged and cut once; the inside of such a hole counts as material.
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
        # True when the drawing had this exact outline twice: a hole edge and the outline of
        # a separate piece that fills that hole. The curve is cut once, and the inside of the
        # hole is material, not scrap.
        self.shared = False

    @property
    def is_hole(self):
        return self.depth % 2 == 1

    @property
    def material_inside(self):
        return not self.is_hole or self.shared


def merge_duplicates(nodes, tol):
    """Drop outlines that are the same curve as one already kept (any start point / direction)."""
    kept = []
    for n in nodes:
        same = next((k for k in kept
                     if abs(k.poly.area - n.poly.area) <= 0.01 * k.poly.area
                     and k.poly.hausdorff_distance(n.poly) <= tol), None)
        if same is None:
            kept.append(n)
        else:
            same.shared = True
    return kept


def build_tree(contours, eps, dup_tol):
    nodes = sorted((Contour(c) for c in contours), key=lambda n: -n.poly.area)
    nodes = merge_duplicates(nodes, dup_tol)
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


def spanning_tree(n, link):
    """Prim's minimum spanning tree over nodes 0..n-1, rooted at node 0.

    link(i, j) -> (length, path) or None when the straight connection is blocked.
    blocked(i, j) is only consulted (it is expensive - it routes around obstacles)
    when no straight connection to the remaining nodes exists.
    Returns a list of (parent, child, path) in the order the nodes were added.
    """
    in_tree, edges = [0], []
    left = set(range(1, n))
    while left:
        best = None
        for i in in_tree:
            for j in left:
                e = link(i, j, route=False)
                if e and (best is None or e[0] < best[0]):
                    best = (e[0], i, j, e[1])
        if best is None:  # everything left is hidden behind other parts - route around them
            for i in in_tree:
                for j in left:
                    e = link(i, j, route=True)
                    if best is None or e[0] < best[0]:
                        best = (e[0], i, j, e[1])
        _, i, j, path = best
        edges.append((i, j, path))
        in_tree.append(j)
        left.remove(j)
    return edges


class Planner:
    """Builds the cut as a tree of detours.

    The wire cuts an outline; at the point closest to a neighbouring shape it branches off
    (travel), cuts that shape (which may branch further), comes back along the same line and
    carries on with the outline it was cutting. Holes are reached the same way via bridges.
    The branches form a minimum spanning tree, so the total connecting length is as short as
    the shapes' layout allows, and the wire always ends where it started: at the origin.
    """

    def __init__(self, contours, origin=(0.0, 0.0), clearance=None):
        allpts = [p for c in contours for p in c] + [origin]
        xs, ys = zip(*allpts)
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        diag = math.hypot(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1])
        self.eps = diag * 1e-6
        self.clearance = clearance if clearance is not None else diag * 0.005
        self.origin = tuple(origin)
        self.roots, self.contours = build_tree(contours, self.eps * 10, dup_tol=diag * 5e-5)
        self.merged = len(contours) - len(self.contours)  # duplicate outlines dropped
        self.moves = []  # list of (kind, [(x, y), ...]) with kind in travel / cut / bridge

    def plan(self):
        minx, miny, maxx, maxy = self.bounds
        m = max(maxx - minx, maxy - miny)
        world = box(minx - m, miny - m, maxx + m, maxy + m)
        # detours[id(contour)] = [(point on contour, kind, path, target contour, point on target)]
        # detours[None] = branches leaving the origin
        self.detours = {}
        self._connect_region(world, self.roots, None)
        self.moves = []
        for _, kind, path, target, b in self.detours.get(None, []):
            self._emit(kind, path)
            self._cut_contour(target, b)
            self._emit(kind, path[::-1])
        return self.moves

    def _emit(self, kind, coords):
        coords = [tuple(c) for c in coords]
        if len(coords) < 2:
            return
        if self.moves and self.moves[-1][0] == kind and math.dist(self.moves[-1][1][-1], coords[0]) < 1e-9:
            self.moves[-1][1].extend(coords[1:])
        else:
            self.moves.append((kind, coords))

    def _connect_region(self, container, parts, root):
        """Link all parts lying in `container` into a tree hanging off `root`
        (the origin when root is None, otherwise the hole contour they sit in)."""
        if not parts:
            return
        # links run through scrap foam, unless the hole is filled by a piece of its own
        kind = "bridge" if root is not None and root.shared else "travel"
        router = Router(container, [p.poly for p in parts], self.clearance, self.eps)
        geoms = [Point(self.origin) if root is None else root.poly.exterior] + [p.poly.exterior for p in parts]
        owners = [root] + list(parts)
        cache = {}

        def link(i, j, route):
            key = (i, j)
            if key not in cache:
                a, b = (p.coords[0] for p in nearest_points(geoms[i], geoms[j]))
                cache[key] = (math.dist(a, b), [a, b]) if router.visible(a, b) else None
            if cache[key] is None and route:
                a, b = (p.coords[0] for p in nearest_points(geoms[i], geoms[j]))
                path = router.route(a, b)
                cache[key] = (LineString(path).length if len(path) > 1 else 0.0, path)
            return cache[key]

        for i, j, path in spanning_tree(len(geoms), link):
            owner = owners[i]
            self.detours.setdefault(None if owner is None else id(owner), []).append(
                (path[0], kind, path, owners[j], path[-1]))

        for part in parts:
            self._connect_holes(part)

    def _connect_holes(self, part):
        """Bridges (cut-ins) so every hole is reachable: grow from the outline, always taking
        the shortest bridge that stays inside the part material."""
        holes = part.children
        material = Polygon(part.poly.exterior.coords, [h.poly.exterior.coords for h in holes])
        material_ok = prep(material.buffer(self.eps * 10))
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
            self.detours.setdefault(id(c), []).append((a, "bridge", [a, b], h, b))
            connected.append(h)
            remaining.remove(h)
        for h in holes:  # parts sitting inside a hole hang off that hole's outline
            self._connect_region(h.poly, h.children, h)

    def _cut_contour(self, contour, start):
        """Cut one closed outline starting and ending at `start`, taking every detour on the way."""
        ring = LineString(contour.poly.exterior.coords)
        L = ring.length
        t0 = ring.project(Point(start))
        detours = sorted((((ring.project(Point(a)) - t0) % L, kind, path, target, b)
                          for a, kind, path, target, b in self.detours.get(id(contour), [])),
                         key=lambda d: d[0])
        cur = 0.0
        for r, kind, path, target, b in detours:
            self._emit("cut", ring_part(ring, t0 + cur, t0 + r))
            self._emit(kind, path)
            self._cut_contour(target, b)
            self._emit(kind, path[::-1])
            cur = r
        self._emit("cut", ring_part(ring, t0 + cur, t0 + L))
