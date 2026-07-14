from __future__ import annotations

from typing import Iterable, Optional

import networkx as nx

from rca.schema import RelType, TopologyEdge

_DEP_RELS = {RelType.depends_on.value, RelType.routes_through.value}


class TopologyGraph:
    def __init__(self, edges: list[TopologyEdge]) -> None:
        self.g = nx.DiGraph()
        for e in edges:
            if e.rel.value in _DEP_RELS:
                self.g.add_edge(e.src, e.dst, weight=float(e.weight), rel=e.rel.value)
            else:
                self.g.add_node(e.src)
                self.g.add_node(e.dst)

    @classmethod
    def from_edges(cls, edges: list[TopologyEdge]) -> "TopologyGraph":
        return cls(edges)

    def nodes(self) -> list[str]:
        return list(self.g.nodes)

    def has_node(self, n: str) -> bool:
        return self.g.has_node(n)

    def fault_relevance_zone(self, seeds: Iterable[str], hops: int = 1) -> list[str]:
        zone: set[str] = set()
        for s in seeds:
            if not self.g.has_node(s):
                continue
            zone.add(s)
            zone.update(self._within(s, hops))
        return list(zone)

    def _within(self, node: str, hops: int) -> set[str]:
        seen = {node}
        frontier = {node}
        for _ in range(hops):
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(self.g.successors(n))
                nxt.update(self.g.predecessors(n))
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
        seen.discard(node)
        return seen

    def personalized_pagerank(
        self,
        personalization: dict[str, float],
        nodes: Optional[Iterable[str]] = None,
        damping: float = 0.85,
        max_iter: int = 100,
    ) -> dict[str, float]:
        sub = self.g.subgraph(list(nodes)).copy() if nodes is not None else self.g
        if sub.number_of_nodes() == 0:
            return {}
        pers = {n: max(0.0, float(personalization.get(n, 0.0))) for n in sub.nodes}
        if sum(pers.values()) <= 0:
            pers = {n: 1.0 for n in sub.nodes}
        try:
            return nx.pagerank(sub, alpha=damping, personalization=pers, weight="weight", max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            return nx.pagerank(sub, alpha=damping, personalization=pers, weight="weight", max_iter=max_iter * 20, tol=1e-3)

    def path(self, src: str, dst: str) -> Optional[list[str]]:
        if not (self.g.has_node(src) and self.g.has_node(dst)):
            return None
        try:
            return nx.shortest_path(self.g, src, dst)
        except nx.NetworkXNoPath:
            return None

    def reaches(self, src: str, dst: str) -> bool:
        return self.path(src, dst) is not None

    def impact_path(self, cause: str, symptom: str) -> Optional[list[str]]:
        p = self.path(symptom, cause)
        if p is None:
            return None
        return list(reversed(p))
