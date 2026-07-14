from __future__ import annotations

from functools import lru_cache
from typing import Optional

from neo4j import GraphDatabase

from rca.config import Settings, get_settings
from rca.schema import Entity, EntityKind, RelType, TopologyEdge

_CONSTRAINTS = [
    "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT host_id IF NOT EXISTS FOR (h:Host) REQUIRE h.id IS UNIQUE",
]

_TRAVERSABLE = "DEPENDS_ON|ROUTES_THROUGH"


class Neo4jClient:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.neo4j_password:
            raise RuntimeError("RCA_NEO4J_PASSWORD is not set")
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        with self.driver.session() as session:
            for stmt in _CONSTRAINTS:
                session.run(stmt)

    def upsert_components(self, entities: list[Entity]) -> None:
        rows = [{"id": e.id, "name": e.name, "kind": e.kind.value} for e in entities]
        with self.driver.session() as session:
            session.run(
                "UNWIND $rows AS r MERGE (c:Component {id: r.id}) SET c.name = r.name, c.kind = r.kind",
                rows=rows,
            )

    def upsert_edges(self, edges: list[TopologyEdge]) -> None:
        by_rel: dict[RelType, list[dict]] = {}
        for e in edges:
            by_rel.setdefault(e.rel, []).append({"src": e.src, "dst": e.dst, "weight": e.weight})
        with self.driver.session() as session:
            for rel, rows in by_rel.items():
                session.run(
                    "UNWIND $rows AS r MATCH (a:Component {id: r.src}) MATCH (b:Component {id: r.dst}) "
                    f"MERGE (a)-[e:{rel.value}]->(b) SET e.weight = r.weight",
                    rows=rows,
                )

    def get_components(self) -> list[Entity]:
        with self.driver.session() as session:
            rows = session.run("MATCH (c:Component) RETURN c.id AS id, c.name AS name, c.kind AS kind")
            return [
                Entity(id=r["id"], name=r["name"] or r["id"], kind=EntityKind(r["kind"] or "service"))
                for r in rows
            ]

    def get_edges(self) -> list[TopologyEdge]:
        out: list[TopologyEdge] = []
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (a:Component)-[e]->(b:Component) "
                "RETURN a.id AS src, b.id AS dst, type(e) AS rel, e.weight AS weight"
            )
            for r in rows:
                try:
                    rel = RelType(r["rel"])
                except ValueError:
                    continue
                out.append(
                    TopologyEdge(src=r["src"], dst=r["dst"], rel=rel, weight=r["weight"] if r["weight"] is not None else 1.0)
                )
        return out

    def dependency_paths(self, from_ids: list[str], to_ids: list[str], max_hops: int = 5, limit: int = 50) -> list[list[str]]:
        with self.driver.session() as session:
            rows = session.run(
                f"MATCH p = (a:Component)-[:{_TRAVERSABLE}*1..{max_hops}]->(b:Component) "
                "WHERE a.id IN $from_ids AND b.id IN $to_ids "
                "RETURN [n IN nodes(p) | n.id] AS path LIMIT $limit",
                from_ids=list(from_ids),
                to_ids=list(to_ids),
                limit=limit,
            )
            return [r["path"] for r in rows]


@lru_cache
def get_graph() -> Neo4jClient:
    return Neo4jClient()
