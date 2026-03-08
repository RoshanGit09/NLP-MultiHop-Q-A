"""
base.py — Abstract KG adapter interface.
All KG store implementations must subclass this.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class KGAdapterBase(ABC):
    """Store-agnostic interface for KG operations."""

    @abstractmethod
    def search_nodes(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Full-text / fuzzy search for nodes whose name contains `text`.
        Returns list of: {id, name, label, degree, match_type}
        """

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "both",   # "in" | "out" | "both"
        time_filter: Optional[Dict] = None,  # {after?, before?} ISO strings
    ) -> List[Dict[str, Any]]:
        """
        Return direct neighbors of node_id.
        Each item: {from_id, from_name, rel, to_id, to_name, to_label,
                    direction, t_announce?, t_effective?, source?}
        """

    @abstractmethod
    def find_paths(
        self,
        start_node_ids: List[str],
        end_node_ids: Optional[List[str]],
        constraints: Optional[Dict] = None,
        max_hops: int = 3,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop path search between start and (optionally) end nodes.
        Returns list of paths: {nodes: [...], triples: [...], score}
        """

    @abstractmethod
    def close(self) -> None:
        """Release connections."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
