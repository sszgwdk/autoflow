from typing import List, Optional, Any

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from backend.app.rag.rag_trim.prefetch_ragtrim import binary_search_evaluate_chunks_relevance
import asyncio

class RagTrimPostprocessor(BaseNodePostprocessor):
    # filters: Optional[MetadataFilters] = None
    use_kv_cache: bool = True
    prefetch: bool = True

    def __init__(
            self, 
            use_kv_cache: bool = True, 
            prefetch: bool = True, 
            **kwargs: Any
        ):
        
        super().__init__(**kwargs)
        self.use_kv_cache = use_kv_cache
        self.prefetch = prefetch

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("RagTrim: Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        # binary search to find the first node that not relevant
        boundry = asyncio.run(
            binary_search_evaluate_chunks_relevance(
                nodes, 
                query_bundle, 
                self.use_kv_cache, 
                self.prefetch
            )
        )

        relevant_nodes = nodes[:boundry]
        return relevant_nodes

