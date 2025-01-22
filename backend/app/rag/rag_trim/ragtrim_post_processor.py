from typing import List, Optional, Any

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from app.rag.rag_trim.prefetch_ragtrim import binary_search_evaluate_chunks_relevance
import asyncio
import logging
import time
_logger = logging.getLogger(__name__)

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

        _logger.warning(f"RagTrim: Postprocessing {len(nodes)} nodes.")
        # binary search to find the first node that not relevant
        start = time.perf_counter()
        boundry = asyncio.run(
            binary_search_evaluate_chunks_relevance(
                query_bundle.query_str,
                nodes, 
                self.use_kv_cache, 
                self.prefetch
            )
        )
        end = time.perf_counter()
        _logger.warning(f"RagTrim: Binary search took {end - start} seconds.")
        _logger.warning(f"RagTrim: Boundry found at {boundry}.")

        relevant_nodes = nodes[:boundry]
        return relevant_nodes

