import asyncio
import torch
from transformers import AutoTokenizer

# Llama Index Related
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore

from app.rag.rag_trim.ragtrim_config import RagTrimConfig
from app.rag.rag_trim.qwen2 import Qwen2ModifiedForCausalLM

def evaluate_chunk_relevance_prompt(chunk, query):
    prompt = f"""

    Document: {chunk}

    Query: {query}

    You are an expert in evaluating documents for their relevance in answering user queries.
    Your task is to analyze a given document and query and determine whether the document is helpful for answering the query.
    Only return True or False:
        - True: If the document provides information relevant to the query.
        - False: If document does not provide information relevant to the query.
    Do not include any explanation, punctuation, or additional content.

    Answer:
    """

    return prompt

def stack_past_key_values(past_key_values_list):
    num_layers = len(past_key_values_list[0])
    batch_past_key_values = []
    for layer in range(num_layers):
        keys = torch.cat([past_key_values[layer][0] for past_key_values in past_key_values_list], dim=2)
        values = torch.cat([past_key_values[layer][1] for past_key_values in past_key_values_list], dim=2)
        batch_past_key_values.append((keys, values))
    return tuple(batch_past_key_values)

ragtrim_config = RagTrimConfig()
# Set up device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load model and tokenizer globally
model_for_ragtrim = Qwen2ModifiedForCausalLM.from_pretrained(
    ragtrim_config.model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer_for_ragtrim = AutoTokenizer.from_pretrained(ragtrim_config.model_name)
# Set up embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=ragtrim_config.embedding_model_name)

def evaluate_chunk_relevance(
        query_text: str, 
        node: NodeWithScore, 
        use_chunk_cache=True, 
        preloaded_kvcache=None
    ) -> bool:
    kvcache_list = []
    
    if use_chunk_cache:
        if preloaded_kvcache:
            kvcache_list.append(preloaded_kvcache)
        else:
            kvcache_path = ragtrim_config.kvcache_dir + f"/{node.node_id()}.pt"
            kvcache = torch.load(kvcache_path)
            kvcache_list.append(kvcache)
    
    prompt = evaluate_chunk_relevance_prompt(node.text, query_text)
    input_ids = tokenizer_for_ragtrim.encode(prompt, return_tensors='pt').to(model_for_ragtrim.device)
    past_kvcache = stack_past_key_values(kvcache_list) if use_chunk_cache else None
    eos_token_ids = [151645, 151643]
    
    with torch.no_grad():
        outputs = model_for_ragtrim.generate(
            input_ids,
            max_new_tokens=1,
            past_key_values=past_kvcache,
            pad_token_id=tokenizer_for_ragtrim.eos_token_id,
            do_sample=False,
            eos_token_id=eos_token_ids,
        )
    
    # Decode the output tokens to text
    decoded_output = tokenizer_for_ragtrim.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    # Fuzzy matching for True or False
    if "t" in decoded_output:
        return True
    elif "f" in decoded_output:
        return False
    else:
        # Handle unexpected output
        raise ValueError(f"Unexpected model output: {decoded_output}")

# async load kvcache
async def async_load_kvcache(kvcache_path: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, torch.load, kvcache_path)

async def binary_search_evaluate_chunks_relevance(
        query_text: str, 
        nodes: list[NodeWithScore], 
        use_chunk_cache=True,
        prefetch=True
    ) -> int:
    left, right = 0, len(nodes) - 1

    preload_tasks = {}  # async tasks:<node_id, task>
    current_kvcache = None

    while left <= right:
        mid = (left + right) // 2

        if use_chunk_cache:
            if not current_kvcache:
                kvcache_path = ragtrim_config.kvcache_dir + f"/{nodes[mid].node_id()}.pt"
                current_kvcache = torch.load(kvcache_path)
            # async prefetch
            if prefetch:
                next_left = (left + mid - 1) // 2
                next_right = (mid + 1 + right) // 2

                # next possible indices
                next_indices = [next_left, next_right]
                next_indices = [idx for idx in next_indices if 0 <= idx < len(nodes)]  # 边界检查

                # start async prefetch tasks
                for idx in next_indices:
                    if nodes[idx].node_id() not in preload_tasks:
                        kvcache_path = ragtrim_config.kvcache_dir + f"/{nodes[idx].node_id()}.pt"
                        preload_tasks[nodes[idx].node_id()] = asyncio.create_task(async_load_kvcache(kvcache_path))

        # use local llm to evaluate relevance of current chunk
        if evaluate_chunk_relevance(query_text, nodes[mid], use_chunk_cache, preloaded_kvcache=current_kvcache):
            left = mid + 1
        else:
            right = mid - 1

        if use_chunk_cache:
            current_kvcache = None
            if prefetch:
                next_idx = (left + right) // 2
                if nodes[next_idx].node_id() in preload_tasks:
                    current_kvcache = await preload_tasks.pop(nodes[next_idx].node_id())

                # clear other tasks
                for task in preload_tasks.values():
                    if not task.done():
                        task.cancel()
                # wait for all tasks to complete, to avoid resource leak
                try:
                    await asyncio.gather(*preload_tasks.values(), return_exceptions=True)
                except Exception:
                    pass  # ignore exceptions
                preload_tasks.clear()

    return left