class RagTrimConfig:
    model_name = "../Qwen2.5-3B-Instruct"
    embedding_model_name = "../bge-small-en-v1.5"
    kvcache_dir = "./chunk_kvcache"
    # storage_dir = "doc_emb"       # index not used
    use_chunk_cache = True