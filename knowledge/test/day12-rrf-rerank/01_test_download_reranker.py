#test_bge_rerank_model.py
from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large",
              local_dir="D:\\soft\\devsoft\\modelscope_cache\\models\\BAAI\\bge-reranker-large")

print(local_dir)