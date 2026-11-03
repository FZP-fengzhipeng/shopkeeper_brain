"""
  @Author:fengzhipeng
  @Time:2026/8/4
  @Desc:
"""
from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-m3",
                              local_dir=r"D:\soft\devsoft\ai_models\modelscope_cache\models\BAAI\bge-m3")
print(local_dir)