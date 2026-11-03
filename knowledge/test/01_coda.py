"""
  @Author:fengzhipeng
  @Time:2026/7/30
  @Desc:
"""
import torch

print(f"{torch.cuda.is_available()}")
print(f"{torch.cuda.get_device_name()}")