import os
from pathlib import Path

root_dir = "output/longbench"
old_name = "pred_modev1_num4_retrieve_topk8_layer8.json"
new_name = "pred_modev1_chunk_num4_retrieve_topk8_layer8.json"

for subdir in Path(root_dir).iterdir():
    if subdir.is_dir():
        old_path = subdir / "llama2-7b-chat/mbr3" / old_name
        new_path = subdir / "llama2-7b-chat/mbr3" / new_name
        
        if old_path.exists():
            print(f"Renaming: {old_path} -> {new_path}")
            old_path.rename(new_path)
        else:
            print(f"File not found: {old_path}")