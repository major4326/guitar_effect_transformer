import shutil
import os

def remove_data(cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

cache_dir_1 = os.path.expanduser("~/.cache/huggingface/datasets")
cache_dir_2 = os.path.expanduser("~/.cache/huggingface/datasets")

remove_data(cache_dir_1)
remove_data(cache_dir_2)