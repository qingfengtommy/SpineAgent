import os
import sys
this_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_file_dir, "../../"))

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
