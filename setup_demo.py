import subprocess
import sys
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "datasets", "evaluate", "transformers[torch]", "accelerate",
    "huggingface", "pyspark", "svgling", "wordfreq"
])

if not Path("assigntools").exists():
    subprocess.check_call([
        "git", "clone", "https://github.com/mdzgr/assigntools.git"
    ])

from google.colab import drive
drive.mount('/content/drive')

import re
import torch
import torch.nn.functional as F
from typing import List, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

from MERGE.utils import *
from MERGE.demo import *
from assigntools.LoLa.read_nli import (
    snli_jsonl2dict,
    sen2anno_from_nli_problems
)
from assigntools.LoLa.deep_nli import (
    predict_nli,
    batch_predict_nli,
    load_tok_model
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "CUDA device:",
    torch.cuda.current_device() if device.type == "cuda" else "CPU"
)
