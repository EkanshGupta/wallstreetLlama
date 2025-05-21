import praw
import pandas as pd
from datetime import datetime
import time
import globals
import helper
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
path = "./"
# helper.prepare_data_and_train()
helper.evaluate_model()