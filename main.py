import os

from hyperparameters import OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from hyperparameters import (
    FILE_INDEX_NAME,
    PARAGRAPH_INDEX_FOLDER_NAME,
)
from utils import load_indexes
from utils.RAG import run_multilevel_conversation

file_index, paragraph_index = load_indexes(file_index_path=FILE_INDEX_NAME, paragraph_index_folder_path=PARAGRAPH_INDEX_FOLDER_NAME)
run_multilevel_conversation(file_index, paragraph_index, top_k_files=3)