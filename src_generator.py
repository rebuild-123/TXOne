import os

from hyperparameters import OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from hyperparameters import (
    DOCUMENT_FOLDER_PATH,
    FILE_INDEX_NAME,
    OPENAI_KEY,
    PARAGRAPH_INDEX_FOLDER_NAME,
    PARAGRAPHS_NAME,
)
from utils import (
    pickle_dumps,
    save_indexes,
)

from utils.preprocess import (
    load_and_split_pdfs,
    create_multilevel_indexes,
)

# 分割pdf成一個個chunk
paragraphs = load_and_split_pdfs(pdf_folder=DOCUMENT_FOLDER_PATH)
pickle_dumps(path=PARAGRAPHS_NAME, target=paragraphs)
# 對chunk做embedding
file_index, paragraph_index = create_multilevel_indexes(file_paragraphs=paragraphs)
save_indexes(
    file_index=file_index, paragraph_index=paragraph_index, 
    file_index_path=FILE_INDEX_NAME, paragraph_index_folder_path=PARAGRAPH_INDEX_FOLDER_NAME,
)