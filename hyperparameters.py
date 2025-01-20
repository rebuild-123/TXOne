import os

CURRENT_PATH = os.getcwd()


#######################################################################
# files

DOCUMENT_FOLDER_NAME = 'documents' # 貯存pdf的資料夾名稱
DOCUMENT_FOLDER_PATH = os.path.join(CURRENT_PATH, DOCUMENT_FOLDER_NAME) # 貯存pdf的資料夾路徑
FILES_NAME = [
    'txsp_guide_1.2.pdf',
    'txso_3.1_ag.pdf',
    'txsp_3.1_ag.pdf',
]


#######################################################################
# src

SRC_PATH = os.path.join(CURRENT_PATH, 'src') # pdf經過前處理後數據的貯存資料夾的路徑
PARAGRAPHS_NAME = os.path.join(SRC_PATH, 'paragraphs.pkl') # 經過NLP模型切割後內容的貯存路徑
FILE_INDEX_NAME = os.path.join(SRC_PATH, 'file_index.faiss') # 文章指標的貯存路徑
PARAGRAPH_INDEX_FOLDER_NAME = os.path.join(SRC_PATH, 'paragraph_index') # 內容指標的貯存路徑
FREE_GPT_MODEL = 'gpt-3.5-turbo' # 用於連續對話的LLM名稱


#######################################################################
# for pdf

CHUNK_SIZE = 500 # NLP用500個字元為大小切文章，是為一個chunk
CHUNK_OVERLAP = 50 # 每一個chunk與前一個chunk重疊50個字元


#######################################################################
# OpenAI

OPENAI_KEY = ''
SPACY_MODEL_NAME = 'en_core_web_sm' # 將文章切成一個個chunk的AI模型名稱。此AI模型自動按語義段落切分，避免語義斷裂。

