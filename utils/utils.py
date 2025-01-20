import os
import typing

import pickle

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def pickle_dumps(path: str, target: typing.Any) -> typing.NoReturn:
    with open(path, "wb") as file:
        file.write(pickle.dumps(target))

def pickle_loads(path: str) -> typing.Any:
    with open(path, "rb") as file:
        return pickle.loads(file.read())

def save_indexes(file_index, paragraph_index, file_index_path: str, paragraph_index_folder_path: str) -> None:
    ''' 貯存檔案的index和各篇文章內容的index
    Args:
        file_index: 各個pdf的index
        paragraph_index: 各pdf的內容的index
        file_index_path (str): 貯存file_index的路徑
        paragraph_index_folder_path (str): 貯存paragraph_index的路徑

    Returns:
        None
    '''
    
    file_index.save_local(file_index_path)
    print(f"File-level index saved to {file_index_path}.")
    
    os.makedirs(paragraph_index_folder_path, exist_ok=True)
    for file_name, index in paragraph_index.items():
        index_path = os.path.join(paragraph_index_folder_path, f"{file_name}.faiss")
        index.save_local(index_path)
    print(f"Paragraph-level indexes saved to {paragraph_index_folder_path}.")

def load_indexes(file_index_path: str, paragraph_index_folder_path: str) -> tuple[list, dict]:
    ''' 下載檔案的index和各篇文章內容的index
    Args:
        file_index_path (str): 貯存file_index的路徑
        paragraph_index_folder_path (str): 貯存paragraph_index的路徑

    Returns:
        tuple[list, dict]: file_index, paragraph_index
    '''
    
    file_index = FAISS.load_local(file_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print(f"File-level index loaded from {file_index_path}.")
    
    paragraph_index = {}
    for file_name in os.listdir(paragraph_index_folder_path):
        if file_name.endswith(".faiss"):
            file_path = os.path.join(paragraph_index_folder_path, file_name)
            index_name = file_name.replace(".faiss", "")
            paragraph_index[index_name] = FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print(f"Paragraph-level indexes loaded from {paragraph_index_folder_path}.")
    
    return file_index, paragraph_index