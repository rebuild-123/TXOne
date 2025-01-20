import os

import spacy

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def load_and_split_pdfs(pdf_folder: str, spacy_model_name: str) -> list[dict]:
    ''' 分割pdf_folder路徑下所有的pdf
    Args:
        pdf_folder (str): 存pdf資料夾的路徑
        spacy_model_name (str): 用來分割文章的AI模型的名稱

    Returns:
        list[dict]: list內有各個pdf被AI模型切割後的chunk
    '''
    nlp = spacy.load(spacy_model_name)

    file_paragraphs = {}
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {pdf_folder}.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {pdf_file}.")

        paragraphs = []
        for page in documents:
            text = page.page_content
            doc = nlp(text)
            current_paragraph = []
            for sent in doc.sents:
                current_paragraph.append(sent.text)
                if sent.text.strip().endswith((".", "!", "?")):
                    paragraph = " ".join(current_paragraph)
                    if is_valid_paragraph(paragraph):
                        paragraphs.append(paragraph)
                    current_paragraph = []
            if current_paragraph:
                paragraph = " ".join(current_paragraph)
                if is_valid_paragraph(paragraph):
                    paragraphs.append(paragraph)
        file_paragraphs[pdf_file] = paragraphs
    return file_paragraphs

def is_valid_paragraph(paragraph: str) -> bool:
    ''' 去除無用的字串
    Args:
        paragraph (str): 某被AI模型切割後的chunk

    Returns:
        bool: 是否為無用的字串
    '''
    if len(paragraph.strip()) < 30 or len(paragraph.strip()) > 1000:
        return False
    if paragraph.strip().isnumeric() or all(char in ".,!" for char in paragraph.strip()):
        return False
    common_phrases = ["page", "copyright", "all rights reserved"]
    if any(phrase in paragraph.lower() for phrase in common_phrases):
        return False
    return True

# 
def create_multilevel_indexes(file_paragraphs: dict) -> tuple[list, dict]:
    ''' 生成各個pdf的index和各pdf內容的index
    Args:
        file_paragraphs (dict): pdf被AI模型切割後的所有chunk

    Returns:
        tuple[list, dict]: 各個pdf的index, 各pdf的內容的index
    '''
    embeddings = OpenAIEmbeddings()

    # 生成各個pdf的index
    from langchain.schema import Document
    file_embeddings = []
    for file_name, paragraphs in file_paragraphs.items():
        file_content = " ".join(paragraphs) 
        file_embeddings.append(Document(page_content=file_content, metadata={"source": file_name}))

    file_index = FAISS.from_documents(file_embeddings, embeddings)
    print("File-level index (coarse retrieval) created.")

    # 生成各pdf內容的index
    paragraph_index = {}
    for file_name, paragraphs in file_paragraphs.items():
        documents = [
            Document(page_content=p, metadata={"source": file_name}) for p in paragraphs
        ]
        paragraph_index[file_name] = FAISS.from_documents(documents, embeddings)
    print("Paragraph-level index (fine retrieval) created.")

    return file_index, paragraph_index
