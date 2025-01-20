
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

from hyperparameters import FREE_GPT_MODEL


def multilevel_search(query: str, file_index, paragraph_index, top_k_files: int=2) -> list:
    ''' 針對問題去各個file去找答案
    Args:
        query (str): 使用者輸入的文字
        file_index: 各個pdf的index
        paragraph_index: 各pdf的內容的index
        top_k_files (int): 參考前k名最相關的pdf

    Returns:
        list: 針對query相關的內容
    '''
    
    coarse_results = file_index.similarity_search(query, k=top_k_files)
    relevant_files = [result.metadata["source"] for result in coarse_results]
    print(f"Top-{top_k_files} relevant files: {relevant_files}")

    fine_results = []
    for file_name in relevant_files:
        fine_results.extend(paragraph_index[file_name].similarity_search(query, k=3))

    return fine_results

def run_multilevel_conversation(file_index, paragraph_index, top_k_files: int=2) -> None:
    ''' 實現連續對話
    Args:
        file_index: 各個pdf的index
        paragraph_index: 各pdf的內容的index
        top_k_files (int): 參考前k名最相關的pdf

    Returns:
        None: RAG的回覆用print呈現。
    '''
    llm = ChatOpenAI(temperature=0, model_name=FREE_GPT_MODEL)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # 找尋問題的相關段落
        relevant_paragraphs = multilevel_search(query, file_index, paragraph_index, top_k_files=top_k_files)
        context = "\n".join([f"{res.page_content} (Source: {res.metadata['source']})" for res in relevant_paragraphs])

        # 加入記憶中的對話歷史
        chat_history = memory.chat_memory.messages
        memory_context = "\n".join([
            f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}"
            for m in chat_history
        ])

        # 生成回答
        input_text = f"Context:\n{context}\n\nMemory:\n{memory_context}\n\nQuestion: {query}"
        response = llm.predict(input_text)

        # 更新記憶
        memory.save_context({"input": query}, {"output": response})

        # 添加答案來源資訊
        source_info = "\nSources:\n" + "\n".join(
            [f"- {res.metadata['source']} | Content: {res.page_content[:100]}..." for res in relevant_paragraphs]
        )

        print("\nBot:", response)
        # print(source_info)