from langchain_text_splitters.character import RecursiveCharacterTextSplitter

def split_text(documents, chunk_size: int, chunk_overlap: int):
    '''
        Функция для деления текстов на чанки
        
        Аргументы: 
            documents - документы, которые требуется разбить на чанки
            chunk_size - размер чанка
            chunk_overlap - перекрытие чанков
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    #print(f"Разбили документ на {len(chunks)} чанков.")
    return chunks
