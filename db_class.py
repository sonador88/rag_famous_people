import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents.base import Document
import chromadb
from func import split_text

class ChromaDB:
  '''
    Класс для работы с векторной БД Chroma
    Аргументы:
      path_to_bd - путь к векторной БД
      emb_model - наименование модели для построения эмбеддингов
  '''
  def __init__(self, path_to_bd: str, emb_model: str):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.embeddings_hf = self.get_embeddings_model(emb_model)
    self.path_to_bd = path_to_bd
    self.client = chromadb.PersistentClient(path=self.path_to_bd)
    #self.db = None

  def get_embeddings_model(self, emb_model: str):
    '''
      Функция для получения функции эмбеддингов
    '''
    model_kwargs = {'device': self.device}
    embeddings_hf = HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs=model_kwargs
    )
    return embeddings_hf

  def delete_collection(self, collection_name: str) -> None:
    '''
      Функция для удаления уже существующей коллекции
    '''
    try:
      #coll = client.get_collection(collection_name)
      self.client.delete_collection(collection_name)
      print(f'delete exist collection {collection_name}')
    except:
      pass

  def create_chromadb_from_filedata(
      self,
      path_to_file: str,
      max_number_doc: int,
      collection_name: str,
      chunk_size: int,
      chunk_overlap: int
    ) -> None:
    '''
      Функция для создания коллекции в векторной БД на основе данных из файла

      Аргументы:
        path_to_file - путь с файлом, данные из которого необходимо переложить в БД
        collection_name - наименование коллекции, в которой будут храниться данные

      Результат:
        Сохраненная коллекция на диске
    '''
    print('start save ...')
    # удаление в случае существования коллекции
    self.delete_collection(collection_name)

    # в файле информация разделена не одним переносом, а двумя, соотв. мы должны "перескакивать" через строку
    # path_to_file = data_path + WIKI_NOTES_PATH
    with open(path_to_file, 'r') as f:
      num_doc = 0
      num_chanks = 0
      for i, article in enumerate(f):
        if i % 2:  # перескакиваем пустые строки в файле
          continue
        num_doc += 1
        #print(num_doc, article)
        # разбиваем полученный текст на чанки
        doc = Document(page_content=article)
        chunks = split_text([doc], chunk_size, chunk_overlap)
        num_chanks += len(chunks)
        if not i:  # при первом добавлении создаем коллекцию
          db = Chroma.from_documents(
              documents=chunks,
              embedding=self.embeddings_hf,
              persist_directory=self.path_to_bd,
              collection_name=collection_name,
          )
        else:  # при последующих итерациях добавляем новые документы
          db.add_documents(documents=chunks)
        # если достигли заданного числа документов - выходим
        if num_doc >= max_number_doc:
          break
    print(f'save {num_doc}({num_chanks} chunks) to bd success!')

  def get_bd_documents(self, collection_name: str, number_doc: int = -1):
    '''
      Функция для получения документов их БД

      Аргументы:
        collection_name -  наименование коллекции
        number_doc - сколько документов требуется возвратить. Если не передано значения - возвращаем все
    '''
    collection = self.client.get_collection(name=collection_name)
    if number_doc < 0:
      res = collection.get(
          include=["documents","metadatas"],
      )
    else:
      res = collection.get(
          include=["documents","metadatas"],
          limit=number_doc
      )
    return res
    
        
