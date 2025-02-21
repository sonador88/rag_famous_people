WIKI_NOTES_PATH = 'data/input/ru_wiki_person.txt'  # путь к файлу со статьями из Wiki 
NUMBER_DOC_FROM_WIKI = 100  # число документов из Wiki, которое будем грузить в бд
CHUNK_SIZE = 1000  # количество символов на одну запись в БД
CHUNK_OVERLAP = 100  # кол-во символов на перекрытие соседних записей из текста для загрузки в БД
CHROMA_PATH = "chroma_db"  # путь, по которому лежит векторная БД
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'  # модель, при помощи которой будем строить ембеддинги
COLLECTION_NAME = f'famous_people_{CHUNK_SIZE}'   # имя коллекции в БД

SIMILARITY_THRESHOLD = 0.75  # начиная от какой доли совпадения будем добавлять тексты в контекст промпта
QUESTIONS_DIALOG = 'data/input/dialog_questions.txt'  # файл с вопросами к LLM
ANSWERS_DIALOG = 'data/output/dialog_answers.json'  # файл с ответами LLM 
# INTERNET_CHROMA_PATH = 'chroma_internet_db'
NUMBER_DOCS_FOR_CONTEXT = 3  # количество документов, которые будут отбираться для контекста модели
LLM_MODEL = 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'

