# RAG для диалоговой системы

## Цель проекта
Создание ассистента, который сможет отвечать на любые вопросы про жизнь известных личностей. Для этого была реализована поддержка диалога в RAG, а также к семантическому поиску по базе знаний добавлен поиск информации в интернете. Поддержка диалога означает, что пользователь сможет уточнять любую информацию по предыдущему вопросу без необходимости задавать весь вопрос целиком.

## Виртуальное окружение
Python 3.11.11

Все необходимые версии библиотек лежат в requirements.txt, для установки выполните следующую команду:
```bush
pip install requirements.txt. 
```

## Используемые технологии
- <b>ChromaDB</b> - в качестве хранилища данных;
- <b>googlesearch</b> - для обращения к поисковику, <b>request</b> и <b>BeautifulSoup</b> для парсинга веб-страниц;
- <b>langchain</b> - как основной фреймворк для построения RAG системы;
- <b>hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4</b> - выбранная модель LLM для генерации ответов;
- <b>intfloat/multilingual-e5-large</b> - модель для построения эмбеддингов.

## Структура проекта
- <a href=https://github.com/sonador88/rag_famous_people/blob/main/rag.ipynb>rag.ipynb</a> - основной файл со всей логикой программы;
- <a href=https://github.com/sonador88/rag_famous_people/blob/main/db_class.py>db_class.py</a> - здесь хранится реализация класса для взаимодействия с бд и ее создания;
- <a href=https://github.com/sonador88/rag_famous_people/blob/main/func.py>func.py</a> - файл с дополнительным функционалом;
- <a href=https://github.com/sonador88/rag_famous_people/blob/main/config.py>config.py</a> - файл с настройками, в котором можно задавать используемые модели, параметры деления текстов на чанки, кол-во отбираемых документов для контекста модели и др.;
- <a href=https://github.com/sonador88/rag_famous_people/tree/main/chroma_db>chroma_db</a> - папка со сгенерированной БД;
- <a href=https://github.com/sonador88/rag_famous_people/tree/main/data>data</a> -  папка с входными данными и выходными;
- <a href=https://github.com/sonador88/rag_famous_people/blob/main/requirements.txt>requirements.txt</a> - файл с зависимостями.

## Алгоритм решения задачи
- Данные из файла <i>data/input/ru_wiki_person.txt</i> переносим в ChromaDB, используя разбиение текстов на чанки. Cохраняем базу данных на диск с названием chroma_db;
- Обходим вопросы из файла <i>data/input/dialog_questions.txt</i>. Первый вопрос задаем LLM без изменений, последующие при помощи этой же модели переформулировываем с учетом истории таким образом, чтобы они имели однозначный ответ;
- Для каждого вопроса к LLM выбираем из базы знаний релевантные тексты. Если таких текстов находится меньше заданного числа - идем в поисковик. Найденные в интернете тексты делим на чанки аналогично тому, как делили исходные документы для сохранения в БД;
- Передаем LLM контекст, состоящий из релевантных документов и переформулированный вопрос с учетом истории диалога;
- Сохраняем ответы LLM в файле <i>data/output/dialog_answers.json</i>.

## Полученный результат
В результате запуска <i>rag.ipynb</i> будет перегенерирована БД в соответствии с файлом <i>data/input/ru_wiki_person.txt</i> и сохранены ответы LLM в файл <i>data/output/dialog_answers.json</i>. Увидеть полученные результаты можно в конце ноутбука.
