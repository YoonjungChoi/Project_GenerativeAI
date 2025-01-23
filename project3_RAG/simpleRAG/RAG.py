import os.path
from os.path import isfile, join
from os import listdir
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

'''
pip install python-dotenv
pip install langchain
pip install openai
pip install langchain_community
pip install langchain_openai
pip install docarray
'''
#load API Key
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY == 'xxxxxxxx':
    raise ValueError("please make sure your OPENAI_API_KEY in .env file")

#load LLM model and embedding
#llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
embeddings = OpenAIEmbeddings()

resource_folder = "./docs/"
DataSource = Literal["Wikipedia", "Research Paper", "My Script"]
SUPPORTED_DATA_SOURCES = get_args(DataSource)

def load_data_set(source: DataSource, query:str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if source == ("Wikipedia"):
        Wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        data = Wikipedia.run(query)
        split_docs = [Document(page_content=sent) for sent in data.split('\n')]
    else:
        files = [f for f in listdir(resource_folder) if isfile(join(resource_folder, f))]
        file = resource_folder + files[0]
        loader = PyPDFLoader(file)
        data = loader.load()
        split_docs = text_splitter.split_documents(data)

    data_set = DocArrayInMemorySearch.from_documents(documents=split_docs, embedding=embeddings)
    return data_set

def retrieve_info(source: DataSource, data_set: DocArrayInMemorySearch, query:str):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = data_set.as_retriever(),
        verbose=True
    )
    output = qa.invoke(query)
    print("LOG output", output)
    return output


def generate_response(selection: DataSource, query: str):
    if selection not in SUPPORTED_DATA_SOURCES:
        raise ValueError(f"selected data source {selection} is not supported.")

    data_set = load_data_set(selection, query)
    response = retrieve_info(selection, data_set, query)
    return response

