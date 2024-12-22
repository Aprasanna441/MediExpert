import os
from langchain.document_loaders import PyPDFLoader,DirectoryLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')


#loading pdf file content
def get_pdf_file(data):
    loader= DirectoryLoader(data,
                        glob="*.pdf",
                        loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

#chunking pdf to chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


def download_embeddings():
  embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"])
  return embeddings