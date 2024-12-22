import os
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

from helper_functions import get_pdf_file,text_split,download_embeddings

extracted_data=get_pdf_file(data='PDF')
text_chunks=text_split(extracted_data)
embeddings=download_embeddings()

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=PINECONE_API_KEY)


environment = "us-east-1-aws"

index_name = "mediexpert"

#MAKING INDEX WITHOUT GOING TO WEBSITE
pc.create_index(
    name=index_name,
    dimension=1536, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


#INSERT CHUNKS AS VECTOR DB TO THE PINECONE
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)