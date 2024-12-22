from flask import Flask,render_template,jsonify,request
app = Flask(__name__)
import os
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from helper_functions import download_embeddings

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_pinecone import PineconeVectorStore
index_name="mediexpert"

embeddings=download_embeddings()

#change our query to make it vectors in  pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

#initialization of openai LLM
llm = OpenAI(temperature=0.4, max_tokens=500)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "If user wants to be interactive, butter him up"
    "Even  if you are assistant,  if user asks how are you?, answer him i am fine"
    "\n\n"
    "You are free to tell that you learnt it from Medicine Encyclopedia book"
    "{context}"
    "You are free to tell the medicine name if you know.But at last give the warning to consult physician before consuming it."
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human", "{input}"),
    ]
)

#perform similarity search on our query and knowledge base already available in the pinecone
#return 3 potential answers to the queries
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/query", methods=["GET", "POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "")
    print(input)
    print("Response to this")
    reply=rag_chain.invoke({"input":msg})
    return {"reply":reply["answer"]}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000, debug= True)