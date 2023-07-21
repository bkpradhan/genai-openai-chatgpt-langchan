import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
#from langchain.tools import format_tool_to_openai_functions
from dotenv import find_dotenv, load_dotenv
import openai
from functions import get_order_desc
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_vectordb_from_pdf(video_url: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    #pages = loader.load_and_split()
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    if a model can handle up to 4097 tokens, set the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    #llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about Application Production Support issues in Capital Markets 
        based on provided production incident data and relevant use cases.
        
        Answer the following question: {question}
        By searching the following usecase data: {docs}
        
        Only use the factual information from the usecase data to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.

        When you are asked further to provide more details on Orders given an order number ( for example, XYZ-1234), get more details on that order
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)


    function_descriptions_multiple = [
        {
            "name": "get_order_desc",
            "description": "Describe a given Order looking at Order number",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number, e.g. GPQ-1234",
                    },
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "show_log_entries",
            "description": "Extract logs from NAS where Order number is present",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number, e.g. XYZ-1234",
                    },
                    "log_text": {
                        "type": "string",
                        "description": "Log lines extracted from NAS",
                    },
                },
                "required": ["order_id", "log_text"],
            },
        },
        {
            "name": "get_order_activities",
            "description": "Get Order activity summary of a given order number from DB or via REST",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number, e.g. XYZ-1234",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "The symbol used to place the order, e.g. IBM",
                    },
                    "quantity": {
                        "type": "string",
                        "description": "Quanity of the order placed",
                    },
                    "side": {
                        "type": "string",
                        "description": "side of the order placed",
                    },
                },
                "required": ["order_id", "symbol", "quantity", "side"],
            },
        },
    ]
# Start a conversation with multiple requests



    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Example usage:
pdf_path = "./DummyUseCase.pdf"
db = create_vectordb_from_pdf(pdf_path)

#query = "Can you tack order number XYZ-1234?"
query = "What could cause an order to be delayed in order blotter?"
print(query)
response, docs = get_response_from_query(db, query)
print(textwrap.fill(response, width=200))
