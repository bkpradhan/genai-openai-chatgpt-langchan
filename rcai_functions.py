import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage, ChatMessage
# from langchain.tools import format_tool_to_openai_functions
from dotenv import find_dotenv, load_dotenv
import openai
from functions import  get_order_desc, show_log_entries, get_order_activities
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_vectordb_from_pdf(video_url: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    db = FAISS.from_documents(docs, embeddings)
    return db


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
                        "description": "The order number, e.g. GPQ-1234",
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
        "description": "Get Order activity summary of a given order number",
        "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number, e.g. GPQ-1234",
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



def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)


# Start a conversation with multiple requests

    
    user_prompt=query
    user_prompt1 = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about Application Production Support issues in Capital Markets 
        based on provided production incident data and relevant use cases.
        
        Answer the following question: {question}
        By searching the following usecase data: {docs}
        
        When you are asked track a given an order number ( for example, GPQ-1234), describe that given order looking at order number
        Provide additional details by extracting logs from NAS using order number.
        Also provide activity summary of that order

        When you are asked further to provide more details on Orders given an order number ( for example, XYZ-1234), get more details on that order
        """,
    )
    

    # Returns the function of the first request 

    first_response = llm.predict_messages(
        [HumanMessage(content=user_prompt)], functions=function_descriptions_multiple
    )

    print(first_response)
    #print(str(first_response.additional_kwargs))

    # Returns the function of the second request (book_flight)
    # It takes all the arguments from the prompt but not the returned information

    second_response = llm.predict_messages(
        [
            HumanMessage(content=user_prompt),
            AIMessage(content=str(first_response.additional_kwargs)),
            AIMessage(
                role="function",
                additional_kwargs={
                    "name": first_response.additional_kwargs["function_call"]["name"]
                },
                content=f"Completed function {first_response.additional_kwargs['function_call']['name']}",
            ),
        ],
        functions=function_descriptions_multiple,
    )

    print(second_response)

    # Returns the function of the third request (file_complaint)

    third_response = llm.predict_messages(
        [
            HumanMessage(content=user_prompt),
            AIMessage(content=str(first_response.additional_kwargs)),
            AIMessage(content=str(second_response.additional_kwargs)),
            AIMessage(
                role="function",
                additional_kwargs={
                    "name": second_response.additional_kwargs["function_call"]["name"]
                },
                content=f"Completed function {second_response.additional_kwargs['function_call']['name']}",
            ),
        ],
        functions=function_descriptions_multiple,
    )

    print(third_response)

    # Conversational reply at the end of requests

    fourth_response = llm.predict_messages(
        [
            HumanMessage(content=user_prompt),
            AIMessage(content=str(first_response.additional_kwargs)),
            AIMessage(content=str(second_response.additional_kwargs)),
            AIMessage(content=str(third_response.additional_kwargs)),
            AIMessage(
                role="function",
                additional_kwargs={
                    "name": third_response.additional_kwargs["function_call"]["name"]
                },
                content=f"Completed function {third_response.additional_kwargs['function_call']['name']}",
            ),
        ],
        functions=function_descriptions_multiple,
    )

    print(fourth_response)


    # It automatically fills the arguments with correct info based on the prompt
    # Note: the function does not exist yet

    output = completion.choices[0].message
    print(output)
    return output, docs

# Example usage:
pdf_path = "./DummyUseCase.pdf"
db = create_vectordb_from_pdf(pdf_path)

user_prompt = """
        You are a helpful assistant that that can answer questions about Application Production Support issues in Capital Markets 
        based on provided production incident data and relevant use cases.
        
        Can you track order number GPQ-1234, get more descriptions on that order
        Provide additional details by extracting logs from NAS using order number.
        Also provide activity summary of that order
       """
query = "Can you tack order number GPQ1234?"
response, docs = get_response_from_query(db, user_prompt)
# print(textwrap.fill(response, width=200))


