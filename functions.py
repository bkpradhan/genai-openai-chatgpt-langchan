import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from datetime import datetime, timedelta

load_dotenv(find_dotenv())

def get_order_desc(order_id):
    """
    In this example, the function converts the input order id to a descriptive text message.
    """
    if order_id.startswith(('XMP', 'XMA', 'XAC')):
        response="the order is an Advisory Order"
    elif order_id.startswith(('GPQ', 'XYZ', 'PQR')):
        response="the order is a Brokerage Order"
    else:
        response ="the order is External FIX Clients"
    return response

def show_log_entries(order_id):
    """
    In this example, the function converts the input order id to a descriptive text message.
    """
    if order_id.startswith(('XMP', 'XMA', 'XAC')):
        response="the order is an Advisory Order"
    elif order_id.startswith(('ABC', 'XYZ', 'PQR')):
        response="the order is a Brokerage Order"
    else:
        response ="the order is External FIX Clients"
    return response

def get_order_activities(order_id):
    """
    In this example, the function converts the input order id to a descriptive text message.
    """
    if order_id.startswith(('XMP', 'XMA', 'XAC')):
        response="the order is an Advisory Order"
    elif order_id.startswith(('ABC', 'XYZ', 'PQR')):
        response="the order is a Brokerage Order"
    else:
        response ="the order is External FIX Clients"
    return response

