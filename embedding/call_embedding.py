import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

def get_embedding(embedding_name: str, embedding_key: str=None):
    if embedding_name == 'm3e':
        return HuggingFaceEmbeddings(model_name="/data/DeHors_yh_0329/code/Homework-yjs1/NLP/moka-ai/m3e-base")
    elif embedding_name == 'bge':
        return HuggingFaceEmbeddings(model_name="/data/DeHors_yh_0329/code/Lab-tianchi-KUAKE-QQR/code/BAAI/bge-large-zh-v1.5")
    elif embedding_name == 'openai':
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    elif embedding_name == 'zhipuai':
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_name}")