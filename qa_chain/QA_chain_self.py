from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import sys
sys.path.append("../")
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import sys
import re

class QA_chain_self():
    """"
    不带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火需要输入
    - api_key：所有模型都需要
    - Spark_api_secret：星火秘钥
    - embeddings：使用的embedding模型  
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq    
    """

    #基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    default_template_rq = """[角色]
你是一个严谨的问答专家，严格依据提供的知识库内容进行回答

[指令]
请按以下步骤处理：
1. 上下文核查：分析<Context>与<Question>的关联性
2. 知识提取：仅使用提供的上下文信息，禁止外部知识
3. 回答组织：
   - 若上下文包含明确答案：用2-5句话总结核心信息，然后剩余内容根据相关性进行补充。
   - 若上下文不相关/不充分：明确告知无法回答
   - 始终在结尾使用：期待为您提供更多帮助

[输出格式要求]
回答必须包含：
- 首段：直接的问题回答（可以直接使用原文内容）
- 第二段：剩余内容根据相关性进行补充
- 末段：固定结语（独立成段）

### 上下文 ###
{context}

### 问题 ###
{question}
有用的回答:"""

    def __init__(self, model:str, temperature:float=0.0, top_k:int=5,  file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None, embedding = "openai",  embedding_key = None, template=default_template_rq):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=self.template)
        self.retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': self.top_k})  #默认similarity，k=5
        # 自定义 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        retriever=self.retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})

    #基于大模型的问答 prompt 使用的默认提示模版
    #default_template_llm = """请回答下列问题:{question}"""
           
    def answer(self, question:str=None, temperature = None, top_k = 5):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """

        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
            
        if top_k == None:
            top_k = self.top_k

        result = self.qa_chain({"query": question, "temperature": temperature, "top_k": top_k})
        answer = result["result"]
        answer = re.sub(r"\\n", '<br/>', answer)
        return answer   
