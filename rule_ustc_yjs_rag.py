# %%
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
DOCS_DIR = './ustc-yjs-data'
loader = PyPDFDirectoryLoader(DOCS_DIR)
pages = loader.load_and_split()
pdf_list = os.listdir(DOCS_DIR)

# %%
pages

# %%
from tqdm import tqdm
pdf_text = { pdf_page.metadata['source'][-17:]:'' for pdf_page  in pages }
for pdf in tqdm(pdf_list):
    for pdf_page in pages:
        if pdf in pdf_page.metadata['source']:
            pdf_text[pdf] += pdf_page.page_content
        else:
            continue
print('key:pdf value:text')

# %%
with open('rule_ustc_yjs.txt','w',encoding = 'utf-8') as file:
    pdf_all = ''.join(list(pdf_text.values())).encode('utf-8', 'replace').decode('utf-8')
    file.write( pdf_all)  

# %%
import re

def split_entries(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 清理页码（如"- 6 -"）
    content = re.sub(r'-\s*\d+\s*-', '', content)  # 关键步骤

    # 在读取后先清除章节标题
    content = re.sub(r'^第[一二三四五六七八九十]+章\s+.+\n', '', content, flags=re.MULTILINE)
    content = content.replace('\r\n', '\n').replace('\r', '\n')  # 统一换行符
    
    # 改进正则：匹配"第x条"且后跟多个空格（标题特征）
    pattern = r'第[一二三四五六七八九十零百千万]+条\s+'
    matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
    
    entries = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i < len(matches)-1 else len(content)
        entries.append(content[start:end].strip())  # 去除首尾空白
    
    return entries

# 使用示例
entries = split_entries('rule_ustc_yjs.txt')

# %%
entries

# %%
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch

DEVICE = "cuda:6"

class FaissRetriever(object):
    def __init__(self, data):
        # 使用固定的 bge-large-zh-v1.5 模型，不允许微调
        self.embeddings = HuggingFaceEmbeddings(
            model_name="/data/DeHors_yh_0329/code/RAG/DS2024-BDCI-RAG/src/BAAI/bge-large-zh-v1.5",  # 指定模型为 bge-large-zh-v1.5
            model_kwargs={"device": DEVICE}       # 使用 GPU 进行推理
        )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))

        # 使用 FAISS 进行向量存储
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def GetTopK(self, query, k):
        # 基于查询找到最相似的前 k 个文档
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    def GetvectorStore(self):
        return self.vector_store


# %%
faissretriever = FaissRetriever(entries)
vector_store = faissretriever.vector_store

# %%
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

import jieba

class BM25(object):

    def __init__(self, documents):

        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if(len(line)<5):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            # docs.append(Document(page_content=tokens, metadata={"id": idx, "cate":words[1],"pageid":words[2]}))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            # full_docs.append(Document(page_content=words[0], metadata={"id": idx, "cate":words[1], "pageid":words[2]}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    # 初始化BM25的知识库
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans


# %%
bm25 = BM25(entries)

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class reRankLLM(object):
    def __init__(self, model_path="/data/DeHors_yh_0329/code/RAG/DS2024-BDCI-RAG/src/BAAI/bge-reranker-large", max_length=512):
        # 使用指定的模型路径 BAAI/bge-reranker-large
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.to(DEVICE)  # 将模型移动到指定的设备
        self.max_length = max_length

    def predict(self, query, docs):
        # 创建 query 和文档内容对
        pairs = [(query, doc.page_content) for doc in docs]
        # Tokenize 输入内容，并将其移动到 GPU
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to(DEVICE)
        
        with torch.no_grad():
            # 使用模型进行推理并计算得分
            scores = self.model(**inputs).logits
        
        # 将得分从 GPU 移动到 CPU
        scores = scores.detach().cpu().clone().numpy()
        # 根据得分对文档排序
        scored_docs = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])
        response = [doc for score, doc in scored_docs]

        # 打印查询和对应的前三个文档及其分数
        print(f"Query: {query}")
        for i, (score, doc) in enumerate(scored_docs[:3]):
            print(f"Rank {i + 1}: Score: {score[0]}, Document: {doc.page_content[:50]}...")  # 仅打印文档的前0个字符
        
        # torch_gc()  # 清理缓存
        return response


# %%
rerank = reRankLLM()

# %%
def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    emb_ans = ""
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

# %%
import pandas as pd
QA_df = pd.read_csv('./ques_ans.csv', sep=',')

# %%
import pandas as pd

# 假设 submit_df 已经在之前的代码中定义并包含问题
# 提取问题列表
queries = QA_df['问'].tolist()
answers = QA_df['答'].tolist()

# 初始化一个字典来存储每个列的数据
data = {
    'query': [],  # 添加查询字段
    'answer': [],  # 添加答案字段
    'faiss_context': [],  # 保存 faiss 检索结果
    'bm25_context': [],  # 保存 bm25 检索结果
    'emb_bm25_merge_inputs': []  # 合并后的输入
}

max_length = 4000

# 遍历每个查询
for idx, query in enumerate(queries):
    # faiss 检索
    faiss_context = faissretriever.GetTopK(query, 15)
    faiss_min_score = 0.0
    
    if len(faiss_context) > 0:
        faiss_min_score = faiss_context[0][1]
    
    cnt = 0
    emb_ans = ""
    
    for doc, score in faiss_context:
        cnt += 1
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans += doc.page_content
        if cnt > 6:
            break

    # bm25 检索
    bm25_context = bm25.GetBM25TopK(query, 15)
    bm25_ans = ""
    cnt = 0
    
    for doc in bm25_context:
        cnt += 1
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans += doc.page_content
        if cnt > 6:
            break

    # 合并 faiss 和 bm25 的输入
    rerank_ans = reRank(rerank, 1, query, bm25_context, faiss_context)
    
    # 将结果添加到数据字典中
    data['query'].append(query)  # 保存查询
    data['answer'].append(answers[idx])  # 保存答案
    data['faiss_context'].append(emb_ans)  # 保存 faiss 检索结果
    data['bm25_context'].append(bm25_ans)  # 保存 bm25 检索结果
    data['emb_bm25_merge_inputs'].append(rerank_ans)  # 保存合并结果

# 将字典转换为 DataFrame
batch_input_df = pd.DataFrame(data)


# %%
batch_input_df

# %%
from langchain_community.llms import SparkLLM
# 通过api调用大模型
llm = SparkLLM(
    spark_app_id="0554a658",
    spark_api_key="783779f9bf56e3f8529f424dd178a4c3",
    spark_api_secret="MmJmZGI1OGQ0MDM4YmYyNzFlODFhODg5",
    spark_api_url="wss://spark-api.xf-yun.com/v1.1/chat",
    model="lite"
)

# %%
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
augmented_prompt = """[角色]
你是一个严谨的问答专家，严格依据提供的知识库内容进行回答

[指令]
请按以下步骤处理：
1. 上下文核查：分析<上下文>与<问题>的关联性
2. 知识提取：仅使用提供的上下文信息，禁止外部知识
3. 回答组织：
   - 若上下文包含明确答案：可以直接使用原文，然后剩余内容根据相关性进行补充。
   - 若上下文不相关/不充分：明确告知无法回答
   - 始终在结尾使用：期待为您提供更多帮助

[输出格式要求]
回答必须包含：
- 首段：直接的问题回答（如原文内容），如果是列点回答，则要完整列出所有点（子项）
- 末段：剩余内容根据相关性进行补充

### 上下文 ###
{source_knowledge}

### 问题 ###
{query}
有用的回答:"""

prompt = PromptTemplate(template=augmented_prompt, input_variables=["source_knowledge" ,"query"])
llm_chain = LLMChain(prompt=prompt, llm=llm  , llm_kwargs = {"temperature":0, "max_length":1024})

# %%
# 假设 batch_input_df 已经在之前的代码中定义并包含 question 和 emb_bm25_merge_inputs
# 初始化一个空列表来存储 LLM 的回答
llm_answers = []

# 遍历每个查询和对应的源知识
for idx, row in batch_input_df.iterrows():
    query = row['query']
    source_knowledge = row['emb_bm25_merge_inputs']
    
    # 使用 SparkLLM 生成回答
    llm_answer = llm_chain.run( {"source_knowledge":source_knowledge ,"query" : query })
    
    # 将生成的回答添加到列表中
    llm_answers.append(llm_answer)

# 将生成的回答添加到 batch_input_df 中的新列 LLM_answer
batch_input_df['LLM_answer'] = llm_answers

# %%
selected_columns = batch_input_df[['query', 'answer', 'LLM_answer']]

# %%
selected_columns

# %%
import datetime

# 可以按日期-时间保存，例如 "2024-10-01 21:28" 记为 "1001_2128"
today = datetime.datetime.now().strftime("%m%d_%H%M")
savepath = f'./rag_output_{today}.csv'
selected_columns.to_csv(savepath, sep=',', index=False)
print(f"Saved path: {savepath}")

# %% [markdown]
# # 评测指标
# 在一般论文中，采用的一般是英文数据集，以多跳问答数据集为主，如HotpotQA。其中一般采用EM，F1，Accuracy等指标进行评测。
# 下面是相关代码实现：
# 
# ```python   
# def normalize_answer(s):
#     """
#     Normalizes the answer string.
# 
#     This function standardizes the answer string through a series of steps including removing articles,
#     fixing whitespace, removing punctuation, and converting text to lowercase. This ensures consistency
#     and fairness when comparing answers.
# 
#     Parameters:
#     s (str): The answer string to be standardized.
# 
#     Returns:
#     str: The standardized answer string.
#     """
# 
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)
# 
#     def white_space_fix(text):
#         return " ".join(text.split())
# 
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)
# 
#     def lower(text):
#         return str(text).lower()
# 
#     return white_space_fix(remove_articles(remove_punc(lower(s))))
# 
# 
# def f1_score(prediction, ground_truth):
#     """
#     Calculates the F1 score between the predicted answer and the ground truth.
# 
#     The F1 score is the harmonic mean of precision and recall, used to evaluate the model's performance in question answering tasks.
# 
#     Parameters:
#     prediction (str): The predicted answer from the model.
#     ground_truth (str): The actual ground truth answer.
# 
#     Returns:
#     tuple: A tuple containing the F1 score, precision, and recall.
#     """
# 
#     normalized_prediction = normalize_answer(prediction)
#     normalized_ground_truth = normalize_answer(ground_truth)
# 
#     ZERO_METRIC = (0, 0, 0)
# 
#     if (
#         normalized_prediction in ["yes", "no", "noanswer"]
#         and normalized_prediction != normalized_ground_truth
#     ):
#         return ZERO_METRIC
# 
#     if (
#         normalized_ground_truth in ["yes", "no", "noanswer"]
#         and normalized_prediction != normalized_ground_truth
#     ):
#         return ZERO_METRIC
# 
#     prediction_tokens = normalized_prediction.split()
#     ground_truth_tokens = normalized_ground_truth.split()
# 
#     # Calculate the number of matching words between the predicted and ground truth answers
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
# 
#     if num_same == 0:
#         return ZERO_METRIC
# 
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
# 
#     return f1, precision, recall
# 
# 
# def exact_match_score(prediction, ground_truth):
#     """
#     Calculates the exact match score between a predicted answer and the ground truth answer.
# 
#     This function normalizes both the predicted answer and the ground truth answer before comparing them.
#     Normalization is performed to ensure that non-essential differences such as spaces and case are ignored.
# 
#     Parameters:
#     prediction (str): The predicted answer string.
#     ground_truth (str): The ground truth answer string.
# 
#     Returns:
#     int: 1 if the predicted answer exactly matches the ground truth answer, otherwise 0.
#     """
# 
#     return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0
# ```
# 
# 其中首先通过标准化函数：
# - 去除冠词：使用正则表达式去除 "a"、"an" 和 "the"。
# - 修复空格：将多余的空格替换为单个空格。
# - 去除标点：移除所有标点符号。
# - 转换为小写：将文本转换为小写字母
# 
# 然后EM：如果它们完全相等，返回 1；否则返回 0
# 
# F1 分数、精确度和召回率：
# - 检查特殊情况（如 "yes"、"no" 和 "noanswer"），如果预测和真实答案不匹配，则返回零分。
# - 将标准化后的答案分割为单词，计算预测和真实答案之间的共同单词数量。
# - 计算精确度和召回率，并基于这两个值计算 F1 分数。

# %%
# 导入csv文件
import pandas as pd

data = pd.read_csv('./rag_output_0322_1535.csv')


# %%
import re
import jieba
import string
from collections import Counter

def normalize_answer_chinese(s):
    """
    中文标准化处理：去标点、修复空格、全角转半角、小写化（可选）
    """
    def remove_punc(text):
         # 中英文标点符号（可根据需求扩展）
        chinese_punctuation = '，。！？；：“”‘’（）【】《》…—·'
        all_punctuation = set(string.punctuation + chinese_punctuation)
        return ''.join(ch for ch in text if ch not in all_punctuation)

    def full_to_half(text):
        # 全角转半角（处理数字和英文字符）
        return text.translate(str.maketrans('１２３４５６７８９０ＡＢＣＤＥＦＧ',
                                          '1234567890ABCDEFH'))

    def white_space_fix(text):
        return " ".join(text.split())

    processed = full_to_half(s)          # 全角转半角
    processed = remove_punc(processed)   # 去标点
    processed = white_space_fix(processed)  # 修复空格
    return processed

def f1_score_chinese(prediction, ground_truth):
    """
    中文F1计算：基于分词后的词粒度匹配
    """
    normalized_pred = normalize_answer_chinese(prediction)
    normalized_gt = normalize_answer_chinese(ground_truth)

    # 处理特殊词（如“是/否/无答案”）
    SPECIAL_TOKENS = ["是", "否", "无答案"]
    if (normalized_pred in SPECIAL_TOKENS or normalized_gt in SPECIAL_TOKENS) \
        and normalized_pred != normalized_gt:
        return (0, 0, 0)

    # 中文分词
    pred_tokens = list(jieba.cut(normalized_pred, cut_all=False))
    gt_tokens = list(jieba.cut(normalized_gt, cut_all=False))

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return (0, 0, 0)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall + 1e-10)  # 避免除零

    return (f1, precision, recall)

def exact_match_chinese(prediction, ground_truth):
    """
    中文精确匹配：标准化后完全一致返回1，否则0
    """
    return 1 if normalize_answer_chinese(prediction) == normalize_answer_chinese(ground_truth) else 0

# %%
# def lenient_accuracy(examples, recall_threshold=0.8):
#     """
#     基于Recall的宽松准确率，研究生学籍管理问答系统不像医疗法律需要严格的精确度，提供的信息尽可能多包含更加适合的答案，因此使用Recall作为准确率的指标。
#     """
#     total = len(examples)
#     correct = 0
#     for _, row in examples.iterrows():
#         _, _, recall = f1_score_chinese(row['LLM_answer'], row['answer'])
#         if recall >= recall_threshold:
#             correct += 1
#     return correct / total

# %%
total_metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

for idx, row in data.iterrows():
    answer = row['answer']
    llm_answer = row['LLM_answer']

    em = float(exact_match_chinese(llm_answer, answer))
    f1, precision, recall = f1_score_chinese(llm_answer, answer)

    total_metrics["em"] += em
    total_metrics["f1"] += f1
    total_metrics["precision"] += precision
    total_metrics["recall"] += recall
    print(f"EM: {em}, F1: {f1}, Precision: {precision}, Recall: {recall}")

total_metrics = {k: v / len(data) for k, v in total_metrics.items()}
print(total_metrics)

# %% [markdown]
# 研究生学籍管理问答系统不像医疗法律需要严格的精确度，答案相对来说较长，LLM提供的信息尽可能多包含更加正确答案即可，因此使用Recall作为指标。
# 


