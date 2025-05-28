本项目是ustc的nlp课程的一个rag项目demo。其中还包含使用规则化的简单rag处理，见`rule_ustc_yjs_rag.ipynb`

rag的demo运行流程：

创建数据库

将文件放置在 data_base/knowledge_db 目录下

然后使用`create_vetcordb.py`创建向量数据库

运行项目
```shell
cd serve
python run_gradio.py -model_name='chatglm_std' -embedding_model='bge'
```

感谢 [动手学大模型应用开发](https://github.com/datawhalechina/llm-universe/tree/main?tab=readme-ov-file) 的教学指导