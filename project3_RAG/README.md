# 1. SimpleRAG

This project is for the study of how to update LLM with the latest content from Wikipedia or Research Paper.

**features**

You can use langchain LLM models with ChatOpenAI and OpenAIEmbeddings APIs 

You can select unknown dataset from wikipedia or a pdf file (research paper).

init prompt is agumented with additional data.

[RetrievalQA APIs](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#)

**set up environment**
1. create a PyCharm Project with venv based on python3.12
2. install libs
```
pip install streamlit --> frontend libs; streamlit API(https://docs.streamlit.io/)
pip install python-dotenv, langchain, openai, langchain_community, langchain_openai, docarray
```
3. create OPENAI_API_KEY in .env file
4. running app
```
streamlit run app.py
```

Reference: [RAG-Simplified](https://github.com/ShahMitul-GenAI/RAG-Simplified/tree/main)

# 2. Course - RAG from Scratch 

Reference: [Youtube-freeCodeCamp](https://youtu.be/sVcwVQRHIc8?si=H8nq24PCdlgISIjS)

# 3. My Project
