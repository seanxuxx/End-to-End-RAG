# ============================================================================ #
# https://github.com/zixinwen98/11711-RAG
# ============================================================================ #

PROMPT_DICT = {
    "phi-2": ("Background Information: {context}\nInstruct: {question}\n Output:"),
    "alpaca": (
        "Below is an question, paired with an document to supply important information. "
        "Write an response that appropriately answers the question.\n\n"
        "### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:"
    ),
    "microsoft/phi-2": "Background Information: {context}\nInstruct: {question}\n Output:",
    "mistralai/Mistral-7B-Instruct-v0.2": "<s>[INST]Please answer a question by information in context. Below is the context and the question.\n Context: {context}\nQuestion: {question}\n Output: [/INST]",
    "google/gemma-7b-it": "{context}\nGiven the context above, answer the following question: {question}\n Below is my answer: ",

}

qa_model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
max_length = 2048

doc_encoder_model_name_or_path = "avsolatorio/GIST-large-Embedding-v0"


# ============================================================================ #
# https://github.com/akshay140601/End-to-End-NLP-System
# ============================================================================ #

embedding_model = "BAAI/bge-large-en-v1.5"
qa_model = "togethercomputer/llama-2-70b-chat"

reranker = FlagEmbedding.FlagReranker('BAAI/bge-reranker-large', use_fp16=True)


# ============================================================================ #
# https://github.com/chaosarium/CMU-LTI-RAG
# ============================================================================ #

generator_models = ["mixtral8x7b", "gemma7b"]


# ============================================================================ #
# https://github.com/Arlene036/RAGinBurgh
# ============================================================================ #

embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
generator = "SciPhi/SciPhi-Self-RAG-Mistral-7B-32k"

RAG_PROMPT = """Your task is to answer a question based on the realted documents and your own knowledge.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Only output the answer related to the question.
Use three sentences maximum and keep the answer as concise as possible.

Related Documents:
{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT_FEW_SHOT = """Your task is to answer a question based on the realted documents and your own knowledge.
If you believe there are no related dodcuments to the question, answer the question based on your own knowledge base.
Only output the answer directly related to the question, like a phrase or a sentence.
Keep the answer as concise as possible.

Related Documents:
# event_name
Yalda Night
# event_time
Dec 20,2024|All Day

Question: When is Yalda Night held?

Helpful Answer: Dec 20, 2024.

Related Documents:
Governor Dinwiddie sent Captain William Trent to build a fort at the Forks of the Ohio On February 17 1754 Trent began construction of the fort the first European habitation17 at the site of presentday Pittsburgh The fort named Fort Prince George was only halfbuilt by April 1754 when over 500 French forces arrived and ordered the 40some colonials back to Virginia The French tore down the British fortification and constructed Fort Duquesne1415

Question: What was the first European-built fort at the site of present-day Pittsburgh?

Helpful Answer: Fort Prince George

Related Documents:
{context}

Question: {question}\n\nHelpful Answer: """
