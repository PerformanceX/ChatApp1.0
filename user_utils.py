from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    PineconeClient(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )
    index_name = pinecone_index_name
    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
    return embeddings

def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(docs, user_input):
    chain = load_qa_chain(ChatOpenAI(model="gpt-4o"), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response

def get_llm_answer(user_input):
    llm = ChatOpenAI(model="gpt-4o",temperature=0.7)

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are PerformanceX, a personalized AI coach. You specialize in mental wellness,injury prevention,preparation techniques and performance optimization. Your are 24/7 personal coach and motivator. Your knowledge spans various topics including sports, fitness, nutrition, and general knowledge. 
        Answer the following question to the best of your ability: {question}
        If the question is not directly related to performance coaching, still provide a helpful and informative answer."""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=user_input)
    return response

def is_relevant(doc, query, threshold=0.5):
    embeddings = create_embeddings()
    doc_embedding = embeddings.embed_query(doc.page_content)
    query_embedding = embeddings.embed_query(query)
    similarity = cosine_similarity([doc_embedding], [query_embedding])[0][0]
    return similarity > threshold