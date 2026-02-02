"""
YouTube RAG Assistant
"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

PERSIST_DIR = "faiss_index"

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

if __name__ == "__main__":

    video_id = input("Enter YouTube video ID: ").strip()

    try:
        transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_data.snippets)
    except TranscriptsDisabled:
        raise RuntimeError("Transcript not available for this video")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        doc.metadata["video_id"] = video_id

    if os.path.exists(PERSIST_DIR):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            PERSIST_DIR,
            embedding,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(chunks, embedding)
        vector_store.save_local(PERSIST_DIR)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20
        }
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.2
    )

    model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"]
    )

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    question = "What challenges are discussed in this video?"

    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    print("Answer:\n")
    print(answer)

    print("\nSources:\n")
    for doc in retrieved_docs:
        print(
            f"- Video {doc.metadata['video_id']} | Chunk {doc.metadata['chunk_id']}"
        )

