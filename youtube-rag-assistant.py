from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

## Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_data.snippets)

except TranscriptsDisabled:
    raise RuntimeError("Transcript not available for this video")

## Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = embedding
vector_store = FAISS.from_documents(chunks, embeddings)

## Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

## Step 3 - Augmentation
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", temperature=0.2)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

parser = StrOutputParser()

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})

main_chain = parallel_chain | prompt | model | parser

print(main_chain.invoke("is the topic of nuclear fusion discussed in this video? if yes then what was discussed"))  
