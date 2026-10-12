import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_classic.chains import create_retrieval_chain
#from langchain.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "wikipedia_rag"
EMBEDDING_MODEL = "granite-embedding:278m"
LLM_MODEL = "qwen2.5:0.5b"
#OLLAMA_URL = "http://localhost:11434"
SOURCES_FILE_PATH = "README.md"
QUESTIONS_FILE_PATH = "questions.md"


class LocalRagSystem:
    def __init__(self):
        print("[RAG] Lokális modellek és kliensek inicializálása...")
        
        # 1. Ollama embedding & chat model
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
        
        # 2. Qdrant client & Vector Store
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.__init_qdrant_collection()
        
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings
        )

    def __init_qdrant_collection(self):
        if not self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768, # A granite-embedding:278m modell 768 dimenziós vektorokat gyárt
                    distance=Distance.COSINE
                )
            )
            print(f"[Qdrant] '{COLLECTION_NAME}' kollekció sikeresen létrehozva.")

    def ingest_wikipedia_links(self, urls: list[str]):
        """Letölti a HTML oldalakat, letisztítja a szöveget és beindexeli a Qdrantba."""
        print(f"[Ingestion] {len(urls)} Wikipédia link beolvasása...")
        
        # HTML Loader (BeautifulSoup)
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(id="mw-content-text") # Csak a Wikipédia érdemi szövegtörzse kell
            )
        )
        docs = loader.load()
        
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        print(f"[Ingestion] A szövegek {len(chunks)} darabra (chunk) lettek felvágva.")
        
        # Embedding & save vectors into Qdrant
        print("[Qdrant] Vektorok generálása és mentése az adatbázisba...")
        self.vector_store.add_documents(chunks)
        print("[Rendszer] Sikeres indexelés!")

    def ask_question(self, question: str):
        """Létrehozza a RAG láncot és megválaszolja a kérdést a kontextus alapján."""
        print(f"\n[Keresés] Releváns dokumentumok kigyűjtése a kérdéshez: '{question}'")
        
        # Vector Store -> Retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5}) # A top 5 legközelebbi darabot kéri le
        
        # Prompt tamplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "A következő kikeresett kontextus alapján válaszolj a kérdésre magyarul. "
                       "Ha nem találod a választ a kontextusban, mondd azt, hogy nem tudod, ne találj ki semmit.\n\nKontextus:\n{context}"),
            ("user", "{input}")
        ])
        
        # LangChain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": question})
        
        print("\n============================ LOKÁLIS AI VÁLASZ =============================")
        print(response["answer"])

        # Források
        print("================================ FORRÁSOK ==================================")
        for i, doc in enumerate(response["context"], 1):
            src_loc = doc.metadata.get("source", "Ismeretlen forrás")
            page_num = doc.metadata.get("page", "-")
            print(f"[{i}] {src_loc} (Oldal: {page_num})")
            #print(f"    Idézet: {doc.page_content[:100]}...")
        
        print("----------------------------------------------------------------------------\n")

        return response["answer"]


class ChatRequest(BaseModel):
    question: str

def collect_wikipedia_links(file_path: str) -> list[str]:
    """Wikipédia URL-ek kigyűjtése az md-fájlból."""
    print(f"Wikipédia linkek kigyűjtése a '{file_path}' fájlból...")
    
    wiki_regex = r'https://hu\.wikipedia\.org/wiki/[^\s\)\"\']+'
    # en ??
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    found_links = re.findall(wiki_regex, content)
    
    pure_links = list(set(found_links))
    
    print(f"[Fájlkezelő] Wikipédia linkek száma: {len(pure_links)}")
    for link in pure_links:
        print(f"   -> {link}")
        
    return pure_links

def collect_questions(file_path: str) -> list[str]:
    """'## Kérdések' alatti listaelemek (kérdések) kigyűjtése az md-fájlból."""
    print(f"Kérdések kigyűjtése a '{file_path}' fájlból...")
    
    quests = []
    inside_quests = False
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            pure_line = line.strip()
            
            if pure_line.startswith("## Kérdések:"):
                inside_quests = True
                continue
            
            # Egy másik főszekciónál kikapcsoljuk a gyűjtést
            if inside_quests and pure_line.startswith("##"):
                break

            # A kérdések sorszámozottak
            if inside_quests and re.match(r'^\d+\.\s+', pure_line):
                quest = re.sub(r'^\d+\.\s*', '', pure_line)
                if quest:
                    quests.append(quest)
                    
    print(f"[Fájlkezelő] Talált kérdések száma: {len(quests)}")

    return quests


def main():
    rag = LocalRagSystem()
    
    # Wikipédia források (Magyar Wikipédia oldalak)
    #wiki_urls = collect_wikipedia_links(SOURCES_FILE_PATH)
    
    # 1. Adatok betöltése
    #rag.ingest_wikipedia_links(wiki_urls)

    # 2/a. Kérdések kigyűjtése a fájlból
    questions = collect_questions(QUESTIONS_FILE_PATH)
    for ord, q in enumerate(questions, 1):
        print(f"\n\n================================ {ord}. KÉRDÉS ================================")
        rag.ask_question(q)

    # 2/b. Kérdés CLI-ből
    #while True:
    #    question = input("Írd be a kérdésed (kilépés: x): ")
    #    if question == "x":
    #        break
    #    rag.ask_question(question)

    # 2/c. FastAPI /api/chat endpointtal
    #app = FastAPI()
    #
    #@app.post("/api/chat")
    #def chat(request: ChatRequest):
    #    answer = rag.ask_question(request.question)
    #    return {"answer": answer}
    #
    #uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
