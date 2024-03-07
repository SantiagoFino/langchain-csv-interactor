from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

csv_path = r"/mnt/docs/personal/langchain-tutorial/data/forecasting.csv"



class CsvInteractor:
    def __init__(self, csv_path) -> None:
        self.docs = CSVLoader(file_path=csv_path).load()

        self.model = Ollama(model="llama2")
        self.embeddings = OllamaEmbeddings()

        self.vectors = FAISS.from_documents(self.docs,
                                            self.embeddings)

        self.doc_chain = None
        self.retrieval_chain = None

    def create_doc_chain(self, doc_prompt):
        self.doc_chain = create_stuff_documents_chain(llm=self.model,
                                                      prompt=doc_prompt)
        return self.doc_chain
    
    def create_retrieval_chain(self):
        retriever = self.vectors.as_retriever()
        self.retrieval_chain = create_retrieval_chain(retriever=retriever,
                                                      combine_docs_chain=self.doc_chain)
        return self.retrieval_chain
    
    def conversational_chain(self, input, chat_history):
        conversation = ConversationChain(llm=self.model,
                                         verbose=True,
                                         memory=ConversationBufferMemory())