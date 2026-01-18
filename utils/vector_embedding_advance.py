# import dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import List
import openai
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pathlib import Path
import pyprojroot
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# create classs

class VectorLoader:
    def __init__(self, 
                 data_directory,
                 chunk_size:int = 3000, # Token size for parent chunk
                 chunk_overlap:int = 500,
                 embedding_model:str = "sentence-transformers/all-MiniLM-L6-v2",
                 k_number = 20,
                 embed_weight= 0.5,
                 bm25_weight = 0.5
                 ) -> None:
        self.data_directory = data_directory

        self.child_splitter =  RecursiveCharacterTextSplitter(
             chunk_size = 300, # Token size for child chunk
             chunk_overlap = 50,
             separators =["\n\n","\n"," ",""]     
            )
        self.parent_splitter = RecursiveCharacterTextSplitter(
             chunk_size = chunk_size,
             chunk_overlap = chunk_overlap,
             separators =["\n\n","\n"," ",""] )
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.k_number = k_number
        self.embed_weight = embed_weight
        self.bm25_weight = bm25_weight
    
    # load data
    def load_pdf(self) -> List:
                file_num = 0
                pdf_meta = []
                if isinstance(self.data_directory, list):
                    for i in self.data_directory:
                        pdf_meta.extend(PyPDFLoader(i).load())
                        file_num += 1
                else:
                    paths = os.listdir(self.data_directory)
                    for i in paths:
                        if i.endswith(".pdf"):
                            pdf_meta.extend(PyPDFLoader(os.path.join(self.data_directory, i)).load())
                            file_num += 1
                print(f"Loaded {file_num} files")    
                return  pdf_meta
    

    # chunk data # add headers
    def _chunk_documents(self,docs:List) -> List:
            childchunks =[]
            print("Chucking Documents")
            parentchunks = self.parent_splitter.split_documents(docs)
            for parent in parentchunks:
                parent.metadata["parent_id"] = hash(parent.page_content[0:100])
                header = f"[SOURCE: {parent.metadata.get("title","unknown")}| PAGE:{parent.metadata.get("page","unknown")}]"
                parent.page_content = header +"\n" + parent.page_content
                children_chunk = self.child_splitter.split_documents([parent])
                for child in children_chunk:
                    child.metadata["parent_id"] = parent.metadata["parent_id"]
                childchunks.extend(children_chunk)

            print("Chucking Complete")
            print(rf"Number of Children Chunks: {len(childchunks)}")
            return parentchunks, childchunks

    # Creating Vector embedding file

    def persist_direct(self) -> str:
         root = pyprojroot.here()
         print(root)
         base_path = [ self.data_directory[0] if  isinstance (self.data_directory, list) else self.data_directory]
         base = Path(base_path[0]).name
         persist_directory = root/"VectorDB"/base
         return str(persist_directory)
    
    def get_bm25_retriever(self,docs):
        return BM25Retriever.from_documents(docs)
    
    def embedding_documents(self):
        pd = self.persist_direct()
        docum = self._chunk_documents(self.load_pdf())
        document_child= docum[1]
        if not os.path.exists(pd):
            print("Embedding Documents")
            vectordb = Chroma.from_documents(
                documents= document_child,
                embedding=self.embedding_model,
                persist_directory=pd
                )
            print("Embedding Complete")
            print("Number of vectors in vectordb:",
                vectordb._collection.count(), "\n\n")
        else:
            print("Embedding Documents Available")
            vectordb = Chroma(
                persist_directory=pd,
                embedding_function=self.embedding_model  
            )

        bm25_retriever = BM25Retriever.from_documents(document_child)
        bm25_retriever.k = self.k_number
        vector_retriever = vectordb.as_retriever(search_kwargs={"k":self.k_number})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]  # weight BM25 vs embeddings
        )

        return {
             'vectordb' : vectordb,
             "retriever": ensemble_retriever,
             "parent" : docum[0]
        }
    
         
    
    

  
