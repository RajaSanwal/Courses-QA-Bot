from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

GOOGLE_API_KEY = "AIzaSyB3N2Q_HOYCILLbJYidWnCMI1QD10OJyKs"

llm = GooglePalm(google_api_key=GOOGLE_API_KEY, temperature=0.1)

gpt4all_embd = GPT4AllEmbeddings()

vector_db_path = "FAISS_INDEX"

def create_vector_db():
    
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column='prompt')
    data = loader.load()
    vector_db = FAISS.from_documents(documents=data, embedding=gpt4all_embd)
    vector_db.save_local(vector_db_path)
    
def get_qa_chain():
    vectordb = FAISS.load_local(vector_db_path, gpt4all_embd)
    retriever = vectordb.as_retriever(score_thershold=0.7)
   

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate( template= prompt_template, input_variables=['context','question'])
    
    

    chain = RetrievalQA.from_chain_type(llm=llm,
                           chain_type='stuff',
                           retriever=retriever,
                           input_key='query',
                           return_source_documents=True,
                            chain_type_kwargs={"prompt":PROMPT})
    
    return chain

if __name__=="__main__":
    
    chain = get_qa_chain()
    
    print(chain("Do you have a javascript course?"))
