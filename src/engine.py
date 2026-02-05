from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGEngine:
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0)

    def get_chain(self, retriever):
        system_prompt = (
            "Jesteś ekspertem. Odpowiadaj w oparciu o kontekst: {context}. "
            "Jeśli nie wiesz, odpowiedz: 'Informacja niedostępna w dokumentach'. "
            "Na końcu zawsze podaj źródło i stronę."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, doc_chain)