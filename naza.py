import os
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA


def initialize_rag_chain():
    """Initializes the RAG chain with Pinecone and Gemini."""
    load_dotenv()

    google_key = os.getenv("GOOGLE_API_KEY")
    pkey = os.getenv("PINECONE_API_KEY")
    # pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

    index_name = os.getenv('PINECONE_INDEX')  # Replace with your Pinecone Index Name
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=google_key
    )
    client = Pinecone(api_key=pkey)
    index = client.Index(name='naza', host=index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, pinecone_api_key=pkey)

    print("initialized vector store")
    model = ChatGoogleGenerativeAI(api_key=google_key, model="gemini-2.0-flash-exp")
    print("model/LLM online")

    qa = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    return qa


def main():
    """Main function to run the Streamlit app."""

    st.title("naza-codex")

    # Initialize the RAG chain
    try:
        qa_chain = initialize_rag_chain()
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask me anything about the documents!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the RAG chain
        try:
            response = qa_chain.invoke(prompt)
            print(f'response / \n {response}')
            answer = response["result"]
            # sources = response["source_documents"]  # Access source documents
            # print(f"Response with sources: {response}")  # Debugging
            # source_str = "\n\n".join(
            #     [str(doc.metadata) for doc in sources]
            # )  # Extracting source metadata

        except Exception as e:
            answer = f"An error occurred: {e}"
            # source_str = "No sources found due to error."

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Display assistant message in chat
        with st.chat_message("assistant"):
            st.markdown(answer)
        # if sources:
        #     with st.expander("Sources"):
        #         st.write(source_str)
        # else:
        #     st.write("No sources found.")


main()
