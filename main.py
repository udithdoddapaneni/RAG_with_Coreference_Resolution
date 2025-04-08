import streamlit as st
from RAG.rag import graph_with_retriever, graph_direct, UpdateDatabase

def main():
    st.set_page_config(page_title="Marci", layout="wide")
    st.title("Marci: A Contextual AI Assistant")

    # Option selector
    mode = st.selectbox("Choose mode:", ["llm", "llm with retriever"])

    # Query Input Section
    query = st.text_input("Enter your query:", "")
    
    if st.button("Submit"):
        if query:
            with st.spinner("Processing your query..."):
                if mode == "llm":
                    response = graph_direct.invoke({"query": query, "context": "", "answer": ""})
                else:  # llm with retriever
                    response = graph_with_retriever.invoke({"query": query, "context": "", "answer": ""})
            
            st.subheader("Response")
            st.write(response["answer"])
        else:
            st.warning("Please enter a query.")

    st.markdown("---")  # Adds a separator line

    # Database Update Section
    st.subheader("Database Management")

    resolve_refs = st.checkbox("Resolve references before updating", value=True)

    if st.button("Update Database"):
        with st.spinner("Updating database... This may take a while ⏳"):
            update_status = UpdateDatabase(resolve_refs=resolve_refs)
        
        if "success" in update_status.lower():
            st.success(update_status)
        else:
            st.error(update_status)

if __name__ == "__main__":
    main()
