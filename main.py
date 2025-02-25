import streamlit as st
from RAG.rag import graph, UpdateDatabase

def main():
    st.set_page_config(page_title="RAG Application", layout="wide")
    st.title("🧠 RAG-Powered QA System")

    # Query Input Section
    query = st.text_input("Enter your query:", "")
    
    if st.button("Submit"):
        if query:
            with st.spinner("Processing your query..."):
                response = graph.invoke({"query": query, "context":"", "answer":""})
            
            st.subheader("Response")
            st.write(response["answer"])
        else:
            st.warning("Please enter a query.")

    st.markdown("---")  # Adds a separator line

    # Database Update Section
    st.subheader("Database Management")
    if st.button("Update Database"):
        with st.spinner("Updating database... This may take a while ⏳"):
            update_status = UpdateDatabase()
        
        if "success" in update_status.lower():
            st.success(update_status)
        else:
            st.error(update_status)


if __name__ == "__main__":
    main()