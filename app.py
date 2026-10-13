import streamlit as st
from rag import ask_rag

# Streamlit App Title
st.title(" E-Commerce and Computer Graphics Subject Knowledge Assistant (RAG + Gemini)")

# User Input
question = st.text_input("Ask a question:")

if question:
    with st.spinner("Searching and generating answer..."):
        try:
            answer, results = ask_rag(question)

            # Show Answer
            st.subheader("Answer")
            st.write(answer)

            # Show Sources
            st.subheader("Sources")
            for i, result in enumerate(results, start=1):
                with st.expander(f"Source {i} (Distance: {result['distance']:.2f})"):
                    st.write(result["text"])

        except Exception as error:
            st.error(f"RAG Error: {error}")
