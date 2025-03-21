import streamlit as st
import requests
import logging
import os
import json  # Import the json module

def safe_json_response(response):
    """Safely parse JSON response and handle errors."""
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {
            "error": "Invalid JSON response received from backend.",
            "status_code": response.status_code,
            "text": response.text[:500],  
        }

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000" 

st.title("📄 AI Document Processing - RAG Pipeline")
tab_upload, tab_chat = st.tabs(["Upload & Process", "Chat"])

# File Upload Section
with tab_upload:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        parser_option = st.selectbox("Select Parsing Method", ["pymupdf", "docling", "mistral_ocr"], key="parser_selectbox")
        

        if st.button("Upload and Process PDF"):
            if uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"parser": parser_option}
                response = requests.post(f"{FASTAPI_URL}/process_pdf", files=files, data=data)
                try:
                    response_data = response.json()
                except requests.exceptions.JSONDecodeError:
                    st.error(f"Error: Unable to parse JSON. Response: {response.text}")
                else:
                    if response.status_code == 200:
                        st.success("File uploaded and processed successfully!")
                        # Store filename in session state after successful upload
                        st.session_state.uploaded_filename = response_data["filename"]
                    else:
                        st.error(f"Error: {response_data.get('detail', 'Unknown error')}")
            else:
                st.warning("Please upload a file first.")

        # Maintain file selection across interactions
        if "files" not in st.session_state:
            st.session_state.files = []
        if "selected_file" not in st.session_state:
            st.session_state.selected_file = None

        # List Available Files Section
        st.header("📂 Available Processed Files")
        if st.button("Refresh File List"):
            response = requests.get(f"{FASTAPI_URL}/files")
            if response.status_code == 200:
                st.session_state.files = response.json()["files"]
                if st.session_state.files:
                    # Unique key for the selectbox displaying files
                    st.session_state.selected_file = st.selectbox("Select a file to process", st.session_state.files, key="file_selectbox_upload", index=0)
                else:
                    st.warning("No processed files available.")
            else:
                st.error("Failed to fetch files.")

        # Processing Section
        st.header("🔍 Process File for RAG")
        chunking_strategy = st.selectbox("Select Chunking Strategy", ["Recursive", "Token", "Semantics"], key="chunking_selectbox")
        vector_db = st.selectbox("Select Vector Database", ["chromadb", "pinecone"], key="vector_db_selectbox")
        quarter = st.selectbox("Select Quarter", ["first", "second", "third", "fourth"], key="quarter_selectbox") # ADDED QUARTER

        if st.session_state.files:
            # Unique key for the selectbox displaying files during processing
            st.session_state.selected_file = st.selectbox("Select a file to process", st.session_state.files, key="file_selectbox_process", index=st.session_state.files.index(st.session_state.selected_file) if st.session_state.selected_file in st.session_state.files else 0)

        if st.button("Process File"):
            #Check if file has been uploaded
            if not hasattr(st.session_state, "uploaded_filename"):
                st.warning("Please upload a file first.")
            elif st.session_state.selected_file:
                # Create the payload for /process_chunks
                payload = {
                    "filename": st.session_state.selected_file,  # Send filename, not file
                    "chunking_strategy": chunking_strategy,
                    "vector_db": vector_db,
                    "quarter": quarter  # Include the selected quarter
                }

                # Send the payload as JSON
                headers = {'Content-Type': 'application/json'}  # Set the content type
                response = requests.post(f"{FASTAPI_URL}/process_chunks", data=json.dumps(payload), headers=headers)

                response_data = safe_json_response(response)
                if "error" in response_data:
                    st.error(f"Error: {response_data['error']} (Status Code: {response_data['status_code']})")
                    st.write("Response Text:", response_data["text"])
                else:
                    st.success("File processed and embeddings stored successfully!")
            else:
                st.warning("Please select a file first.")
    
with tab_chat:    
    st.header("Ask a Question")

    # User inputs
    question = st.text_area("Enter your question:", key="question_textarea")
    
    # Quarter Selection for Filtering
    selected_quarters = st.multiselect(
        "Select the quarter(s) data to be used to answer the query:",
        ["first", "second", "third", "fourth"],
        key="query_quarters_selectbox"
    )
    
    # Ensure at least one quarter is selected
    if not selected_quarters:
        st.warning("⚠️ Please select at least one quarter.")
    
    # Button to trigger the query
    if st.button("Get Answer"):
        if question.strip() and selected_quarters:
            with st.spinner("🔎 Searching for relevant documents..."):
                # Create the payload with the question, vector_db, and selected_quarters
                payload = {
                    "question": question,
                    "vector_db": "pinecone",  
                    "quarters": selected_quarters
                }

                # Make sure you're sending the payload as JSON
                response = requests.post(f"{FASTAPI_URL}/query", json=payload)

                # Handle the response
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Answer Retrieved")

                    # Display the generated answer
                    st.subheader("💡 Answer:")
                    st.write(result["answer"])

                    # Display relevant document context
                    st.subheader("📄 Retrieved Context:")
                    for i, chunk in enumerate(result["context"], start=1):
                        with st.expander(f"Context {i}"):
                            st.write(chunk)

                else:
                    st.error("⚠️ Error fetching answer")
                    st.write(response.json().get("detail", "Unknown error"))
        else:
            st.warning("⚠️ Please enter a question and select at least one quarter before submitting.")
