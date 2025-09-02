# Chatbot

# 📄 PDF Q&A Chatbot

### Project Description

This project is a conversational AI application that allows users to ask questions about the content of any PDF document. It leverages a modern approach called **Retrieval-Augmented Generation (RAG)** to provide factual and accurate answers. The entire application, from the file upload to the chat interface, is built using **Streamlit**.

### Key Features

- **Document-Grounded Answers:** Provides answers that are strictly based on the content of the uploaded PDF, minimizing the risk of "hallucinations."
- **Retrieval-Augmented Generation (RAG):** Integrates an efficient RAG pipeline to retrieve the most relevant information from the document before generating a response.
- **Conversational Memory:** Remembers the context of previous questions to enable natural, multi-turn conversations.
- **User-Friendly Interface:** A simple and intuitive web interface powered by Streamlit for easy file uploads and real-time interaction.

### How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/priyagarg-01/Chatbot.git](https://github.com/priyagarg-01/Chatbot.git)
    cd Chatbot
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
   

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    

### How to Use

1.  Upload a PDF document using the file uploader on the sidebar.
2.  Wait for the processing.
3.  Type your question in the chat box and press Enter.
4.  The chatbot will retrieve the relevant information and provide an answer.


### Technologies Used

-   **Python**
-   **Streamlit**
-   **LangChain / Haystack** 
-   **ChromaDB** 
-   **Hugging Face Models** 


---
