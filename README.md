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
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project environment.)*

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    *(Note: Assuming your main application file is named `app.py`.)*

### How to Use

1.  Upload a PDF document using the file uploader on the sidebar.
2.  Wait for the "Processing done!" message.
3.  Type your question in the chat box and press Enter.
4.  The chatbot will retrieve the relevant information and provide an answer.

### Demo

*(Optional: Add a screenshot or a short GIF here to visually show how the chatbot works. This is highly recommended for user engagement.)*

### Technologies Used

-   **Python**
-   **Streamlit**
-   **LangChain / Haystack** *(Choose the one you used, or list the specific libraries for RAG)*
-   **Faiss / ChromaDB** *(Specify the vector store you used)*
-   **Hugging Face Models** *(If you used a local LLM or Sentence Transformer model)*

### License

This project is licensed under the MIT License.

---
