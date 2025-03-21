# Assignment4.2


## Links 
Codelabs : https://codelabs-preview.appspot.com/?file_id=1FsFp8n3X6xz7IsCHB0NFCE-KWDJJl6qW_YntvJh2XEo#5

Airflow deploy : https://dockerpipeline-618360345344.us-east1.run.app

Streamlit+fastApi: https://dockerpipeline-618360345344.us-east1.run.app

video link : https://youtu.be/uRaUynX7a2M
 

Traditional search and retrieval systems struggle with processing and extracting meaningful insights from unstructured data sources. Existing approaches often lack modularity, extensibility, and efficiency when handling large volumes of text, such as NVIDIA’s quarterly reports over the past five years. Additionally, manually computing embeddings and cosine similarity is computationally expensive, limiting scalability. This project aims to overcome these limitations by implementing a robust, automated RAG pipeline that integrates multiple PDF parsing methods, vector databases (Pinecone, ChromaDB), and advanced chunking strategies to optimize retrieval.
---

## 🛠️ Technology Used

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
[![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![Amazon AWS](https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-%232496ED?style=for-the-badge&logo=Docker&color=blue&logoColor=white)](https://www.docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com)
---

## 🏗️ Architecture Diagram
![AI Application Data Pipeline](https://github.com/Bigdata2025Team5/DAMG7245_Team5_Assignment4/blob/dd602cd51efd57c7d7c9e2042b38a1fd1ee84e7b/Diagrams/architecture_diagram.png)

---

## 🔑 How to Use the Application
Select a PDF Parser: Choose a PDF parser (Docling, Mistral OCR, PyMuPDF) for text extraction based on the document type.
Upload a PDF: Upload an NVIDIA quarterly report, then click "Process" to extract and preprocess the document.
Select a RAG Method: Choose a retrieval method:
	Pinecone vector search
	ChromaDB for advanced retrieval
Choose a Chunking Strategy: Select how the document should be segmented:
	Section-based chunking
	Table-based chunking
	Sliding window chunking
Filter by Quarter: Specify the quarterly report data range to refine retrieval accuracy.
Ask Questions on the Document: Enter a question related to the document, click "Ask Question," and receive an AI-generated response.
Download Processed Data: Select a processed document and click "Download" to retrieve the extracted text and processed insights.
 
---

## 📂 Project Structure
```
├── Backend
│   └── local_chormadb
    ├── Dockerfile
    ├── doclingextract.py
    ├── main.py
    ├── mistralextract.py
    ├── opensource.py
    ├── requirements.txt
├── Diagrams
│   ├── architecture_diagrams.pmg
├── Documentation
├── Frontend
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
├── POC
├── airflow
│   ├── dags
│        ├── etl_pipeline_dag.py
    ├── Dockerfile
    ├── requirements.txt
    ├── docker-compose.yml   
├── AiDisclosure.md
├── README.md

```

---
## References

- Docling GitHub Repository
- Mistral OCR Documentation
- Apache Airflow Documentation
- Pinecone Documentation 
- ChromaDB Documentation
- FastAPI & Streamlit Documentation
- AWS S3 Best Practices

---

## 👥 Team Information
| Name            | Student ID    | Contribution |
|----------------|--------------|--------------|
| **Pranjal Mahajan** | 002375449  | 33.33% |
| **Srushti Patil**  | 002345025  | 33.33% |
| **Ram Putcha**  | 002304724  | 33.33% |

---
