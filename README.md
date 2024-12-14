# Retrieval-Augmented Generation for Insurance Q&A

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system to answer insurance-specific questions using a combination of retrieval-based context selection and large language models (LLMs). By integrating retrieval and generation, the system ensures accurate and contextually relevant responses, making it ideal for domain-specific applications.

## Features
- **Exploratory Data Analysis (EDA):** Determined optimal token limit (1024) for efficient processing of question-answer pairs.
- **Chunking Strategies:** Evaluated various methods and finalized treating each Q&A pair as a single chunk for better retrieval performance.
- **Contextual Retrieval:** Utilized **FAISS (Facebook AI Similarity Search)** to retrieve the top-3 most relevant contexts for each query.
- **LLM Integration:** Generated responses using **OpenAI GPT**, ensuring outputs are grounded in retrieved contexts.
- **Evaluation Pipeline:** Assessed system performance with **RAGAs metrics** like answer relevancy, faithfulness, context precision, and recall.

## Workflow
1. **Preprocessing:**
   - Added metadata (e.g., category, subcategory) to enhance context filtering.
   - Embedded Q&A pairs using `text-embedding-ada-002`.
2. **Retrieval:**
   - Indexed embeddings with FAISS for fast similarity-based retrieval.
   - Retrieved top-3 relevant contexts for each user query.
3. **Generation:**
   - Used a prompt template to ensure responses rely solely on the retrieved contexts.

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies:**
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run scripts.py
   ```

## Key Files
- `scripts.py`: Streamlit interface for user interaction.
- `data/insurance.csv`: Dataset containing insurance-specific Q&A pairs.
- `faiss_index/`: Precomputed FAISS index for embeddings.

## Results
- **Token Limit:** Optimized at 1024 tokens for efficiency and coverage.
- **Chunking Strategy:** Treating each Q&A pair as a single chunk improved retrieval and reduced redundancy.
- **Evaluation Metrics:**
  - **Answer Relevancy:** High precision and recall for retrieved answers.
  - **Faithfulness:** Generated responses remained consistent with the provided context.

## Future Work
- Explore fine-tuning GPT for domain-specific language generation.
- Incorporate additional metadata to further refine retrieval.
- Expand dataset to include broader insurance topics.

## License
This project is licensed under the MIT License.
