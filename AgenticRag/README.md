# Agentic RAG Implementation

## 📋 Overview

This repository contains a complete implementation of the **Agentic RAG (Retrieval-Augmented Generation)** system based on the [LangChain LangGraph tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/).

### What is Agentic RAG?

Agentic RAG is an intelligent retrieval system that can:
- 🤖 **Decide** whether to retrieve documents or respond directly
- 📊 **Grade** retrieved documents for relevance  
- ✏️ **Rewrite** questions if documents aren't relevant
- 💡 **Generate** final answers using retrieved context

## 📁 Repository Structure

```
genai/
├── agentic_rag_tutorial.md           # Complete tutorial and instructions
├── agentic_rag_implementation.py     # Full Python implementation
├── agentic_rag_demo.ipynb           # Interactive Jupyter notebook
├── requirements.txt                  # Updated dependencies
├── package_installation_guide.md    # Package installation guide
└── README_AGENTIC_RAG.md            # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- OpenAI API key
- Virtual environment (`.venv` is already set up)

### 2. Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install required packages (use public PyPI for corporate networks)
pip install --index-url https://pypi.org/simple/ langgraph "langchain[openai]" langchain-community langchain-text-splitters langchain-openai tiktoken beautifulsoup4
```

### 3. Set Environment Variables

```python
import os
import getpass

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
```

### 4. Run the Implementation

#### Option A: Run the complete Python script
```bash
python agentic_rag_implementation.py
```

#### Option B: Use the Jupyter notebook
```bash
jupyter lab agentic_rag_demo.ipynb
```

#### Option C: Import and use in your code
```python
from agentic_rag_implementation import main
graph = main()
```

## 📖 Documentation Files

### 1. [`agentic_rag_tutorial.md`](./agentic_rag_tutorial.md)
- **Purpose**: Complete step-by-step tutorial
- **Content**: 
  - Prerequisites and setup
  - Implementation steps 1-8
  - Code examples for each component
  - Usage examples and troubleshooting
  - Next steps and customization

### 2. [`agentic_rag_implementation.py`](./agentic_rag_implementation.py)
- **Purpose**: Full working implementation
- **Features**:
  - Modular functions for each component
  - Built-in testing with multiple scenarios
  - Error handling and logging
  - Ready to run standalone

### 3. [`agentic_rag_demo.ipynb`](./agentic_rag_demo.ipynb)
- **Purpose**: Interactive demonstration
- **Features**:
  - Step-by-step execution
  - Installation commands
  - Interactive testing
  - Educational explanations

## 🔧 Key Components

### 1. Document Preprocessing
- Fetches Lilian Weng's blog posts
- Splits documents into chunks using tiktoken
- Creates embeddings with OpenAI

### 2. Retriever Tool
- Vector store with InMemoryVectorStore
- Semantic search capabilities
- Tool interface for the agent

### 3. Graph Nodes
- **generate_query_or_respond**: Main decision node
- **grade_documents**: Relevance scoring
- **rewrite_question**: Query optimization
- **generate_answer**: Final response generation

### 4. Workflow Graph
- Conditional routing based on tool calls
- Document relevance checks
- Self-correction loops

## 🧪 Testing

The implementation includes three test scenarios:

1. **Direct Response**: "Hello! How are you today?"
2. **Retrieval Required**: "What does Lilian Weng say about types of reward hacking?"
3. **Complex Question**: "What are the main challenges in video generation using diffusion models?"

## 📋 Dependencies

### Core Requirements
```txt
langgraph>=0.1.0
langchain[openai]>=0.1.0
langchain-community>=0.1.0
langchain-text-splitters>=0.1.0
langchain-openai>=0.1.0
tiktoken>=0.5.0
pydantic>=2.0.0
beautifulsoup4>=4.12.0
```

### Installation in Corporate Networks
If you're behind a corporate firewall, use:
```bash
pip install --index-url https://pypi.org/simple/ [package_name]
```

## 🔍 How It Works

### Workflow Diagram
```
Start → Generate Query/Respond → [Decision] 
                ↓                    ↓
         [Tool Call?]          [Direct Response]
                ↓                    ↓
            Retrieve              End
                ↓
         Grade Documents
                ↓
         [Relevant?]
         ↙        ↘
  Generate Answer  Rewrite Question
         ↓              ↓
        End      → Back to Generate
```

### Example Flow
1. **User asks**: "What does Lilian Weng say about reward hacking?"
2. **System decides**: Need to retrieve documents
3. **Retrieves**: Documents about reward hacking
4. **Grades**: Documents are relevant (score: "yes")
5. **Generates**: Final answer using retrieved context

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   # Solution: Set your OpenAI API key
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Package Installation Issues**
   ```bash
   # Solution: Use public PyPI
   pip install --index-url https://pypi.org/simple/ package_name
   ```

3. **Memory Issues**
   ```python
   # Solution: Reduce chunk size
   chunk_size=50  # Instead of 100
   ```

4. **Rate Limits**
   ```python
   # Solution: Add delays
   import time
   time.sleep(1)  # Between API calls
   ```

## 🎯 Next Steps

### Customization Options
1. **Different Documents**: Replace URLs with your own content
2. **Different Models**: Use other LLMs (Claude, Llama, etc.)
3. **Better Embeddings**: Try different embedding models
4. **Persistent Storage**: Use Pinecone, Weaviate, etc.
5. **Web Interface**: Add Gradio or Streamlit UI

### Advanced Features
1. **Memory**: Add conversation history
2. **Multi-hop**: Chain multiple retrievals
3. **Metadata**: Use document metadata for filtering
4. **Evaluation**: Add metrics and benchmarking

## 📚 References

- [Original LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## 🤝 Contributing

Feel free to:
- Add new document sources
- Improve the grading logic
- Add new test cases
- Enhance error handling
- Create web interfaces

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the tutorial documentation
3. Ensure all dependencies are installed correctly
4. Verify your OpenAI API key is valid

---

**Happy building with Agentic RAG! 🚀** 