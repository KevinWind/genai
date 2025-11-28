# Agentic RAG Tutorial Implementation

Based on the [LangChain LangGraph Agentic RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#8-run-the-agentic-rag)

## Overview

This tutorial builds a retrieval agent that can decide whether to retrieve context from a vectorstore or respond directly to the user. The agent will:

1. Fetch and preprocess documents for retrieval
2. Index documents for semantic search and create a retriever tool
3. Build an agentic RAG system that decides when to use retrieval

## Prerequisites

### Required Packages

Add to your `requirements.txt`:
```txt
# Agentic RAG dependencies
langgraph>=0.1.0
langchain[openai]>=0.1.0
langchain-community>=0.1.0
langchain-text-splitters>=0.1.0
langchain-openai>=0.1.0
tiktoken>=0.5.0
```

### Installation Commands

```python
# In notebook cell
import sys
!{sys.executable} -m pip install --index-url https://pypi.org/simple/ langgraph "langchain[openai]" langchain-community langchain-text-splitters langchain-openai tiktoken
```

### API Keys Setup

You'll need an OpenAI API key. Set it as an environment variable:

```python
import getpass
import os

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")

_set_env("OPENAI_API_KEY")
```

## Implementation Steps

### Step 1: Document Preprocessing

Create documents for our RAG system using Lilian Weng's blog posts:

```python
from langchain_community.document_loaders import WebBaseLoader

# Fetch documents
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/", 
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

# Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
```

### Step 2: Create Retriever Tool

Set up vector store and create a retriever tool:

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# Create vector store
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)
```

### Step 3: Generate Query Node

Build the main decision-making node:

```python
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("openai:gpt-4.1", temperature=0)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response or decide to retrieve."""
    response = (
        response_model
        .bind_tools([retriever_tool])
        .invoke(state["messages"])
    )
    return {"messages": [response]}
```

### Step 4: Document Grading

Create a function to grade document relevance:

```python
from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model("openai:gpt-4.1", temperature=0)

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments)
        .invoke([{"role": "user", "content": prompt}])
    )
    
    score = response.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
```

### Step 5: Question Rewriting

Create a function to rewrite questions when documents aren't relevant:

```python
REWRITE_PROMPT = (
    "You are a question re-writer that converts an input question to a better version that is optimized \n "
    "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."
    "\n ------- \n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [{"role": "user", "content": response.content}]}
```

### Step 6: Answer Generation

Create the final answer generation function:

```python
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [response]}
```

### Step 7: Assemble the Graph

Build the complete workflow:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Create the workflow
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# Add edges
workflow.add_edge(START, "generate_query_or_respond")

# Conditional edges
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile the graph
graph = workflow.compile()
```

### Step 8: Run the Agentic RAG

Test the complete system:

```python
# Test the agentic RAG system
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print(f"Update from node {node}")
        update["messages"][-1].pretty_print()
        print("\n" + "="*50 + "\n")
```

## Complete Implementation File

Create a Python file with all the code combined. See `agentic_rag_implementation.py` for the complete implementation.

## Usage Examples

```python
# Example 1: Direct response (no retrieval needed)
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
})

# Example 2: Question requiring retrieval
result = graph.invoke({
    "messages": [{"role": "user", "content": "What are the types of reward hacking mentioned by Lilian Weng?"}]
})

# Example 3: Complex question that might need rewriting
result = graph.invoke({
    "messages": [{"role": "user", "content": "Tell me about ML stuff from the blog"}]
})
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Make sure your API key is set correctly
2. **Package Conflicts**: Use the `--index-url https://pypi.org/simple/` flag if packages aren't found
3. **Memory Issues**: Reduce chunk_size if you encounter memory problems
4. **Rate Limits**: Add delays between API calls if needed

### Debugging Tips

```python
# Enable debug mode to see detailed execution
import logging
logging.basicConfig(level=logging.DEBUG)

# Visualize the graph
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

## Next Steps

1. **Customize the retriever**: Use different embedding models or vector stores
2. **Add more document sources**: Expand beyond blog posts
3. **Improve grading**: Use more sophisticated relevance scoring
4. **Add memory**: Make the agent remember previous conversations
5. **Deploy**: Turn this into a web app using Gradio or Streamlit

## References

- [Original Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#8-run-the-agentic-rag)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/) 