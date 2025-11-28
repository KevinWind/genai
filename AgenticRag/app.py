#!/usr/bin/env python3
"""
Agentic RAG Implementation
Based on: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/

This script implements a retrieval agent that can decide whether to retrieve context
from a vectorstore or respond directly to the user.
"""

import getpass
import os
from typing import Literal

from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition


def setup_environment():
    """Set up environment variables for API keys."""
    def _set_env(key: str):
        if key not in os.environ:
            os.environ[key] = getpass.getpass(f"{key}:")
    
    _set_env("OPENAI_API_KEY")


def preprocess_documents():
    """Fetch and preprocess documents for the RAG system."""
    print("📄 Fetching documents...")
    
    # Fetch documents from Lilian Weng's blog
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/", 
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    
    docs = [WebBaseLoader(url).load() for url in urls]
    print(f"✅ Loaded {len(docs)} documents")
    
    # Split documents into chunks
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"✅ Split into {len(doc_splits)} chunks")
    
    return doc_splits


def setup_retriever_tool(doc_splits):
    """Create a retriever tool from document splits."""
    print("🔍 Creating retriever tool...")
    
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
    
    print("✅ Retriever tool created")
    return retriever_tool


def create_graph_nodes(retriever_tool):
    """Create all the nodes for the RAG graph."""
    print("🏗️ Creating graph nodes...")
    
    # Initialize models
    response_model = init_chat_model("openai:gpt-4.1", temperature=0)
    grader_model = init_chat_model("openai:gpt-4.1", temperature=0)
    
    # Node 1: Generate query or respond
    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response or decide to retrieve."""
        response = (
            response_model
            .bind_tools([retriever_tool])
            .invoke(state["messages"])
        )
        return {"messages": [response]}
    
    # Document grading class
    class GradeDocuments(BaseModel):
        """Grade documents using a binary score for relevance check."""
        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )
    
    # Node 2: Grade documents
    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    
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
        print(f"📊 Document relevance score: {score}")
        
        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"
    
    # Node 3: Rewrite question
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
        
        print(f"🔄 Rewrote question: {response.content}")
        return {"messages": [{"role": "user", "content": response.content}]}
    
    # Node 4: Generate answer
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
    
    print("✅ Graph nodes created")
    return generate_query_or_respond, grade_documents, rewrite_question, generate_answer


def build_graph(retriever_tool, generate_query_or_respond, grade_documents, rewrite_question, generate_answer):
    """Build and compile the complete RAG graph."""
    print("🔗 Building graph...")
    
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
    
    print("✅ Graph compiled successfully")
    return graph


def test_agentic_rag(graph):
    """Test the agentic RAG system with example queries."""
    
    test_cases = [
        {
            "name": "Direct Response Test",
            "question": "Hello! How are you today?"
        },
        {
            "name": "Retrieval Required Test", 
            "question": "What does Lilian Weng say about types of reward hacking?"
        },
        {
            "name": "Complex Question Test",
            "question": "What are the main challenges in video generation using diffusion models?"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"{'='*60}")
        print(f"Question: {test_case['question']}")
        print("-" * 60)
        
        # Run the graph
        for chunk in graph.stream({
            "messages": [
                {
                    "role": "user", 
                    "content": test_case['question']
                }
            ]
        }):
            for node, update in chunk.items():
                print(f"🔄 Update from node: {node}")
                if hasattr(update["messages"][-1], 'pretty_print'):
                    update["messages"][-1].pretty_print()
                else:
                    print(update["messages"][-1])
                print()


def main():
    """Main function to run the complete Agentic RAG implementation."""
    print("🚀 Starting Agentic RAG Implementation")
    print("=" * 50)
    
    try:
        # Step 1: Setup environment
        setup_environment()
        
        # Step 2: Preprocess documents
        doc_splits = preprocess_documents()
        
        # Step 3: Create retriever tool
        retriever_tool = setup_retriever_tool(doc_splits)
        
        # Step 4: Create graph nodes
        generate_query_or_respond, grade_documents, rewrite_question, generate_answer = create_graph_nodes(retriever_tool)
        
        # Step 5: Build graph
        graph = build_graph(retriever_tool, generate_query_or_respond, grade_documents, rewrite_question, generate_answer)
        
        # Step 6: Test the system
        print("\n🧪 Testing the Agentic RAG system...")
        test_agentic_rag(graph)
        
        print("\n✅ Agentic RAG implementation completed successfully!")
        
        # Return graph for further use
        return graph
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    graph = main() 