"""
LangChain Output Parsers with LCEL and LangGraph

This script demonstrates:
- EnumOutputParser for structured Yes/No responses
- LCEL chains for parsing
- LangGraph StateGraph for conditional actions based on parsed output
"""

from enum import Enum
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, field_validator
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.config import GEMINI_API_KEY


# Define the Yes/No Enum
class YesNoEnum(str, Enum):
    """Enum for Yes/No responses"""
    YES = "yes"
    NO = "no"


# Pydantic model for Yes/No response
class YesNoResponse(BaseModel):
    """Pydantic model for structured Yes/No output"""
    answer: YesNoEnum = Field(description="The answer to the question, must be either 'yes' or 'no'")
    
    @field_validator('answer', mode='before')
    @classmethod
    def parse_answer(cls, v):
        """Validate and convert answer to YesNoEnum"""
        if isinstance(v, YesNoEnum):
            return v
        
        # Convert string to lowercase
        v_lower = str(v).strip().lower()
        
        # Try to match to enum values
        if v_lower in ['yes', 'y', 'true', '1']:
            return YesNoEnum.YES
        elif v_lower in ['no', 'n', 'false', '0']:
            return YesNoEnum.NO
        else:
            # Default to NO if unclear
            return YesNoEnum.NO


# Define the state structure for LangGraph
class QuestionState(TypedDict):
    """State structure for the graph"""
    question: str
    raw_response: str
    parsed_response: YesNoResponse
    action_taken: str
    result_message: str


def ask_question_node(state: QuestionState) -> QuestionState:
    """
    Node 1: Ask a question to Gemini and get raw response.
    Uses LCEL chain with EnumOutputParser.
    """
    print(f"\n🤔 Asking question: {state['question']}")
    
    # Create the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3  # Lower temperature for more consistent yes/no answers
    )
    
    # Create PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=YesNoResponse)
    
    # Create prompt template with parser instructions
    prompt = PromptTemplate(
        template="""Answer the following question with ONLY 'yes' or 'no'.

{format_instructions}

Question: {question}""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Create LCEL chain: prompt | llm | parser
    chain = prompt | llm | parser
    
    print("⏳ Waiting for Gemini's response...")
    
    try:
        # Invoke the chain - returns a YesNoResponse object
        parsed_result = chain.invoke({"question": state["question"]})
        
        print(f"✅ Parsed response: {parsed_result.answer.value}")
        
        return {
            "question": state["question"],
            "raw_response": parsed_result.answer.value,
            "parsed_response": parsed_result,
            "action_taken": "",
            "result_message": ""
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Default to NO on error
        default_response = YesNoResponse(answer=YesNoEnum.NO)
        return {
            "question": state["question"],
            "raw_response": "error",
            "parsed_response": default_response,
            "action_taken": "",
            "result_message": f"Error occurred: {str(e)}"
        }


def action_positive_node(state: QuestionState) -> QuestionState:
    """
    Node 2a: Execute action when response is YES.
    """
    print("\n✅ Response is POSITIVE (YES)")
    print("🎉 Executing positive action...")
    
    action = "Positive Action Executed"
    message = "Great! The answer is YES. Proceeding with the affirmative workflow."
    
    # Simulate some action
    print(f"   → {message}")
    
    return {
        **state,
        "action_taken": action,
        "result_message": message
    }


def action_negative_node(state: QuestionState) -> QuestionState:
    """
    Node 2b: Execute action when response is NO.
    """
    print("\n❌ Response is NEGATIVE (NO)")
    print("⚠️  Executing negative action...")
    
    action = "Negative Action Executed"
    message = "The answer is NO. Following the negative workflow path."
    
    # Simulate some different action
    print(f"   → {message}")
    
    return {
        **state,
        "action_taken": action,
        "result_message": message
    }


def finalize_node(state: QuestionState) -> QuestionState:
    """
    Node 3: Finalize and display results.
    """
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS")
    print("=" * 70)
    print(f"Question: {state['question']}")
    print(f"Parsed Response: {state['parsed_response'].answer.value.upper()}")
    print(f"Action Taken: {state['action_taken']}")
    print(f"Result: {state['result_message']}")
    print("=" * 70)
    
    return state


def route_by_response(state: QuestionState) -> Literal["positive", "negative"]:
    """
    Conditional router: Route based on parsed Yes/No response.
    """
    if state["parsed_response"].answer == YesNoEnum.YES:
        print("🔀 Routing to POSITIVE path")
        return "positive"
    else:
        print("🔀 Routing to NEGATIVE path")
        return "negative"


def create_question_workflow() -> StateGraph:
    """
    Create a StateGraph with conditional routing based on parsed output.
    
    Graph structure:
    START -> ask_question -> [conditional routing by YES/NO]
                          -> action_positive -> finalize -> END
                          -> action_negative -> finalize -> END
    """
    # Initialize the graph
    workflow = StateGraph(QuestionState)
    
    # Add nodes
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("action_positive", action_positive_node)
    workflow.add_node("action_negative", action_negative_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.add_edge(START, "ask_question")
    
    # Conditional edge based on parsed enum output
    workflow.add_conditional_edges(
        "ask_question",
        route_by_response,
        {
            "positive": "action_positive",
            "negative": "action_negative"
        }
    )
    
    # Both actions lead to finalize
    workflow.add_edge("action_positive", "finalize")
    workflow.add_edge("action_negative", "finalize")
    
    # Finalize leads to end
    workflow.add_edge("finalize", END)
    
    return workflow


def visualize_graph(graph: StateGraph, output_path: str = "question_workflow.png"):
    """
    Generate and save a PNG visualization of the graph.
    """
    try:
        print(f"\n📊 Generating graph visualization...")
        
        # Compile the graph
        compiled_graph = graph.compile()
        
        # Get the graph as PNG bytes
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(png_bytes)
        
        print(f"✅ Graph saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating graph visualization: {type(e).__name__}")
        print(f"   Details: {str(e)}")


def run_output_parser_example():
    """
    Main function to demonstrate EnumOutputParser with LangGraph.
    """
    print("=" * 70)
    print("LANGCHAIN OUTPUT PARSERS WITH LCEL AND LANGGRAPH")
    print("=" * 70)
    print("\nDemonstrating:")
    print("  1. EnumOutputParser for Yes/No responses")
    print("  2. LCEL chains for parsing")
    print("  3. LangGraph conditional routing based on parsed output")
    print("=" * 70)
    
    # Create the workflow
    workflow = create_question_workflow()
    
    # Compile the graph
    app = workflow.compile()
    
    # Test questions (designed to get Yes/No answers)
    test_questions = [
        "Is Python a programming language?",
        "Is the sky green?",
        "Do humans need water to survive?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'#' * 70}")
        print(f"TEST CASE {i}")
        print(f"{'#' * 70}")
        
        # Initial state
        initial_state = {
            "question": question,
            "raw_response": "",
            "parsed_response": YesNoResponse(answer=YesNoEnum.NO),  # Default
            "action_taken": "",
            "result_message": ""
        }
        
        # Run the workflow
        result = app.invoke(initial_state)
        
        # Small separator between tests
        print()
    
    # Visualize the graph
    graph_output_path = os.path.join(os.path.dirname(__file__), "question_workflow.png")
    visualize_graph(workflow, graph_output_path)
    
    print(f"\n{'=' * 70}")
    print("✅ Output Parser example completed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_output_parser_example()
