"""
LangGraph StateGraph Example with Conditionals and Reducers

This script demonstrates how to use LangGraph's StateGraph API with:
- State management with REDUCERS
- Conditional edges
- Graph visualization (PNG export)
- Accumulation of history using reducers
"""

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


# Custom reducer for counting
def increment_counter(old: int, new: int) -> int:
    """Custom reducer: always increment by 1 regardless of new value"""
    return old + 1


# Define the state structure WITH REDUCERS
class AgentState(TypedDict):
    """State structure for the graph with reducers"""
    input: str  # No reducer: replaced each time
    sentiment: str  # No reducer: replaced each time
    response: str  # No reducer: replaced each time
    
    # REDUCER: Accumulate sentiment history (concatenate lists)
    sentiment_history: Annotated[list[str], add]
    
    # REDUCER: Accumulate all responses (concatenate lists)
    response_history: Annotated[list[str], add]
    
    # REDUCER: Custom increment counter
    iteration_count: Annotated[int, increment_counter]


def analyze_sentiment(state: AgentState) -> AgentState:
    """
    Node 1: Analyze the sentiment of the input text.
    Demonstrates reducer: adds sentiment to history list.
    """
    input_text = state["input"].lower()
    
    # Simple sentiment analysis
    positive_words = ["good", "great", "happy", "excellent", "wonderful", "love"]
    negative_words = ["bad", "sad", "terrible", "awful", "hate", "poor"]
    
    positive_count = sum(1 for word in positive_words if word in input_text)
    negative_count = sum(1 for word in negative_words if word in input_text)
    
    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    print(f"📊 Sentiment Analysis: {sentiment}")
    
    # Return updates - reducers will handle accumulation
    return {
        "sentiment": sentiment,
        "sentiment_history": [sentiment],  # Reducer will ADD to list
        "iteration_count": 1  # Reducer will INCREMENT
    }


def handle_positive(state: AgentState) -> AgentState:
    """
    Node 2a: Handle positive sentiment.
    Demonstrates reducer: adds response to history list.
    """
    print("✅ Handling positive sentiment...")
    
    response = "That's wonderful! I'm glad to hear positive feedback. How can I help you further?"
    
    return {
        "response": response,
        "response_history": [response]  # Reducer will ADD to list
    }


def handle_negative(state: AgentState) -> AgentState:
    """
    Node 2b: Handle negative sentiment.
    Demonstrates reducer: adds response to history list.
    """
    print("⚠️ Handling negative sentiment...")
    
    response = "I understand your frustration. Let me help you resolve this issue."
    
    return {
        "response": response,
        "response_history": [response]  # Reducer will ADD to list
    }


def handle_neutral(state: AgentState) -> AgentState:
    """
    Node 2c: Handle neutral sentiment.
    Demonstrates reducer: adds response to history list.
    """
    print("ℹ️ Handling neutral sentiment...")
    
    response = "I see. Could you provide more details so I can assist you better?"
    
    return {
        "response": response,
        "response_history": [response]  # Reducer will ADD to list
    }


def check_need_followup(state: AgentState) -> AgentState:
    """
    Node 3: Check if follow-up is needed based on iteration count.
    """
    print(f"🔄 Checking follow-up need (iteration: {state['iteration_count']})...")
    
    return state


def route_by_sentiment(state: AgentState) -> Literal["positive", "negative", "neutral"]:
    """
    Conditional router: Route to different nodes based on sentiment.
    This function determines which path to take in the graph.
    """
    sentiment = state["sentiment"]
    print(f"🔀 Routing based on sentiment: {sentiment}")
    return sentiment


def route_by_iteration(state: AgentState) -> Literal["end", "reanalyze"]:
    """
    Conditional router: Decide if we need to end or continue.
    This demonstrates loop control in the graph.
    """
    max_iterations = 2
    
    if state["iteration_count"] >= max_iterations:
        print(f"🛑 Max iterations ({max_iterations}) reached. Ending.")
        return "end"
    else:
        print(f"🔁 Iteration {state['iteration_count']} < {max_iterations}. Could reanalyze if needed.")
        return "end"  # For this example, we end after first analysis


def create_sentiment_graph() -> StateGraph:
    """
    Create a StateGraph with conditional routing.
    
    Graph structure:
    START -> analyze_sentiment -> (conditional routing based on sentiment)
                                -> handle_positive -> check_followup -> (conditional routing)
                                -> handle_negative -> check_followup -> (conditional routing)
                                -> handle_neutral -> check_followup -> (conditional routing)
                                                                    -> END or back to analyze
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("handle_positive", handle_positive)
    workflow.add_node("handle_negative", handle_negative)
    workflow.add_node("handle_neutral", handle_neutral)
    workflow.add_node("check_followup", check_need_followup)
    
    # Add edges
    # Start with sentiment analysis
    workflow.add_edge(START, "analyze_sentiment")
    
    # Conditional edge: Route based on sentiment
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_by_sentiment,
        {
            "positive": "handle_positive",
            "negative": "handle_negative",
            "neutral": "handle_neutral"
        }
    )
    
    # All sentiment handlers go to follow-up check
    workflow.add_edge("handle_positive", "check_followup")
    workflow.add_edge("handle_negative", "check_followup")
    workflow.add_edge("handle_neutral", "check_followup")
    
    # Conditional edge: Decide if we end or loop back
    workflow.add_conditional_edges(
        "check_followup",
        route_by_iteration,
        {
            "end": END,
            "reanalyze": "analyze_sentiment"  # Loop back if needed
        }
    )
    
    return workflow


def visualize_graph(graph: StateGraph, output_path: str = "sentiment_graph.png"):
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
        print("   Note: Graph visualization requires graphviz to be installed on your system.")


def run_graph_example():
    """
    Main function to demonstrate the StateGraph.
    """
    print("=" * 70)
    print("LANGGRAPH STATEGRAPH EXAMPLE WITH CONDITIONALS")
    print("=" * 70)
    print()
    
    # Create the graph
    workflow = create_sentiment_graph()
    
    # Compile the graph
    app = workflow.compile()
    
    # Test cases
    test_inputs = [
        "This is a great product! I love it!",
        "This is terrible. I hate this experience.",
        "The weather is okay today."
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {i}")
        print(f"{'=' * 70}")
        print(f"Input: {test_input}\n")
        
        # Initial state with empty lists for reducers
        initial_state = {
            "input": test_input,
            "sentiment": "",
            "response": "",
            "sentiment_history": [],  # Will be accumulated by reducer
            "response_history": [],   # Will be accumulated by reducer
            "iteration_count": 0      # Will be incremented by reducer
        }
        
        # Run the graph
        result = app.invoke(initial_state)
        
        # Display results
        print(f"\n📋 RESULTS:")
        print(f"   Current Sentiment: {result['sentiment']}")
        print(f"   Current Response: {result['response']}")
        print(f"   Iterations: {result['iteration_count']}")
        print(f"\n📚 REDUCER DEMONSTRATIONS:")
        print(f"   Sentiment History (list reducer): {result['sentiment_history']}")
        print(f"   Response History (list reducer): {result['response_history']}")
        print(f"   💡 Notice how reducers ACCUMULATED the data instead of replacing it!")
    
    print(f"\n{'=' * 70}")
    
    # Visualize the graph
    graph_output_path = os.path.join(os.path.dirname(__file__), "sentiment_graph.png")
    visualize_graph(workflow, graph_output_path)
    
    print(f"\n{'=' * 70}")
    print("✅ LangGraph example completed!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_graph_example()
