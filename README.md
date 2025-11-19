# LangChain Testing Scripts

This repository contains various Python scripts for testing and learning LangChain capabilities with different LLM providers.

## 📋 Prerequisites

- Python 3.13+
- Virtual environment (`.venv`)
- API keys configured in `.env` file

## 🔑 Configuration

Create a `.env` file in the root directory with the following keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
```

## 📁 Scripts Description

### `src/config.py`
Configuration file that loads environment variables from `.env` using `python-dotenv`. Provides a `set_environment()` function to expose all uppercase variables as environment variables.

### `src/chat/chat.py`
**Purpose:** Test LangChain with Google Gemini for conversational AI with chaining.

**Features:**
- Uses `ChatGoogleGenerativeAI` with Gemini 2.5 Flash model
- Demonstrates `PromptTemplate` for structured prompts
- Implements LCEL (LangChain Expression Language) with pipe operator (`|`)
- Tests multiple chaining approaches:
  - **Basic chains**: Separate joke generation and analysis
  - **RunnablePassthrough**: Progressive dictionary building
  - **itemgetter**: Data extraction between chain steps
  - **Combined elegant chain**: Full data pipeline with intermediate results
- Temperature set to 2.0 for maximum creativity
- Output parsing with `StrOutputParser`

**Use case:** Generate a joke and then analyze why it's funny using two sequential LLM calls.

### `src/explain_concept/explain_concept.py`
**Purpose:** Test LangChain with Ollama (local LLM) for concept explanation.

**Features:**
- Uses `ChatOllama` with DeepSeek-R1:1.5b model
- Centralized error handling with `invoke_llm_chain_with_error_handling()` function
- Custom prompt template for simplifying complex concepts
- LCEL chain implementation
- Comprehensive error management:
  - Connection errors (Ollama not running)
  - Timeout errors
  - Value errors
  - Generic exceptions with helpful suggestions

**Use case:** Explain quantum computing in simple terms using a local LLM with robust error handling.

### `src/generate_image/generate_image.py`
**Purpose:** Test LangChain for image generation using Replicate API.

**Features:**
- Uses Replicate's Flux 1.1 Pro model (Black Forest Labs)
- Custom LangChain chain with `RunnableLambda` components
- LCEL pipeline with three stages:
  1. `call_replicate_api`: Generate image via Replicate
  2. `prepare_download`: Transform URL to download parameters
  3. `download_image`: Save image locally
- Output: High-quality PNG images (1024x1024)
- Handles multiple FileOutput formats
- Complete error handling with helpful messages

**Use case:** Generate a surreal image of "upside-down houses on a street" using AI image generation through a LangChain chain.

### `src/list_models.py`
**Purpose:** Utility script to list all available Gemini models.

**Features:**
- Lists all models supporting `generateContent` method
- Helps identify available model versions
- Useful for troubleshooting model availability

**Use case:** Discover which Gemini models are available with your API key.

### `src/langGraph/langGraph.py`
**Purpose:** Test LangGraph's StateGraph API with conditional routing and reducers.

**Features:**
- **StateGraph**: Build complex workflow graphs with nodes and edges
- **Conditional edges**: Dynamic routing based on sentiment analysis
- **Reducers**: State accumulation patterns
  - `Annotated[list, add]`: Concatenate lists for history tracking
  - Custom `increment_counter`: Always increment by 1
- **State management**: TypedDict with typed state structure
- **Graph visualization**: Export graph structure to PNG using Mermaid
- Multi-path routing based on sentiment (positive/negative/neutral)
- Loop control with iteration limits
- Comprehensive state tracking:
  - Current values (sentiment, response)
  - Accumulated history (sentiment_history, response_history)
  - Iteration counter

**Graph Structure:**
```
START → analyze_sentiment → [conditional by sentiment]
         ↓                 ↓                 ↓
   handle_positive  handle_negative  handle_neutral
         ↓                 ↓                 ↓
         └─────────→ check_followup ←───────┘
                          ↓
                [conditional by iteration]
                    ↓         ↓
                  END    reanalyze (loop)
```

**Use case:** Demonstrate sentiment analysis workflow with multiple paths and state accumulation using reducers.

### `src/output_parser/output_parser.py`
**Purpose:** Test LangChain output parsers with structured responses using LCEL and LangGraph.

**Features:**
- **PydanticOutputParser**: Parse LLM responses into structured Pydantic models
- **YesNoEnum**: Custom enum for binary Yes/No responses
- **Pydantic validation**: Automatic validation and type conversion with `@field_validator`
- **LCEL chain**: `prompt | llm | parser` for structured output parsing
- **LangGraph conditional routing**: Different actions based on parsed Yes/No response
- **Graph visualization**: Export workflow to PNG
- Robust parsing with multiple variations ("yes", "y", "true", "no", "n", "false")
- Error handling with default fallback to NO

**Graph Structure:**
```
START → ask_question → [conditional by YES/NO]
              ↓                    ↓
      action_positive      action_negative
              ↓                    ↓
              └──→ finalize ←──────┘
                      ↓
                    END
```

**Use case:** Ask Gemini yes/no questions, parse structured responses with PydanticOutputParser, and execute different actions based on the parsed enum value.

## 🚀 Running the Scripts

```bash
# Activate virtual environment
source .venv/bin/activate

# Run conversational AI with chaining examples
python src/chat/chat.py

# Run concept explanation with Ollama
python src/explain_concept/explain_concept.py

# Run image generation (requires Replicate API token)
python src/generate_image/generate_image.py

# List available Gemini models
python src/list_models.py

# Run LangGraph StateGraph example with conditionals and reducers
python src/langGraph/langGraph.py

# Run output parser example with PydanticOutputParser
python src/output_parser/output_parser.py
```

## 🧪 Key LangChain Concepts Tested

### Core LangChain
- **PromptTemplate**: Reusable prompt structures with variables
- **LCEL (LangChain Expression Language)**: Chain composition with `|` operator
- **RunnablePassthrough**: Pass and enrich data through chains
- **itemgetter**: Extract specific values from dictionaries
- **RunnableLambda**: Custom chain components
- **Output Parsers**: Extract structured data from LLM responses
  - **PydanticOutputParser**: Parse to Pydantic models with validation
  - **StrOutputParser**: Extract plain text
  - Structured output with enums and custom validation
- **Error Handling**: Centralized error management patterns
- **Multi-provider Support**: Gemini, Ollama, Replicate integration

### LangGraph
- **StateGraph**: Build complex workflow graphs with conditional logic
- **Reducers**: State accumulation with `Annotated[type, reducer]`
  - `operator.add`: Concatenate lists/strings, sum numbers
  - Custom reducers: Define your own merge logic
- **Conditional Edges**: Dynamic routing based on state
- **Graph Visualization**: Export to PNG/Mermaid diagrams
- **Stateful Workflows**: TypedDict state management across nodes

## 📦 Dependencies

```bash
pip install langchain langchain-core langchain-google-genai langchain-ollama langgraph python-dotenv google-generativeai replicate requests pygraphviz
```

**Note:** `pygraphviz` requires system-level graphviz installation:
```bash
# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev

# macOS
brew install graphviz

# Fedora
sudo dnf install graphviz graphviz-devel
```

## 🔒 Security

- API keys are stored in `.env` (not versioned)
- `.gitignore` configured to exclude sensitive files
- Follow best practices for API key management

## 📝 Notes

- `src/chat/chat.py` requires internet connection and Gemini API access
- `src/explain_concept/explain_concept.py` requires Ollama running locally (`ollama serve`)
- `src/generate_image/generate_image.py` requires Replicate API token with credits
- All scripts implement proper error handling and informative messages

## 🎯 Learning Objectives

This repository demonstrates:
1. How to build LangChain chains with LCEL
2. How to integrate multiple LLM providers (Gemini, Ollama, Replicate)
3. How to handle errors gracefully in production scenarios
4. How to create reusable prompt templates
5. How to chain multiple LLM calls for complex workflows
6. How to extend LangChain with custom components (image generation)
7. **How to build stateful workflows with LangGraph**
8. **How to use reducers for state accumulation**
9. **How to implement conditional routing in graphs**
10. **How to visualize and debug complex workflows**
11. **How to parse structured outputs with PydanticOutputParser**
12. **How to validate LLM responses with Pydantic models and enums**
