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

### `config.py`
Configuration file that loads environment variables from `.env` using `python-dotenv`. Provides a `set_environment()` function to expose all uppercase variables as environment variables.

### `chat.py`
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

### `explain_concept.py`
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

### `generate_image.py`
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

### `list_models.py`
**Purpose:** Utility script to list all available Gemini models.

**Features:**
- Lists all models supporting `generateContent` method
- Helps identify available model versions
- Useful for troubleshooting model availability

**Use case:** Discover which Gemini models are available with your API key.

## 🚀 Running the Scripts

```bash
# Activate virtual environment
source .venv/bin/activate

# Run conversational AI with chaining examples
python chat.py

# Run concept explanation with Ollama
python explain_concept.py

# Run image generation (requires Replicate API token)
python generate_image.py

# List available Gemini models
python list_models.py
```

## 🧪 Key LangChain Concepts Tested

- **PromptTemplate**: Reusable prompt structures with variables
- **LCEL (LangChain Expression Language)**: Chain composition with `|` operator
- **RunnablePassthrough**: Pass and enrich data through chains
- **itemgetter**: Extract specific values from dictionaries
- **RunnableLambda**: Custom chain components
- **Output Parsers**: Extract structured data from LLM responses
- **Error Handling**: Centralized error management patterns
- **Multi-provider Support**: Gemini, Ollama, Replicate integration

## 📦 Dependencies

```bash
pip install langchain langchain-core langchain-google-genai langchain-ollama python-dotenv google-generativeai replicate requests
```

## 🔒 Security

- API keys are stored in `.env` (not versioned)
- `.gitignore` configured to exclude sensitive files
- Follow best practices for API key management

## 📝 Notes

- `chat.py` requires internet connection and Gemini API access
- `explain_concept.py` requires Ollama running locally (`ollama serve`)
- `generate_image.py` requires Replicate API token with credits
- All scripts implement proper error handling and informative messages

## 🎯 Learning Objectives

This repository demonstrates:
1. How to build LangChain chains with LCEL
2. How to integrate multiple LLM providers
3. How to handle errors gracefully in production scenarios
4. How to create reusable prompt templates
5. How to chain multiple LLM calls for complex workflows
6. How to extend LangChain with custom components (image generation)
