# PXGPT - Plant Analysis Tool

**PXGPT** (Phenotype eXplore GPT) is a command-line tool for analyzing plant images using multiple LLM providers (Anthropic Claude, OpenAI, Google, Ollama, LM Studio).

## Features

- **Multiple LLM Providers**: Support for Anthropic, OpenAI, Google, Ollama, and LM Studio
- **Prompt Caching**: Automatic prompt caching for Anthropic (reduces costs on repeated requests)
- **Unified Interface**: Consistent CLI across all providers
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Structured Output**: Support for JSON schema validation

## 📖 User Manual

**For complete plant phenotyping workflows, advanced usage, and troubleshooting, please read the [User Manual](user_manual.md).**

The user manual covers:
- Complete plant phenotyping workflow (analyze → merge → manual schema generation → structured analysis)
- Provider configuration and optimization
- Best practices for prompt engineering
- Batch processing large germplasm collections
- Schema design using GUI interfaces (Claude.app, ChatGPT, Gemini)
- Troubleshooting and debugging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xavierzheng/pxgpt.git
cd pxgpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

4. Set up your environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
# Required: At least one provider API key
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# For local Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Default provider
DEFAULT_PROVIDER=anthropic
```

### Local Provider Setup

#### Using LM Studio

LM Studio provides an OpenAI-compatible API. To use it:

1. **Start LM Studio** and load your preferred vision model (e.g., gemma3:12b, llama-3.2-1b-instruct)

2. **Configure your .env** to point to LM Studio:
   ```bash
   # Point OpenAI provider to LM Studio
   OPENAI_BASE_URL=http://localhost:1234/v1
   OPENAI_API_KEY=lm-studio
   OPENAI_MODEL=gemma3:12b  # Match the model you loaded in LM Studio
   ```

3. **Use with openai provider**:
   ```bash
   pxgpt analyze --provider openai --input-folder /path/to/images --output results.txt
   ```

**Alternative: Temporary override without .env changes:**
```bash
OPENAI_BASE_URL=http://localhost:1234/v1 OPENAI_API_KEY=lm-studio OPENAI_MODEL=gemma3:12b \
pxgpt analyze --provider openai --input-folder /path/to/images --output results.txt
```

#### Using Ollama

For Ollama, use the default configuration:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b  # Or your preferred vision model
```

## Usage

### Basic Analysis

Analyze images with text output:

```bash
pxgpt analyze \
  --input-folder /path/to/images \
  --output results.txt \
  --system-prompt system.txt \
  --prompt user_prompt.txt \
  --provider anthropic
```

### Structured Analysis

Generate structured JSON output with schema validation:

```bash
pxgpt schema \
  --input-folder /path/to/images \
  --output results.json \
  --system-prompt system.txt \
  --schema plant_schema.json \
  --prompt user_prompt.txt \
  --provider anthropic
```

## Commands

### `pxgpt analyze`

Basic image analysis with text output.

**Options:**
- `--input-folder`: Path to folder containing .jpg images
- `--output`: Output file path  
- `--system-prompt`: System prompt file path
- `--prompt`: User prompt file path
- `--provider`: LLM provider (anthropic, openai, google, ollama)
  - Note: Use `openai` for LM Studio with custom base URL

### `pxgpt schema`

Structured JSON analysis with schema validation.

**Options:**
- `--input-folder`: Path to folder containing .jpg images
- `--output`: Output file path
- `--system-prompt`: System prompt file path
- `--schema`: JSON schema file path
- `--prompt`: User prompt file path  
- `--provider`: LLM provider (anthropic, openai, google, ollama)
  - Note: Use `openai` for LM Studio with custom base URL

## Providers

### Anthropic Claude
- **Prompt Caching**: Enabled (reduces costs for repeated schema/system prompts)
- **Models**: claude-3-7-sonnet-20250219
- **Token Estimation**: Supported

### OpenAI GPT-4 Vision
- **Prompt Caching**: Not supported
- **Models**: gpt-5-2025-08-07
- **Via**: LiteLLM

### Google Gemini
- **Prompt Caching**: Not supported  
- **Models**: gemini-2.5-pro
- **Via**: LiteLLM

### Ollama (Local)
- **Prompt Caching**: Not supported
- **Models**: gemma3:12b (or other vision models)
- **Local**: Runs on your machine

### LM Studio (Local)
- **Prompt Caching**: Not supported
- **Models**: Any model you load (gemma3:12b, etc.)
- **Local**: Runs on your machine
- **Setup**: Use OpenAI provider with custom base URL

## Error Handling

The tool includes robust error handling:

- **Rate Limits**: Automatic retry with configurable delays
- **Connection Errors**: Exponential backoff with jitter
- **File Errors**: Clear error messages for missing/unreadable files
- **API Errors**: Provider-specific error handling

## Development

Project structure:
```
pxgpt/
├── core/           # Common utilities
├── providers/      # LLM provider implementations  
├── commands/       # CLI commands
└── main.py         # CLI entry point
```

## Documentation

For comprehensive usage instructions, workflow examples, and troubleshooting, see the **[User Manual](user_manual.md)**.

## License

MIT License
