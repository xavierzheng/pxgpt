# PXGPT - Plant Analysis Tool

A command-line tool for analyzing plant images using multiple LLM providers (Anthropic Claude, OpenAI, Google, Ollama).

## Features

- **Multiple LLM Providers**: Support for Anthropic, OpenAI, Google, and local Ollama models
- **Prompt Caching**: Automatic prompt caching for Anthropic (reduces costs on repeated requests)
- **Unified Interface**: Consistent CLI across all providers
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Structured Output**: Support for JSON schema validation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PlantGPT/script/ForGitHub
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

### `pxgpt schema`

Structured JSON analysis with schema validation.

**Options:**
- `--input-folder`: Path to folder containing .jpg images
- `--output`: Output file path
- `--system-prompt`: System prompt file path
- `--schema`: JSON schema file path
- `--prompt`: User prompt file path  
- `--provider`: LLM provider (anthropic, openai, google, ollama)

## Providers

### Anthropic Claude
- **Prompt Caching**: Enabled (reduces costs for repeated schema/system prompts)
- **Models**: claude-3-7-sonnet-20250219
- **Token Estimation**: Supported

### OpenAI GPT-4 Vision
- **Prompt Caching**: Not supported
- **Models**: gpt-4-vision-preview
- **Via**: LiteLLM

### Google Gemini
- **Prompt Caching**: Not supported  
- **Models**: gemini-1.5-pro-latest
- **Via**: LiteLLM

### Ollama (Local)
- **Prompt Caching**: Not supported
- **Models**: llama3.2-vision (or other vision models)
- **Local**: Runs on your machine

## Error Handling

The tool includes robust error handling:

- **Rate Limits**: Automatic retry with configurable delays
- **Connection Errors**: Exponential backoff with jitter
- **File Errors**: Clear error messages for missing/unreadable files
- **API Errors**: Provider-specific error handling

## Migration from Original Scripts

The original scripts are preserved:
- `AskClaude_001_each.py` → `pxgpt analyze`
- `AskClaude_004_apply_schema.py` → `pxgpt schema`

Both maintain the same functionality but with improved error handling and multi-provider support.

## Development

Project structure:
```
pxgpt/
├── core/           # Common utilities
├── providers/      # LLM provider implementations  
├── commands/       # CLI commands
└── main.py         # CLI entry point
```

## License

MIT License