# PXGPT Development Plan

## Architecture Philosophy (Linus-style thinking)

**Core Principle**: Don't fucking over-engineer this. We have two scripts that do similar things with different interfaces. Fix the interface, share the common code, make it extensible without being masturbatory about it.

**The Problem**: 
- Duplicated infrastructure (API clients, error handling, image processing)
- Inconsistent CLI interfaces 
- Anthropic-only with no provider flexibility
- Manual prompt caching handling that won't work with other providers

**The Solution**: 
Build a proper CLI tool with subcommands that shares common infrastructure but handles provider-specific optimizations correctly.

## Project Structure

```
pxgpt/
├── pxgpt/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── image_utils.py   # Image processing utilities
│   │   ├── file_utils.py    # File operations and text cleaning
│   │   └── config.py        # Configuration management
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base provider
│   │   ├── anthropic_provider.py  # Anthropic with caching
│   │   └── litellm_provider.py    # LiteLLM for other providers
│   └── commands/
│       ├── __init__.py
│       ├── analyze.py       # Basic analysis command
│       └── schema.py        # Structured analysis command
├── setup.py
├── requirements.txt
├── .env.example
└── README.md
```

## CLI Interface Design

### Main Command Structure
```bash
pxgpt <command> [options]

Commands:
  analyze     Basic image analysis with text output
  schema      Structured JSON analysis with schema validation

Global Options:
  --provider {anthropic,openai,google,ollama}  # Default: anthropic
  --config FILE                               # Config file path
  --verbose                                   # Verbose output
```

### Subcommand Interfaces

**pxgpt analyze**
```bash
pxgpt analyze --input-folder PATH --output FILE --prompt FILE --system-prompt FILE [--provider PROVIDER]
```

**pxgpt schema**  
```bash
pxgpt schema --input-folder PATH --output FILE --prompt FILE --system-prompt FILE --schema FILE [--provider PROVIDER]
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. **Provider Abstraction Layer**
   - Abstract base class for all providers
   - Unified interface for different LLM providers
   - Handle provider-specific features (caching for Anthropic, none for others)

2. **Unified Error Handling & Rate Limiting**
   - Exponential backoff with jitter
   - Provider-specific error codes mapping
   - Configurable retry policies
   - Proper logging for debugging

3. **Common Utilities**
   - Image processing (base64 encoding)
   - File operations with proper error handling
   - Text cleaning utilities
   - Configuration management

### Phase 2: Provider Implementations

**Anthropic Provider:**
- Use prompt caching for system prompts and schema
- Handle Anthropic-specific token counting
- Rate limit handling with 60s sleep

**LiteLLM Provider:**
- Support OpenAI, Google, Ollama through LiteLLM
- NO prompt caching (not supported by other providers)
- Generic rate limiting and error handling
- Provider-specific configuration

### Phase 3: CLI Commands
- Migrate existing logic to command classes
- Consistent argument parsing
- Proper error reporting
- Progress indicators for long operations

## Provider-Specific Handling

### Anthropic (with caching)
```python
system=[
    {"type": "text", "text": system_prompt},
    {"type": "text", "text": json_schema, "cache_control": {"type": "ephemeral"}}
]
```

### Other Providers (no caching)
```python
# Combine system prompt and schema into single system message
combined_system = f"{system_prompt}\n\nUse this JSON schema:\n{json_schema}"
```

## Environment Configuration

### .env.example
```bash
# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Google
GOOGLE_API_KEY=your_google_key_here

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434

# Default provider
DEFAULT_PROVIDER=anthropic
```

## Error Handling Strategy

1. **Network Errors**: Exponential backoff with max retries
2. **Rate Limits**: Provider-specific handling (60s for Anthropic, configurable for others)
3. **File Errors**: Clear error messages with suggestions
4. **API Errors**: Map provider-specific errors to common error types
5. **Validation Errors**: Pre-flight checks for required files and arguments

## Development Phases

### Phase 1: Foundation (Priority 1)
- [ ] Create project structure
- [ ] Implement base provider class
- [ ] Create unified error handling
- [ ] Implement common utilities

### Phase 2: Providers (Priority 1) 
- [ ] Implement Anthropic provider with caching
- [ ] Implement LiteLLM provider
- [ ] Add provider configuration management

### Phase 3: CLI (Priority 2)
- [ ] Create main CLI entry point
- [ ] Implement analyze command
- [ ] Implement schema command
- [ ] Add configuration file support

### Phase 4: Polish (Priority 3)
- [ ] Add progress indicators
- [ ] Improve error messages
- [ ] Add comprehensive logging
- [ ] Write documentation

## Testing Strategy

- Unit tests for each provider
- Integration tests with mock APIs
- CLI tests with sample data
- Error condition testing

## Migration Path

1. Keep original scripts functional
2. Implement new CLI alongside old scripts
3. Test thoroughly with existing workflows
4. Replace old scripts once validated

---

**Bottom Line**: We're building a tool that works reliably, handles errors gracefully, and doesn't lock users into a single provider. No fancy bullshit, just solid engineering.