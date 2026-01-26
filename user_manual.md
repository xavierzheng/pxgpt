# pxGPT User Manual - Plant Phenotyping with Large Language Models

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Complete Workflow](#complete-workflow)
4. [Command Reference](#command-reference)
5. [Provider Configuration](#provider-configuration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Introduction

**pxGPT** (Phenotype eXplorer GPT) is a command-line tool designed for large-scale plant phenotyping using Large Language Models (LLMs). It enables researchers to automatically analyze plant images and extract both descriptive and structured phenotypic data from germplasm collections.

### What is Plant Phenotyping?

Plant phenotyping is the comprehensive assessment of plant traits such as growth, development, tolerance, resistance, architecture, physiology, ecology, yield, and the basic functional units of a plant. pxGPT automates this process by:

1. **Analyzing plant images** with natural language descriptions
2. **Merging descriptions** from multiple samples
3. **Generating JSON schemas** that capture phenotypic variation
4. **Extracting structured data** using validated schemas

### Why Use pxGPT?

- **Scale**: Process hundreds or thousands of plant images automatically
- **Consistency**: Standardized analysis across your entire germplasm collection
- **Flexibility**: Support for multiple LLM providers (Anthropic, OpenAI, Google, Ollama, LM Studio)
- **Structured Output**: Generate both human-readable descriptions and machine-readable JSON
- **Cost Optimization**: Automatic prompt caching for Anthropic Claude reduces API costs

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd PlantGPT/script/ForGitHub

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
vim .env
```

If you want to use commercial LLM provider: At least one API key
```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here  
GOOGLE_API_KEY=your_google_key_here
```

### 3. First Analysis

```bash
# Analyze a folder of plant images
pxgpt analyze \
  --input-folder /path/to/plant/images \
  --output plant_descriptions.txt \
  --system-prompt system_prompt.txt \
  --prompt user_prompt.txt \
  --provider anthropic
```

---

## Complete Workflow

### Overview

The complete plant phenotyping workflow consists of four main steps:

```
Images → [analyze] → Descriptions → [merge] → Combined Report → [schema] → JSON Template → [analyze with schema] → Structured Data
```

### Step 1: Initial Analysis

Generate descriptive text for each plant image:

```bash
pxgpt analyze \
  --input-folder germplasm_images/cultivar_001 \
  --output results/cultivar_001_description.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt \
  --provider anthropic
```

**Input**: 
- Folder containing `.jpg` plant images
- System prompt defining the analysis context
- User prompt asking for specific phenotypic traits

**Output**: 
- Text file with `<report>...</report>` tagged descriptions

### Step 2: Merge Descriptions

Combine descriptions from multiple cultivars/samples:

```bash
# Create list of all sample names
ls germplasm_images/ > sample_names.txt

# Merge all descriptions
for i in $(cat sample_names.txt); do
    echo "# This is cultivar ${i}" >> combined_phenotypes.txt
    python extract_report_tags.py results/${i}_description.txt >> combined_phenotypes.txt
    echo " " >> combined_phenotypes.txt
    echo " " >> combined_phenotypes.txt
done
```

**Purpose**: Create a comprehensive document containing all phenotypic variation observed across your germplasm collection.

### Step 3: Generate JSON Schema (Manual GUI Approach)

**Why Use GUI Instead of CLI:** Creating comprehensive phenotyping schemas requires iterative refinement and deep analysis of phenotypic variation. Using GUI interfaces like claude.app, ChatGPT, or Gemini allows for:
- Extended thinking mode for complex schema design
- Multiple conversation rounds to build comprehensive coverage
- Real-time refinement and clarification
- Better handling of edge cases and rare phenotypes

**Recommended Approach:**

1. **Prepare your input:**
   - Use the `combined_phenotypes.txt` file from Step 2
   - This contains all phenotypic variation across your germplasm collection

2. **Choose your platform:**
   - **Claude.app** (Recommended): Enable extended thinking for deep analysis
   - **ChatGPT**: Use GPT-4 or newer for best results
   - **Gemini**: Google's AI platform

3. **Start the conversation with this prompt:**

```
You are a professional botanist and data scientist. Your task is based on the provided document, which contains phenotyping reports of individual cultivars, to generate a comprehensive JSON schema to cover all possible phenotypic descriptions. This will serve as the standard template for future vision-based LLM plant phenotyping, so use ontology-like terms and be as comprehensive as possible. 

Do not ignore any feature found in only one cultivar. They are all critical. For qualitative descriptions, transform them into ordinal catalog ontology using enum. For quantitative descriptions, standardize units. You have to list all possibilities in as much detail as possible.

[Paste your combined_phenotypes.txt content here]

Please use extended thinking to analyze all the phenotypic variation and create a comprehensive JSON schema.
```

4. **Iterative refinement:**
   - Review the generated schema carefully
   - Ask for specific additions: "Add more detailed leaf shape categories"
   - Request clarifications: "Standardize all measurement units to metric"
   - Build on previous responses: "Expand the stress indicators section"

5. **Save the final schema:**
   - Copy the final JSON schema to a file (e.g., `phenotype_schema.json`)
   - Validate JSON syntax using online validators
   - Test with a small dataset before full deployment

### Step 4: Structured Analysis

Apply the generated schema to extract structured phenotypic data:

```bash
pxgpt schema \
  --input-folder germplasm_images/cultivar_001 \
  --output structured_data/cultivar_001.json \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --schema phenotype_schema.json \
  --prompt prompts/extract_traits.txt \
  --provider anthropic
```

**Output**: Validated JSON with structured phenotypic measurements and classifications.

---

## Command Reference

### `pxgpt analyze`

Basic image analysis with descriptive text output.

**Syntax:**
```bash
pxgpt analyze --input-folder PATH --output FILE --prompt FILE --system-prompt FILE [OPTIONS]
```

**Required Arguments:**
- `--input-folder PATH`: Directory containing `.jpg` images
- `--output FILE`: Output file path for results
- `--prompt FILE`: User prompt file
- `--system-prompt FILE`: System prompt file

**Optional Arguments:**
- `--provider {anthropic,openai,google,ollama}`: LLM provider (default: anthropic)

**Example:**
```bash
pxgpt analyze \
  --input-folder /data/wheat_images \
  --output wheat_analysis.txt \
  --system-prompt system_wheat.txt \
  --prompt describe_morphology.txt \
  --provider anthropic
```

### `pxgpt schema`

Structured analysis with JSON schema validation.

**Syntax:**
```bash
pxgpt schema --input-folder PATH --output FILE --prompt FILE --system-prompt FILE --schema FILE [OPTIONS]
```

**Required Arguments:**
- `--input-folder PATH`: Directory containing `.jpg` images  
- `--output FILE`: Output JSON file path
- `--prompt FILE`: User prompt file
- `--system-prompt FILE`: System prompt file
- `--schema FILE`: JSON schema file for validation

**Optional Arguments:**
- `--provider {anthropic,openai,google,ollama}`: LLM provider (default: anthropic)

**Example:**
```bash
pxgpt schema \
  --input-folder /data/rice_images \
  --output rice_phenotypes.json \
  --system-prompt system_rice.txt \
  --prompt extract_traits.txt \
  --schema rice_schema.json \
  --provider anthropic
```

---

## Provider Configuration

### Anthropic Claude (Recommended for Research)

**Advantages:**
- Prompt caching reduces costs for repeated analysis
- Excellent vision capabilities
- Reliable structured output

**Configuration:**
```bash
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-7-sonnet-20250219
```

### OpenAI GPT-4/GPT-5

**Note:** GPT-5 models only support `temperature=1`

**Configuration:**
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-2025-08-07
```

### Google Gemini

**Configuration:**
```bash
GOOGLE_API_KEY=your_key_here  
GOOGLE_MODEL=gemini-2.5-pro
```

### Ollama (Local)

**Advantages:**
- No API costs
- Complete data privacy
- Works offline

**Setup:**
1. Install Ollama: https://ollama.ai/
2. Pull a vision model: `ollama pull gemma3:12b`
3. Configure:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
```

### LM Studio (Local)

**Setup:**
1. Install LM Studio: https://lmstudio.ai/
2. Load a vision-capable model
3. Start the local server
4. Configure as OpenAI provider:
```bash
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=your-loaded-model-name
```

**Usage:**
```bash
pxgpt analyze --provider openai --input-folder /data/images --output results.txt
```

---

## Best Practices

### Prompt Engineering for Plant Phenotyping

**System Prompt Template:**
```
You are an expert plant biologist specializing in [CROP] phenotyping. 
Analyze the provided plant images and describe the following traits:

MORPHOLOGICAL TRAITS:
- Plant height and architecture
- Leaf shape, size, and arrangement  
- Stem characteristics
- Root system (if visible)

DEVELOPMENTAL TRAITS:
- Growth stage
- Flowering status
- Fruit/seed development

STRESS INDICATORS:
- Disease symptoms
- Pest damage  
- Nutritional deficiencies
- Environmental stress

Format your response within <report></report> tags.
Be precise, consistent, and use standardized botanical terminology.
```

**User Prompt Example:**
```
Please analyze these [CROP] images and provide a comprehensive phenotypic description. 
Focus on traits that vary between genotypes and could be used for:
- Breeding selection
- Genetic mapping
- Stress tolerance assessment

Quantify measurements when possible (e.g., "leaves approximately 15cm long" rather than "long leaves").
```

### Image Organization

**Recommended Directory Structure:**
```
germplasm_collection/
├── cultivar_001/
│   ├── plant_001.jpg
│   ├── plant_002.jpg
│   └── plant_003.jpg
├── cultivar_002/
│   ├── plant_001.jpg
│   └── plant_002.jpg
└── cultivar_N/
    └── ...
```

### Schema Design

**Effective JSON Schema Structure:**
```json
{
  "type": "object",
  "properties": {
    "morphology": {
      "type": "object", 
      "properties": {
        "plant_height_cm": {"type": "number"},
        "leaf_length_cm": {"type": "number"},
        "leaf_shape": {"enum": ["oval", "lanceolate", "linear"]},
        "branching_pattern": {"enum": ["alternate", "opposite", "whorled"]}
      }
    },
    "development": {
      "type": "object",
      "properties": {
        "growth_stage": {"enum": ["vegetative", "flowering", "fruiting"]},
        "maturity_days": {"type": "number"}
      }
    },
    "stress_indicators": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

### Cost Optimization

**For Anthropic Claude:**
- Use prompt caching by keeping system prompts and schemas consistent
- Process images in batches with the same schema
- Estimated cost savings: 50-90% for repeated schema usage

**For All Providers:**
- Optimize image sizes
- Use specific, focused prompts to reduce output tokens
- Batch process multiple images per request when supported

---

## Troubleshooting

### Common Issues

**1. "Provider not configured" Error**
```
Error: Provider 'anthropic' not configured
```
**Solution:** Add your API key to `.env`:
```bash
ANTHROPIC_API_KEY=your_actual_key_here
```

**2. "No .jpg files found" Error**
```
Error: No .jpg files found in directory
```
**Solution:** 
- Ensure images are in `.jpg` format (not `.jpeg`, `.png`, etc.)
- Use absolute paths for input folders
- Check file permissions

**3. Rate Limit Errors**
```
Error: Rate limit exceeded
```
**Solution:**
- Wait for the automatic retry (60s for Anthropic)
- Reduce batch size
- Spread requests over time

**4. Schema Validation Failures**
```
Error: Response does not match schema
```
**Solution:**
- Simplify your JSON schema
- Make more properties optional
- Improve your prompts to be more specific about expected format

**5. GPU Memory Issues (Ollama/LM Studio)**
```
Error: CUDA out of memory
```
**Solution:**
- Use smaller models (e.g., `gemma3:7b` instead of `gemma3:12b`)
- Reduce image resolution
- Process fewer images simultaneously

### Debug Mode

Enable verbose logging:
```bash
export pxGPT_DEBUG=1
pxgpt analyze --verbose --input-folder /data/images --output results.txt
```

### Provider-Specific Issues

**Anthropic:**
- Ensure your API key has sufficient credits
- Check model availability in your region

**OpenAI:**
- GPT-5 models only support `temperature=1.0`
- Verify model name spelling

**Ollama:**
- Ensure Ollama service is running: `ollama serve`
- Verify model is downloaded: `ollama list`

---

## Advanced Usage

### Batch Processing Script

Process entire germplasm collections:

```bash
#!/bin/bash
# batch_phenotype.sh

GERMPLASM_DIR="/data/germplasm_collection"
OUTPUT_DIR="/results"
PROVIDER="anthropic"

mkdir -p "$OUTPUT_DIR"

# Step 1: Analyze all cultivars
for cultivar in $(ls "$GERMPLASM_DIR"); do
    echo "Processing $cultivar..."
    
    pxgpt analyze \
        --input-folder "$GERMPLASM_DIR/$cultivar" \
        --output "$OUTPUT_DIR/${cultivar}_description.txt" \
        --system-prompt system_phenotype.txt \
        --prompt describe_plant.txt \
        --provider "$PROVIDER"
done

# Step 2: Merge all descriptions
echo "Merging descriptions..."
for cultivar in $(ls "$GERMPLASM_DIR"); do
    echo "# Cultivar: $cultivar" >> "$OUTPUT_DIR/combined_phenotypes.txt"
    python extract_report_tags.py "$OUTPUT_DIR/${cultivar}_description.txt" >> "$OUTPUT_DIR/combined_phenotypes.txt"
    echo "" >> "$OUTPUT_DIR/combined_phenotypes.txt"
done

# Step 3: Generate schema (manual step - review combined_phenotypes.txt first)
echo "Review combined_phenotypes.txt and run schema generation manually"
echo "pxgpt analyze --input-folder sample_images --output schema.json --prompt combined_phenotypes.txt --system-prompt schema_system.txt"
```

### Custom Configuration Files

Create project-specific configurations:

```bash
# wheat_config.env
DEFAULT_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-7-sonnet-20250219
MAX_TOKENS=8192
TEMPERATURE=0.5

# Load custom config
set -a && source wheat_config.env && set +a
pxgpt analyze --input-folder wheat_images --output wheat_results.txt
```

### Integration with Research Pipelines

**With R for Statistical Analysis:**
```r
# Load structured phenotype data
library(jsonlite)
phenotypes <- fromJSON("structured_phenotypes.json")

# Perform GWAS, heritability analysis, etc.
# ...
```

**With Python for Machine Learning:**
```python
import json
import pandas as pd

# Load and process phenotype data
with open("structured_phenotypes.json") as f:
    data = json.load(f)

df = pd.json_normalize(data)
# Continue with ML pipeline...
```

---

## Support and Contributing

### Getting Help

1. Check this user manual first
2. Review the [README.md](README.md) for installation issues
3. Enable debug mode for detailed error messages
4. Check provider-specific documentation

### Reporting Issues

When reporting bugs, please include:
- Complete error message
- pxGPT version
- Provider and model used  
- Sample images (if possible)
- Configuration (remove API keys)

### Citation

If you use pxGPT in your research, please cite:
```
[Your citation format here]
```

---

**pxGPT** - Empowering plant research through automated phenotyping with Large Language Models.
