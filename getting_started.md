# Getting Started with Hybrid Translation Demo

This guide will help you set up and run the Hybrid Translation Demo system quickly.

## Prerequisites

- Python 3.7+ installed
- pip package manager
- API keys for at least one of:
  - OpenAI or Azure OpenAI
  - Google Cloud Translation
  - DeepL (optional)

## Installation Steps

1. **Clone the repository and navigate to the project folder**

```bash
git clone https://github.com/yourusername/hybrid-translation-demo.git
cd hybrid-translation-demo
```

2. **Create and activate a virtual environment**

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up API keys**

Copy the example environment file:

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
- For Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_BASE`, etc.
- For OpenAI: `OPENAI_API_KEY`
- For Google: Set path to your Google credentials JSON file
- For DeepL: `DEEPL_API_KEY`

## Running the Demo

### Interactive Mode

The easiest way to get started is with interactive mode:

```bash
python translate_demo.py --interactive
```

This will prompt you for:
- Text to translate
- Target language
- Translation mode (hybrid, llm, google, or deepl)

### Command Line Options

Translate text directly:

```bash
python translate_demo.py --text "Hello, how are you?" --language Japanese
```

Translate from a file:

```bash
python translate_demo.py --file samples/math_problem.txt --language Spanish --mode hybrid
```

Save the translation to a file:

```bash
python translate_demo.py --text "Hello world" --language French --save
```

### Sample Examples

Run the examples script to see translations of sample texts:

```bash
python examples.py
```

## Customizing Prompts

You can customize the translation prompts by editing the JSON files in the `prompts` directory:

- For LLM translation: `prompts/math/llm.json`
- For hybrid translation: `prompts/math/hybrid.json`

## Troubleshooting

- **"No module named 'X'"**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

- **"No API key found"**: Check that your `.env` file contains the necessary API keys

- **Language detection issues**: Make sure `langdetect` is installed with `pip install langdetect`

- **Request errors**: Check your internet connection and API key validity

## Next Steps

- Try different types of text (technical, conversational, mathematical)
- Compare translation quality between different modes
- Customize prompts for your specific use case

For more information, refer to the [README.md](README.md) file.