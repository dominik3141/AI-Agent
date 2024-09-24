# AI Assistant with Tool Integration

This project implements an AI assistant powered by OpenAI's GPT models, capable of using various tools to enhance its capabilities. The assistant can execute Python code, run shell commands, perform web searches, and delegate tasks to "intern" AI models.

## Features

- Interaction with OpenAI's GPT models
- Execution of Python code
- Running of shell commands
- Web search capabilities using Bing Search API
- Integration with Perplexity.ai for additional research
- Task delegation to "intern" AI models
- Conversation history visualization

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install openai requests
   ```
3. Set up the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PERPLEXITY_API_KEY`: Your Perplexity.ai API key
   - `BING_SEARCH_API_KEY`: Your Bing Search API key

## Usage

Run the main script with a query:

```
python main.py -q "Your query here"
```

The conversation will be saved as a JSON file in the `conversations` folder.

## Visualization

To visualize the conversation:

1. Open the `visualizer.html` file in a web browser
2. Use the file input to select a JSON conversation file
3. The conversation will be displayed with statistics and formatted messages

## Project Structure

- `main.py`: Main script containing the AI assistant logic
- `tools.py`: Contains functions for Perplexity and Bing search integrations
- `visualizer.html`: HTML file for visualizing conversations

