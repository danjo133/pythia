# pythia.py

This Python script provides a command-line interface to interact with OpenAI's GPT-4 model. It allows you to send text prompts to the model and receive generated responses. It also supports storing and loading conversations, manipulating the conversation context, and even generating images using DALL-E.

## Features

- Interact with GPT-4 model via a command-line interface.
- Store and load conversation contexts.
- Add and remove lines from the current conversation.
- Generate images using DALL-E.
- Ask GPT-4 about something in an image.
- Store the conversation as a file.
- Set the GPT model.

## Usage

Run the script in a Python environment where the `openai`, `readline`, and `rich` libraries are installed. You will need to set the `OPENAI_API_KEY` environment variable to your OpenAI API key.

```bash
./pythia.py
```

Once the script is running, you can type your prompts directly into the console. The script also provides several commands for manipulating the conversation context and interacting with the GPT-4 model. These commands are prefixed with an underscore (`_`). For example, to send the current conversation to the model, use the `_send` command.

For a full list of commands, use the `_help` command.

## Requirements

- Python 3.6 or later
- openai
- readline
- rich
- python-dotenv

The python requirements are best managed with pipenv.

## Note

This script is intended for educational and experimental purposes. It is not designed for production use.
