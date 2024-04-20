#!/usr/bin/env python

import cmd
from openai import OpenAI
import groq

import os
import readline
from rich.console import Console
from rich.markdown import Markdown
import requests
import base64
import argparse

from dotenv import load_dotenv
load_dotenv()

class DB():
    stored = {}
    current = []
    system = ["You are a helpful assistant with good knowledge about programming and security."]
    assistant = ["The user is a skilled programmer that likes in depth technical answers."]

    def add_line(self,line):
        self.current.append(["user",line])

    def add_system_line(self, line):
        self.system.append(line)

    def add_assistant_line(self, line):
        self.assistant.append(line)

    def add_response(self, line):
        self.current.append(["agent",line])

    def get_lines(self):
        return self.current

    def get_pretty_lines(self):
        output = [line[0] + ":\n\n" + line[1] + "\n\n" for line in self.current]
        return output

    def get_system(self):
        return self.system

    def get_assistant(self):
        return self.assistant

    def get_topics(self):
        return self.stored.keys()

    def remove_line(self, lineno):
        try:
            del self.current[lineno]
        except:
            print("Could not delete line",lineno)

    def store(self, topic):
        self.stored[topic] = self.current.copy()

    def load(self, topic):
        self.current = self.stored[topic].copy()

    def clear_current(self):
        self.current = []

    def clear_topics(self):
        self.stored = {}

    def clear_system(self):
        self.system = []

    def clear_assistant(self):
        self.assistant = []

    def remove_topic(self, topic):
        del self.stored[topic]

class GPT(cmd.Cmd):
    db = DB()
    intro = "PythiaGPT - Ask me anything!\n"

    model= None
    api = None

    prompt = None
    console = Console()
    
    # If true, the user can input multiple lines before sending.
    send_disabled = False
    lock_line = False
    temp_line = ""

    client = None

    histfile = '.gpt_chat_history'
    histfile_size = 1000

    def __init__(self, query=None, model="gpt-4-1106-preview", base_url=None, api="OpenAI"):
        super().__init__()
        if model:
            self.model = model
        self.api = api

        self.prompt = f"{self.model.upper()}[{self.api}] > "
        if self.api == "OpenAI":
            if base_url:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
            else:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.api == "Groq":
            api_key=os.getenv("GROQ_API_KEY")
            self.client = groq.Client(api_key=api_key)

        if query:
            self.intro = None
            self.do__add_line(query)
            self.do__send("")

    def add_file(self, line):
        with open(line, "r") as f:
            file_content = f.read()
            self.db.add_line(file_content)
            self.console.print(file_content)


    def make_request(self, system_lines, assistant_lines, lines, model="gpt-4"):
        messages = []
        messages.extend([{"role": "system", "content": line} for line in system_lines])
        messages.extend([{"role": "assistant", "content": line} for line in assistant_lines])
        messages.extend([{"role": "user", "content": line} for line in lines])
        response = self.client.chat.completions.create(model=model,
        messages=messages)
        return response.choices[0].message.content

    def generate_image(self, prompt: str, quality="standard", model="dall-e-3", size="1024x1024"):
        # Always openai
        client = OpenAI()
        response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
        )

        image_url = response.data[0].url
        print(image_url)

        r = requests.get(image_url, allow_redirects=True, timeout=20)
        with open('image.png', 'wb') as f:
            f.write(r.content)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def ask_image(self, prompt: str, image_path: str, detail="low", model="gpt-4-vision-preview"):
        # always openai
        client = OpenAI()
        url = ""
        if "http" in image_path:
            url = image_path
        else:
            base64_image = self.encode_image(image_path)
            if "jpg" in image_path:
                url = f"data:image/jpeg;base64,{base64_image}"
            elif "jpeg" in image_path:
                url = f"data:image/jpeg;base64,{base64_image}"
            elif "png" in image_path:
                url = f"data:image/png;base64,{base64_image}"

        with open ("debug.txt","w") as f:
            f.write(url)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                        },
                    },
                ],
                }
            ],
            max_tokens=300,
        )
        print(response.choices[0].message.content)

    def preloop(self):
        if readline and os.path.exists(self.histfile):
            readline.read_history_file(self.histfile)

    def postloop(self):
        if readline:
            readline.set_history_length(self.histfile_size)
            readline.write_history_file(self.histfile)

    def default(self, line):
        if self.lock_line:
            self.temp_line += line + "\n"
        else:
            self.db.add_line(line)
            if not self.send_disabled:
                self.do__send("")
            else:
                self.do__list_current("")

    def do_exit(self, _):
        print("Exiting..")
        return True

    def do__disable_send(self, _):
        self.send_disabled = not self.send_disabled
    
    def do__lock_line(self, _):
        if len(self.temp_line) > 0:
            self.db.add_line(self.temp_line)
            self.temp_line = ""
        self.lock_line = not self.lock_line

    def do__send(self, _):
        lines = self.db.get_pretty_lines()
        try:
            response = self.make_request(
                self.db.get_system(),
                self.db.get_assistant(),
                lines,
                self.model)
            self.db.add_response(response)

            self.console.print(Markdown(response))

        except Exception as e:
            print(e)
            print("Error in request, please try again")

        print()

    def do__list_current(self, line):
        for idx, line in enumerate(self.db.get_lines()):
            print(idx,":", line)

    def do__rm_line(self, line):
        try:
            line_no = int(line)
            self.db.remove_line(line_no)
        except:
            print(line,"Not a number")

    def do__rm_topic(self, topic):
        self.db.remove_topic(topic)

    def do__store(self, topic):
        self.db.store(topic)

    def do__load(self, topic):
        self.db.load(topic)

    def do__push(self, topic):
        self.db.store(topic)
        self.db.clear_current()

    def do__pop(self, topic):
        self.db.load(topic)
        self.db.remove_topic(topic)

    def do__clear_current(self, _):
        self.db.clear_current()

    def do__clear_topics(self, _):
        self.db.clear_topics()

    def do__list_topics(self, line):
        for idx,line in enumerate(self.db.get_topics()):
            print(idx,":",line)

    def do__clear_system(self, _):
        self.db.clear_system()

    def do__clear_assistant(self, _):
        self.db.clear_assistant()

    def do__add_system(self, line):
        self.db.add_system_line(line)

    def do__add_assistant(self, line):
        self.db.add_assistant_line(line)

    def do__list_system(self, _):
        for idx, line in enumerate(self.db.get_system()):
            print(idx,":", line)

    def do__list_assistant(self, _):
        for idx, line in enumerate(self.db.get_assistant()):
            print(idx,":", line)

    def do__add_file(self, line):
        self.add_file(line)

    def do__store_conversation(self, filename):
        with open(filename, "w") as f:
            f.write("\n".join(self.db.get_pretty_lines()))

    def do__store_message(self, line):
        [idx, filename] = line.split(" ")
        lines = self.db.get_lines()
        if len(filename) and 0 <= int(idx) < len(lines):
            with open(filename, "w") as f:
                f.write(lines[int(idx)][1])

    def do__add_line(self, line):
        self.db.add_line(line)

    def do__set_model(self, line):
        self.model = line

    def do__get_model(self, _):
        print(self.model)

    def do__cost(self, _):
        num_characters = 0
        for line in self.db.get_lines():
            num_characters = num_characters + len(line[1])
        # Approx 5 characters per token, 0.1kr per 1000 tokens
        print(f"num_chars: {num_characters}, num_tokens: {num_characters/5}, cost: {0.1 * num_characters/5000}kr")

    def do__generate_image(self, prompt):
        self.generate_image(prompt)

    def do__ask_image(self, line):
        filename = line.split(" ")[0]
        prompt = " ".join(line.split(" ")[1:])
        self.ask_image(prompt,filename)

    def do_EOF(self, _):
        print("\nExiting..")
        return True

    def do__help(self, _):
        print("<line>                    - ask a question")
        print("System commands:")
        print("_help                     - Print this help")
        print("_store          <topic>   - stores current conversation under topic")
        print("_load           <topic>   - makes topic current conversation")
        print("_push           <topic>   - stores current conversation under topic, clears current")
        print("_pop            <topic>   - makes topic current conversation, removes stored topic")
        print("_clear_current            - clears current conversation")
        print("_clear_topic    <topic>   - clears topic")
        print("_clear_system             - clears system config")
        print("_clear_assistant          - clears assistant config")
        print("_disable_send             - disables sending after each line for composing multiline messages")
        print("_lock_line               - don't create a new line on enter allowing to paste multiline for example")
        print("_add_line       <line>    - add line to context")
        print("_add_system     <line>    - add line to system config")
        print("_add_assistant  <line>    - add line to assistant config")
        print("_list_topics              - lists available topics")
        print("_list_current             - lists current conversation")
        print("_list_system              - lists the system prompt")
        print("_list_assistant           - lists the assistant prompt")
        print("_send                          - sends the conversation")
        print("_generate_image <prompt>  - Generate an image from the prompt, download to image.png")
        print("_ask_image <filename> <question> - Ask gpt4 about something in the image")
        print("_store_conversation <filename> - stores the conversation as a file with filename")
        print("_store_message <id> <filename> - stores the message at id as a file with filename")
        print("_rm             <lineno>  - removes line number from current conversation")
        print("_add_file      <filename>- read entire file into one line")
        print("_set_model      <model>   - set gpt model, default is gpt-3.5-turbo")
        print("_get_model                - get current gpt model")



def run_webserver(gpt, host="0.0.0.0", port=5000):
    # start webserver
    from flask import Flask, request
    app = Flask(__name__)

    @app.route('/help')
    @app.route('/cheat')
    def print_cheat():
        cheat = """curl -X POST -H "content-type: application/x-www-form-urlencoded" -d "query=Create a python script for google searches" http://localhost:5000/ask
        curl -F"file=@LICENSE" http://localhost:5000/add_file
        """
        return cheat
    @app.route('/ask', methods=['POST'])
    def ask():
        line = request.form['query']
        gpt.do__add_line(line)
        response = gpt.make_request(
            gpt.db.get_system(),
            gpt.db.get_assistant(),
            gpt.db.get_pretty_lines(),
            gpt.model)
        gpt.db.add_response(response)
        return response
    @app.route('/add_file', methods=['POST'])
    def add_file():
        file = request.files['file']
        filename = file.filename
        lines = file.stream.readlines()
        query_line = f"{filename}:\n" + "".join([line.decode("utf-8") for line in lines])
        gpt.do__add_line(query_line)
        return "file added"

    @app.route('/')
    def hello_world():
        return 'Hello, World!'
    app.run(host=host,port=port)


if __name__ == '__main__':
    # Parse arguments with argparse, if the program is called with the flag --query, run the query and exit
    parser = argparse.ArgumentParser(description='Pythia answers your questions')
    parser.add_argument('--query', '-q', help='Query to initialize the dialogue with', required=False)
    parser.add_argument('--file', '-f', help='Add file before query', required=False)
    parser.add_argument('--oneshot','-o', help='Run a single query and exit', required=False, action='store_true')
    parser.add_argument('--api', '-a', help='API to use, supported: OpenAI, Groq. default is OpenAI', required=False,default="OpenAI")
    parser.add_argument('--model', '-m', help='Model to use, suggested: gpt-3.5-turbo gpt-4 gpt-4-32k gpt-4-1106-preview, llama3-8b-8192, llama3-70b-8192 ', required=False)
    parser.add_argument('--debug', '-d', help='Debug mode', required=False, action='store_true')
    parser.add_argument('--base-url', '-b', help='Run against different llm, for example: http://127.0.0.1:1234/v1 to run against localhost', required=False)
    parser.add_argument('--webserver', '-w', help='Start webserver', required=False, action='store_true')
  
    args = parser.parse_args()

    if args.base_url:
        if not args.model:
            print("You need to specify a model when using a custom base url")
            os.exit(1)

    if args.debug:
        print("Debug mode enabled")
        print(args)

    try:
        if args.oneshot:
            gpt = GPT(model=args.model,base_url=args.base_url, api=args.api)
            if args.file:
                gpt.do__add_file(args.file)
            gpt.default(args.query)
        elif args.webserver:
            gpt = GPT(model=args.model,base_url=args.base_url, api=args.api)
            run_webserver(gpt=gpt)
        else:
            GPT(query=args.query, model=args.model,base_url=args.base_url, api=args.api).cmdloop()
    except KeyboardInterrupt:
        print("\nExiting..")
