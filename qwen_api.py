from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai
import os
import json
# import yaml

import random
from http import HTTPStatus
from dashscope import Generation
import dashscope
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes

# Load environment variables from .env file
load_dotenv()

#Model Studio endpoint
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

@app.route('/api/chat', methods=['POST'])
def query_endpoint():
    try:
        # Parse JSON payload from the request
        payload = request.get_json()
        messages = payload.get('messages', [])
        temperature = payload.get('temperature', 0.4)
        max_output_tokens = payload.get('max_output_tokens', 1000)

        # Format messages (you can replace this with your actual logic)
        data = format_messages(messages)
        messages = data['messages']

        print("Received messages:", messages)

        # Use a generator to stream responses back to the frontend
        def generate_responses():
            yield from inference_loop(messages)

        # Return a streaming response with the correct content type
        return Response(generate_responses(), content_type='text/event-stream')

    except Exception as e:
        # Handle errors gracefully
        return {"error": str(e)}, 400


def inference_loop(messages):
    while True:
        ####### MODEL STUDIO ##########################
        response = Generation.call(
            model="qwen-max-latest",
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message'
        )

        if response.status_code != HTTPStatus.OK:
            error_message = f"Request id: {response.request_id}, Status code: {response.status_code}, " \
                            f"Error code: {response.code}, Error message: {response.message}"
            print(error_message)
            yield json.dumps({'error': 'Model inference failed.', 'details': error_message}) + "\n"
            break

        # Extract the assistant's response
        assistant_response = response.output.choices[0].message.content
        print("Assistant Response:", assistant_response)

        # Add the assistant's response to the messages list
        messages.append({"role": "assistant", "content": assistant_response})

        # Stream the assistant's response back to the frontend
        yield json.dumps({'role': 'assistant', 'content': assistant_response}) + "\n"

        # Check if the response contains a tool call
        tool_call_data = None
        try:
            tool_call_data = parse_tool_call(assistant_response)
        except ValueError as e:
            print(f"No valid tool call found: {e}")

        if tool_call_data:
            # Stream the tool call message back to the frontend
            yield json.dumps({'role': 'tool_call', 'content': f"Tool call: {tool_call_data}"}) + "\n"

            # Execute the tool with the provided parameters
            tool_name = tool_call_data["name"]
            tool_input = tool_call_data.get("input", {})
            print(f"Executing tool: {tool_name} with input: {tool_input}")
            
            # Assume `execute_tool` is a predefined function
            tool_result = execute_tool(tool_name, tool_input)

            # Add the tool result as a "user" message in the conversation
            tool_message = f"Tool result: {tool_result}"
            messages.append({"role": "user", "content": tool_message})
            print(f"Tool executed. Result: {tool_result}")

            # Stream the tool result back to the frontend
            yield json.dumps({'role': 'tool_call', 'content': tool_message}) + "\n"
        else:
            # If no tool call, terminate the loop
            break

def format_messages(messages):
    model = ''
    endpoint = ''

    tools_available = get_tools_available()
    tools_format = get_tools_format()
    print(tools_available)
    print(tools_format)
    system_prompt = f"""You are Qwen-Max, an advanced AI model. You will assist the user with tasks, using tools available to you.

You have the following tools available:
{tools_available}

{tools_format}

"""
    system_message = {"role": "system", "content": system_prompt}
    messages.insert(0, system_message)

    return {'messages': messages, 'model': model, 'endpoint': endpoint } 

def get_tools_available():
    tools_available = """
-get-cwd: Get the current working directory
    Parameters: None. This tool does not need a parameter.
    Returns: String - information about the current working directory

-read-file: Read a file in the filesystem
    Parameters:
    - path (required, string): path and filename of the file to read 
    Returns: String - the contents of the file specified in `path`

-write-file: Write content to a file in the filesystem
    Parameters:
    - path (required, string): path and filename of the file to write
    - content (required, string): the content to write to the file
    Returns: String - confirmation message indicating success or failure

-create-directory: Create a new directory in the filesystem
    Parameters:
    - path (required, string): path of the directory to create
    Returns: String - confirmation message indicating success or failure

-list-directory: List the contents of a directory in the filesystem
    Parameters:
    - path (optional, string): path of the directory to list. If not provided, lists the current working directory.
    Returns: String - a list of files and directories in the specified path
"""
    return tools_available



def get_tools_format():
    
    tools_format = """

When you want to use a tool, make a tool call (no explanations) using this exact format:

```
[[qwen-tool-start]]
{{
    "name": "tool_name",
    "input": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
[[qwen-tool-end]]
```

Note that the triple backticks (```) are part of the format!

Example 1:
************************
User: What is your current working directory?
Qwen-Max:
```
[[qwen-tool-start]]
{{
    "name": "get-cwd",
    "input": ""
}}
[[qwen-tool-end]]
```
**********************


Example 2:
************************
User: List the files in your current working directory.
Qwen-Max:
```
[[qwen-tool-start]]
{{
    "name": "list-directory",
    "input": {{
        "path": "."
    }}
}}
[[qwen-tool-end]]
```
**********************

Immediately end your response after calling a tool and the final triple backticks.

After receiving the results of a tool call, do not parrot everything back to the user.
Instead, just briefly summarize the results in 1-2 sentences.

"""
    return tools_format

def parse_tool_call(response):
    """
    Parses the tool call information from an LLM response.
    
    Args:
        response (str): The LLM's response containing the tool call.
        
    Returns:
        dict: A dictionary containing the tool name and input parameters.
              Example: {"name": "tool_name", "input": {"param1": "value1", "param2": "value2"}}
              
    Raises:
        ValueError: If the tool call format is invalid or cannot be parsed.
    """
    # Define markers for the tool call block
    start_marker = "[[qwen-tool-start]]"
    end_marker = "[[qwen-tool-end]]"
    
    try:
        # Extract the JSON block between the markers
        start_index = response.find(start_marker) + len(start_marker)
        end_index = response.find(end_marker)
        
        if start_index == -1 or end_index == -1:
            raise ValueError("Tool call markers not found in the response.")
        
        tool_call_block = response[start_index:end_index].strip()
        
        # Parse the JSON content
        tool_call_data = json.loads(tool_call_block)
        
        # Validate the structure of the tool call
        if "name" not in tool_call_data:
            raise ValueError("Tool call must include a 'name' field.")
        
        return tool_call_data
    
    except json.JSONDecodeError as e:
        print(f"Failed to parse tool call JSON: {e}. Please make sure the tool call is valid JSON")


def execute_tool(tool_name, tool_input):
    """
    Executes the specified tool with the given input parameters.

    Args:
        tool_name (str): The name of the tool to execute.
        tool_input (dict): A dictionary containing the input parameters for the tool.

    Returns:
        str: The result of the tool execution.

    Raises:
        ValueError: If the tool_name is invalid or the tool function raises an error.
    """
    # Map tool names to their respective functions
    tool_functions = {
        "get-cwd": get_cwd,
        "read-file": read_file,
        "write-file": write_file,
        "create-directory": create_directory,
        "list-directory": list_directory,
    }

    # Check if the tool exists
    if tool_name not in tool_functions:
        raise ValueError(f"Unknown tool: {tool_name}")

    # Retrieve the tool function
    tool_function = tool_functions[tool_name]

    try:
        # Execute the tool function with the provided input
        result = tool_function(**tool_input)
        return result
    except Exception as e:
        raise ValueError(f"Error executing tool '{tool_name}': {e}")


import os

def get_cwd():
    """
    Get the current working directory.
    
    Returns:
        str: The current working directory path.
    """
    try:
        return os.getcwd()
    except Exception as e:
        return f"Error getting current working directory: {e}"

def read_file(path):
    """
    Read the contents of a file.
    
    Args:
        path (str): The path to the file to read.
        
    Returns:
        str: The contents of the file, or an error message if reading fails.
    """
    try:
        with open(path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path, content):
    """
    Write content to a file.
    
    Args:
        path (str): The path to the file to write.
        content (str): The content to write to the file.
        
    Returns:
        str: A confirmation message, or an error message if writing fails.
    """
    try:
        with open(path, 'w') as file:
            file.write(content)
        return f"File written successfully: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def create_directory(path):
    """
    Create a new directory.
    
    Args:
        path (str): The path of the directory to create.
        
    Returns:
        str: A confirmation message, or an error message if creation fails.
    """
    try:
        os.makedirs(path, exist_ok=True)  # `exist_ok=True` ensures no error if the directory already exists
        return f"Directory created successfully: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error creating directory: {e}"

def list_directory(path="."):
    """
    List the contents of a directory.
    
    Args:
        path (str): The path of the directory to list. Defaults to the current directory (".").
        
    Returns:
        str: A list of files and directories, or an error message if listing fails.
    """
    try:
        contents = os.listdir(path)
        return f"Contents of directory '{path}': {', '.join(contents)}"
    except FileNotFoundError:
        return f"Directory not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


if __name__ == '__main__':
    app.run(debug=True, port="5001")
