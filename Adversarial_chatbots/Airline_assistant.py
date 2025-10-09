import os
import json
import requests
import gradio as gr

# --- Configuration ---
# URL for your local Ollama API
OLLAMA_API_URL = "http://localhost:11434/api/chat"
# The model you want to use
MODEL = "phi3:latest" # Note: Not all models support tool calling well. You may need a model fine-tuned for function calling.
# Headers for the API request
HEADERS = {"Content-type": "application/json"}

# --- System Prompt ---
# This message defines the chatbot's persona and instructions.
# We instruct the model to return a specific JSON format for tool calls.
system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."
system_message += "To get a ticket price, you must call a tool by responding with ONLY the following JSON format. "
system_message += """
[EXAMPLE]
User: What is the price for a flight to tokyo?
Assistant: {"tool_name": "get_ticket_price", "arguments": {"destination_city": "tokyo"}}
[/EXAMPLE]
"""

# --- Tools Definition ---
ticket_prices = {"london": "$799", "paris": "$699", "new york": "$399", "tokyo": "$999", "berlin": "$499"}

def get_ticket_price(destination_city: str) -> str:
    """Gets the ticket price for a given destination."""
    print(f"Getting ticket price for {destination_city}")
    if not destination_city:
        return "an unknown destination"

    normalized_city = destination_city.lower()
    # Loop through our known cities
    for city_key in ticket_prices.keys():
        # Check if the model's output CONTAINS one of our cities
        if city_key in normalized_city:
            print(f"Found match: '{city_key}'. Returning price.")
            return ticket_prices[city_key]

    print("No match found in the dictionary.")
    return "an unknown destination"

def handle_tool_call(tool_call_json: dict) -> dict:
    """Handles the tool call based on the JSON provided by the model."""
    tool_name = tool_call_json.get("tool_name")
    
    if tool_name == "get_ticket_price":
        arguments = tool_call_json.get("arguments", {})
        city = arguments.get("destination_city")
        price = get_ticket_price(city)
        
        # We'll use the 'user' role to feed the tool's response back to the model.
        # This is more compatible than the 'tool' role for many models.
        return {
            "role": "user",
            "content": f"Tool response for get_ticket_price(destination_city='{city}'): The price is {price}"
        }
    return {"role": "user", "content": "Tool not found or invalid tool name."}


def chat(message: str, chat_history: list):
    """
    Handles the chat logic, including prompt-based tool calls to the Ollama API.
    """
    # 1. Prepare message history and initial API call
    messages = [{"role": "system", "content": system_message}] + chat_history
    messages.append({"role": "user", "content": message})

    # The `tools` parameter is removed as it causes the 400 error.
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        # First API call to the model
        resp = requests.post(OLLAMA_API_URL, headers=HEADERS, data=json.dumps(payload))
        resp.raise_for_status()
        response_json = resp.json()
        response_message = response_json.get("message", {})
        bot_message_content = response_message.get("content", "").strip()

    except requests.exceptions.RequestException as e:
        print(f"Error contacting model server: {e}")
        return "Error: Could not connect to the model server. Please ensure Ollama is running."
    except ValueError:
        return "Error: The model server returned an invalid response."

    # 2. Check if the model's response is a JSON for a tool call
    try:
        tool_call_data = json.loads(bot_message_content)
        if isinstance(tool_call_data, dict) and "tool_name" in tool_call_data:
            # It's a tool call, so we handle it.
            print(f"Detected tool call: {tool_call_data}")
            
            # Add the model's tool request to history
            messages.append({"role": "assistant", "content": bot_message_content})
            
            # Execute the tool and get the result
            tool_response = handle_tool_call(tool_call_data)
            messages.append(tool_response)
            
            # Make a second call to the model with the tool's result to get a natural response
            second_payload = { "model": MODEL, "messages": messages, "stream": False }
            
            second_resp = requests.post(OLLAMA_API_URL, headers=HEADERS, data=json.dumps(second_payload))
            second_resp.raise_for_status()
            
            second_response_json = second_resp.json()
            final_message = second_response_json.get("message", {}).get("content", "")
            return final_message
        else:
            # It's JSON, but not a valid tool call format, so return as is.
            return bot_message_content
            
    except (json.JSONDecodeError, TypeError):
        # The response is not JSON, so it's a regular text response.
        return bot_message_content


if __name__ == "__main__":
    chatbot_interface = gr.ChatInterface(
        fn=chat,
        title="FlightAI Assistant",
        description="Ask me questions about your flights, like ticket prices!",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="E.g., How much is a ticket to Paris?", container=False, scale=7),
        theme="soft",
        type="messages"
    )
    
    print("Launching Gradio Chat UI...")
    chatbot_interface.launch()

