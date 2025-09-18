import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr 

load_dotenv(override=False)
google_api_key = os.getenv("GOOGL_API_KEY")


class Website:
    url:str
    title:str
    text:str
    
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script","style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)
        
    def get_content(self):
        return f"Webpage Title:\n{self.title}\nWebpage Content:\n{self.text}\n\n"
    
# With massive thanks to Bill G. who noticed that a prior version of this had a bug! Now fixed.

system_message = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."

def stream_gemini(prompt):
    messages = [
        {"role": "system", "content":system_message},
        {"role":"user", "content":prompt}
    ]
    gemini_via_openai_client = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    try:
        response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        stream=True
        )
        
        result = ""
        for chunk in response:
            if chunk.choices:
                # Safely get the content attribute, default to an empty string if it doesn't exist
                content = getattr(chunk.choices[0].delta, 'content', "")
                if content:
                    result += content
                    yield result
    except Exception as e:
        yield f"An error occurred: {e}"  
        
          
def stream_brochure(company_name, url):
    yield ""
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_content()
    result = stream_gemini(prompt)
    yield from result
    
    
view = gr.Interface(
    fn = stream_brochure,
    inputs=[
        gr.Textbox(label="Company name;"),
        gr.Textbox(label="Landing page URL including http:// or https://"),
    ],
    outputs=[gr.Markdown(label="Brochure")],
    flagging_mode="never"
)
view.launch()