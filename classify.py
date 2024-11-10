import requests
import base64

# Configure Ollama API
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "x/llama3.2-vision:latest"  # Replace with your specific model name if different

def classify_document(image_path):
    # Read and encode the image in base64
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # # Create the prompt to guide the model's classification
    # prompt = ("Classify the document type from the following categories: "
    #           "Invoice, Resume, Contract, ID Card, Bank Statement. "
    #           "Return only the category in JSON format.")

    # Create the prompt to guide the model's classification
    prompt = ("Classify the document type "
              "Return only the category in JSON format.")
    
    # Request payload
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],  # Add the image data
        "format": "json",          # Ensure output is in JSON format
        "stream": False
    }

    # Make a POST request to the Ollama API
    response = requests.post(OLLAMA_API_URL, json=data)
    response_json = response.json()

    # Parse the JSON response to extract document type
    if response_json.get("done"):
        output_json = response_json.get("response")
        if output_json:
            try:
                document_info = eval(output_json)
                return document_info.get("category")  # Example key for parsed document type
            except (SyntaxError, KeyError):
                print("Unexpected JSON format in response.")
                print(output_json)
    return "Unknown document type"

# Example usage
image_path = "/Users/internalis/Documents/SortiFile/Data/invoive1.png"  # Replace with your document image path
document_type = classify_document(image_path)
print("Document Type:", document_type)
