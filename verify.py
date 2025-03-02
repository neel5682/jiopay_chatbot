import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyCuvt31hJeflv7Hxlg9UOBZdlxEbOJ-2JM")

# Use an available model
model = genai.GenerativeModel("gemini-1.5-pro-latest")
response = model.generate_content("Hello, world!")
print(response.text)
