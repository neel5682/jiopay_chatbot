import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyCuvt31hJeflv7Hxlg9UOBZdlxEbOJ-2JM")

# Get and print available models
models = list(genai.list_models())
for model in models:
    print(model.name)
