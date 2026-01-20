from google import genai
from google.genai.types import HttpOptions
import PyPDF2 as pdf
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Shared client (configured via `configure_genai`) to avoid recreating per-call
_GENAI_CLIENT = None

def configure_genai(api_key: str, api_version: str = "v1"):
    """Configure a shared genai client used by `get_gemini_response`.

    Call this once with a valid `api_key` (e.g. from .env) before generating.
    """
    global _GENAI_CLIENT
    if not api_key:
        raise ValueError("API key must be provided to configure genai client")

    _GENAI_CLIENT = genai.Client(
        api_key=api_key,
        http_options=HttpOptions(api_version=api_version)
    )


def get_gemini_response(prompt):
    """Generate a response using the current STABLE Gemini 2.5 model."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize client with explicit stable API version v1
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1")
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        if not response or not response.text:
            raise Exception("Empty response received from Gemini")
            
        # Extract JSON from the response text
        import re
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response.text, re.DOTALL)
        if match:
            return match.group()
        return response.text
                
    except Exception as e:
        raise Exception(f"Model Error: {str(e)}")

def extract_pdf_text(uploaded_file):
    """Extract text from PDF with error handling."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)
                
        if not text:
            raise Exception("No text could be extracted. Ensure the PDF is not a scanned image.")
            
        return " ".join(text)
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def prepare_prompt(resume_text, job_description):
    """Prepare the input prompt for the ATS evaluation."""
    if not resume_text or not job_description:
        raise ValueError("Resume text and job description cannot be empty")
        
    prompt_template = """
    Act as an expert ATS (Applicant Tracking System) specialist.
    Evaluate the following resume against the job description.
    
    Resume: {resume_text}
    Job Description: {job_description}
    
    Provide a response in this JSON format ONLY:
    {{
        "JD Match": "percentage between 0-100",
        "MissingKeywords": ["keyword1", "keyword2"],
        "Profile Summary": "detailed analysis and suggestions"
    }}
    """
    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip()
    )

def configure_genai(api_key):
    """Old function kept for compatibility with app.py imports."""

    pass
