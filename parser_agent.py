import requests
import zipfile
import io
import os
import time
import streamlit as st
import tempfile
import shutil
import json
from openai import OpenAI
import base64
from llm import LLMCaller, get_text_prompt
from utils import read_md, doi_encode, doi_decode

def parser_pdf(pdf_path, token=None, output_dir=None):
    """
    Parse PDF files using MinerU API
    
    Parameters:
        pdf_path: Path to the PDF file
        token: MinerU API token, if None uses environment variable or cached token
        output_dir: Output directory, if None uses a temporary directory
        
    Returns:
        dict: Dictionary containing parsing results, including markdown text, JSON data and other parsing information
    """
    # Use st.spinner to display progress information
    with st.spinner("Parsing PDF file..."):
        # Get token
        if token is None:
            token = st.session_state.get('mineru_token', '')
            if not token:
                st.error("No MinerU API token provided, please configure in settings")
                return None
        
        # Create temporary directory or use specified directory
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            # Get filename and data ID
            file_name = os.path.basename(pdf_path)
            data_id = os.path.splitext(file_name)[0]
            
            # Create dedicated output directory for current file
            file_output_dir = os.path.join(output_dir, data_id)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Log information
            st.info(f"Started processing file: {file_name}")
            
            # Step 1: Get upload link
            upload_api = 'https://mineru.net/api/v4/file-urls/batch'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}'
            }
            data = {
                "enable_formula": True,
                "language": "en",
                "enable_table": True,
                "files": [
                    {"name": file_name, "is_ocr": True, "data_id": data_id}
                ]
            }
            
            st.text("Requesting upload link...")
            response = requests.post(upload_api, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            upload_url = res_json['data']['file_urls'][0]
            batch_id = res_json['data']['batch_id']
            st.text(f"Successfully obtained upload link, batch_id: {batch_id}")
            
            # Step 2: Upload PDF file
            st.text(f"Uploading file...")
            with open(pdf_path, 'rb') as f:
                upload_response = requests.put(upload_url, data=f)
                if upload_response.status_code == 200:
                    st.text("File uploaded successfully")
                else:
                    st.error(f"File upload failed, status code: {upload_response.status_code}")
                    return None
            
            # Step 3: Poll parsing status
            st.text("Waiting for parsing to complete...")
            status_url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
            progress_bar = st.progress(0)
            
            start_time = time.time()
            while True:
                time.sleep(3)
                check = requests.get(status_url, headers=headers)
                result = check.json()
                extract_result = result["data"]["extract_result"][0]
                state = extract_result["state"]
                
                # Update progress bar
                elapsed_time = time.time() - start_time
                if elapsed_time > 60:  # Assume maximum 2 minutes needed
                    progress = min(0.9, elapsed_time / 120)
                else:
                    progress = min(0.5, elapsed_time / 60)
                    
                progress_bar.progress(progress)
                
                if state == "done":
                    progress_bar.progress(1.0)
                    st.success("Parsing completed")
                    zip_url = extract_result["full_zip_url"]
                    break
                elif state == "failed":
                    st.error(f"Parsing failed: {extract_result.get('err_msg', 'Unknown error')}")
                    return None
                else:
                    st.text(f"Current status: {state}, continuing to wait...")
            
            # Step 4: Download ZIP file
            st.text("Downloading parsing results...")
            zip_response = requests.get(zip_url)
            local_zip_path = os.path.join(file_output_dir, f"{data_id}.zip")
            with open(local_zip_path, "wb") as f:
                f.write(zip_response.content)
            
            # Step 5: Extract ZIP and extract content
            st.text("Extracting parsed content...")
            extraction_results = {}
            
            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
                z.extractall(file_output_dir)
                
                # Find and save Markdown file
                md_content = None
                for name in z.namelist():
                    if name.endswith(".md"):
                        md_content = z.read(name).decode("utf-8")
                        md_path = os.path.join(file_output_dir, f"{data_id}.md")
                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(md_content)
                        extraction_results['markdown_path'] = md_path
                        extraction_results['markdown_content'] = md_content
                        break
                
                # Find and save JSON file
                json_content = None
                for name in z.namelist():
                    if name.endswith(".json"):
                        json_content = z.read(name).decode("utf-8")
                        json_path = os.path.join(file_output_dir, f"{data_id}.json")
                        with open(json_path, "w", encoding="utf-8") as f:
                            f.write(json_content)
                        extraction_results['json_path'] = json_path
                        extraction_results['json_content'] = json.loads(json_content)
                        break
                
                # Find and save image files
                image_files = []
                for name in z.namelist():
                    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        img_path = os.path.join(file_output_dir, name)
                        # Image is already extracted by extractall
                        image_files.append({
                            'name': name,
                            'path': img_path
                        })
                
                if image_files:
                    extraction_results['images'] = image_files
            
            # Save parsing result metadata
            extraction_results['data_id'] = data_id
            extraction_results['file_name'] = file_name
            extraction_results['output_dir'] = file_output_dir
            extraction_results['zip_path'] = local_zip_path
            extraction_results['batch_id'] = batch_id
            
            st.success(f"PDF parsing completed! Results saved to temporary directory")
            return extraction_results
            
        except Exception as e:
            st.error(f"Error occurred during parsing: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
        finally:
            # If using a temporary directory, can clean up here
            # But since you need to use these files later, don't clean up
            pass

# Example function for use in Streamlit application
def parse_pdf_in_streamlit(uploaded_file, token):
    """
    Process uploaded PDF file in Streamlit application
    
    Parameters:
        uploaded_file: Streamlit uploaded file object
        token: MinerU API token
        
    Returns:
        dict: Parsing results
    """
    if uploaded_file is None:
        return None
        
    # Create temporary file to save uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        # Call parsing function
        results = parser_pdf(pdf_path, token)
        return results
    finally:
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

def extract_info(markdown_content=None, markdown_path=None, api_key=None, output_path=None):
    """
    Extract structured information from parsed PDF content using DeepSeek API
    
    Parameters:
        markdown_content: Markdown content as string
        markdown_path: Path to markdown file (alternative to markdown_content)
        api_key: DeepSeek API key
        output_path: Path to save extracted information
        
    Returns:
        dict: Structured data extracted from the document
    """
    try:
        with st.spinner("Extracting structured information using DeepSeek API..."):
            # Check if either markdown_content or markdown_path is provided
            if markdown_content is None and markdown_path is None:
                st.error("Either markdown_content or markdown_path must be provided")
                return None
                
            # Get API key
            if api_key is None:
                api_key = st.session_state.get('deepseek_api_key', '')
                if not api_key:
                    st.error("No DeepSeek API key provided, please configure in settings")
                    return None
            
            # If markdown_path is provided but content is not, read the file
            if markdown_content is None and markdown_path:
                try:
                    markdown_content = read_md(markdown_path)
                except Exception as e:
                    st.error(f"Error reading markdown file: {str(e)}")
                    return None
            
            # Base configuration for DeepSeek API
            base_config = {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": api_key
            }
            
            # Create LLM instance
            llm = LLMCaller(
                model=base_config["model"],
                api_key=base_config["api_key"],
                base_url=base_config["base_url"]
            )
            
            # Generate prompt from content
            prompt = get_text_prompt(markdown_content)
            
            # Call LLM to extract information
            st.text("Processing with DeepSeek AI...")
            result = llm.call_llm(prompt, response_json=True, stream=False)
            
            # Parse result
            result_dict = json.loads(result)
            
            # Save result if output_path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                st.success(f"Extracted information saved to {output_path}")
            
            return result_dict
            
    except Exception as e:
        st.error(f"Error extracting information: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

