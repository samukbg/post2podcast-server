from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Body, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uuid
import time
import json
import shutil
from typing import Optional, Dict, Any, List
import logging
from .text_processor import format_content, generate_script
from .audio_generator import text_to_speech
from .pdf2audio import (
    generate_audio_from_text,
    update_instructions,
    STANDARD_TEXT_MODELS,
    STANDARD_AUDIO_MODELS,
    STANDARD_VOICES,
    REASONING_EFFORTS,
    INSTRUCTION_TEMPLATES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from . import db_utils # Import the new db_utils module

# Create app
app = FastAPI(
    title="Text to Podcast API",
    description="API for converting text to podcast audio",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving audio files
app.mount("/static", StaticFiles(directory="audio_output"), name="static")

# Create directories if they don't exist
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "../uploads")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "../audio_output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Store job status
jobs = {}

class GenerateRequest(BaseModel):
    file_id: str
    voice1: str = "en-US-JennyNeural"
    voice2: str = "en-US-GuyNeural"
    speaker1_instructions: str = "Speak in an emotive and friendly tone."
    speaker2_instructions: str = "Speak in a friendly, but serious tone."
    openai_api_key: str

class PDF2AudioRequest(BaseModel):
    openai_api_key: str
    text_model: str = "o3-mini"
    reasoning_effort: str = "N/A"
    audio_model: str = "tts-1"
    speaker_1_voice: str = "alloy"
    speaker_2_voice: str = "echo"
    speaker_1_instructions: str = "Speak in an emotive and friendly tone."
    speaker_2_instructions: str = "Speak in a friendly, but serious tone."
    api_base: Optional[str] = None
    template: str = "podcast"
    wp_user_email: Optional[str] = None # Added for service tier tracking
    wp_site_url: Optional[str] = None   # Added for service tier tracking
    edited_transcript: Optional[str] = None
    user_feedback: Optional[str] = None
    original_text: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    audio_id: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Text to Podcast API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a text file for processing"""
    try:
        # Generate a unique ID for the file
        file_id = f"{uuid.uuid4()}.txt"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_id}")
        
        return {"message": "File uploaded successfully", "file_id": file_id}
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=JobStatus)
async def generate_audio(background_tasks: BackgroundTasks, request: GenerateRequest):
    """Generate podcast audio from uploaded text file"""
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        jobs[job_id] = {
            "status": "processing",
            "message": "Job started",
            "start_time": time.time()
        }
        
        # Start background task
        background_tasks.add_task(
            process_audio_generation,
            job_id,
            request.file_id,
            request.voice1,
            request.voice2,
            request.speaker1_instructions,
            request.speaker2_instructions,
            request.openai_api_key
        )
        
        return JobStatus(
            job_id=job_id,
            status="processing",
            message="Audio generation started"
        )
    
    except Exception as e:
        logger.error(f"Error starting generation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of an audio generation job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        audio_id=job.get("audio_id"),
        error=job.get("error")
    )

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Get the generated audio file"""
    audio_path = os.path.join(AUDIO_DIR, audio_id)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"podcast_{audio_id}"
    )

@app.get("/voices")
async def get_voices():
    """Get available voice options"""
    # This would typically come from a database or external API
    # For now, we'll return a static list
    voices = {
        "en": [
            {"id": "en-US-JennyNeural", "name": "Jenny (Female)"},
            {"id": "en-US-GuyNeural", "name": "Guy (Male)"},
            {"id": "en-US-AriaNeural", "name": "Aria (Female)"},
            {"id": "en-US-DavisNeural", "name": "Davis (Male)"}
        ],
        "es": [
            {"id": "es-ES-ElviraNeural", "name": "Elvira (Female)"},
            {"id": "es-ES-AlvaroNeural", "name": "Alvaro (Male)"}
        ],
        "fr": [
            {"id": "fr-FR-DeniseNeural", "name": "Denise (Female)"},
            {"id": "fr-FR-HenriNeural", "name": "Henri (Male)"}
        ]
    }
    
    return voices

@app.get("/pdf2audio/models")
async def get_pdf2audio_models():
    """Get available models for PDF2Audio"""
    return {
        "text_models": STANDARD_TEXT_MODELS,
        "audio_models": STANDARD_AUDIO_MODELS,
        "voices": STANDARD_VOICES,
        "reasoning_efforts": REASONING_EFFORTS,
        "templates": list(INSTRUCTION_TEMPLATES.keys())
    }

@app.get("/pdf2audio/template/{template_name}")
async def get_template_instructions(template_name: str):
    """Get instructions for a specific template"""
    if template_name not in INSTRUCTION_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    
    intro, text_instructions, scratch_pad, prelude, dialog = update_instructions(template_name)
    
    return {
        "intro": intro,
        "text_instructions": text_instructions,
        "scratch_pad": scratch_pad,
        "prelude": prelude,
        "dialog": dialog
    }

@app.post("/pdf2audio/generate/text")
async def generate_pdf2audio_from_text(background_tasks: BackgroundTasks, request: PDF2AudioRequest):
    """Generate podcast audio directly from text using PDF2Audio"""
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        jobs[job_id] = {
            "status": "processing",
            "message": "Job started",
            "start_time": time.time()
        }
        
        # Get template instructions
        intro, text_instructions, scratch_pad, prelude, dialog = update_instructions(request.template)
        
        # Start background task
        background_tasks.add_task(
            process_pdf2audio_generation,
            job_id,
            request.openai_api_key,
            request.text_model,
            request.reasoning_effort,
            request.audio_model,
            request.speaker_1_voice,
            request.speaker_2_voice,
            request.speaker_1_instructions,
            request.speaker_2_instructions,
            request.api_base,
            intro,
            text_instructions,
            scratch_pad,
            prelude,
            dialog,
            request.edited_transcript,
            request.user_feedback,
            request.original_text,
            request.wp_user_email, # Pass new field
            request.wp_site_url    # Pass new field
        )
        
        return JobStatus(
            job_id=job_id,
            status="processing",
            message="Audio generation started"
        )
    
    except Exception as e:
        logger.error(f"Error starting PDF2Audio generation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdf2audio/generate")
async def generate_pdf2audio(background_tasks: BackgroundTasks, request: PDF2AudioRequest):
    """Generate podcast audio from text using PDF2Audio"""
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        jobs[job_id] = {
            "status": "processing",
            "message": "Job started",
            "start_time": time.time()
        }
        
        # Get template instructions
        intro, text_instructions, scratch_pad, prelude, dialog = update_instructions(request.template)
        
        # Start background task
        background_tasks.add_task(
            process_pdf2audio_generation,
            job_id,
            request.openai_api_key,
            request.text_model,
            request.reasoning_effort,
            request.audio_model,
            request.speaker_1_voice,
            request.speaker_2_voice,
            request.speaker_1_instructions,
            request.speaker_2_instructions,
            request.api_base,
            intro,
            text_instructions,
            scratch_pad,
            prelude,
            dialog,
            request.edited_transcript,
            request.user_feedback,
            request.original_text,
            request.wp_user_email, # Pass new field
            request.wp_site_url    # Pass new field
        )
        
        return JobStatus(
            job_id=job_id,
            status="processing",
            message="Audio generation started"
        )
    
    except Exception as e:
        logger.error(f"Error starting PDF2Audio generation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdf2audio/upload")
async def upload_text_file(file: UploadFile = File(...)):
    """Upload a text file for PDF2Audio processing"""
    try:
        # Generate a unique ID for the file
        file_id = f"{uuid.uuid4()}.txt"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded for PDF2Audio: {file_id}")
        
        return {"message": "File uploaded successfully", "file_id": file_id}
    
    except Exception as e:
        logger.error(f"Error uploading file for PDF2Audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_pdf2audio_generation(
    job_id: str,
    openai_api_key: str, # Key from WordPress request
    text_model: str,
    reasoning_effort: str,
    audio_model: str,
    speaker_1_voice: str,
    speaker_2_voice: str,
    speaker_1_instructions: str,
    speaker_2_instructions: str,
    api_base: str,
    intro_instructions: str,
    text_instructions: str,
    scratch_pad_instructions: str,
    prelude_dialog: str,
    podcast_dialog_instructions: str,
    edited_transcript: str,
    user_feedback: str,
    original_text: str,
    # These will be passed from the request object by FastAPI when calling the background task
    # However, the background task signature needs to match how it's called.
    # Let's assume they are passed through the 'request: PDF2AudioRequest' object
    # and accessed via request.wp_user_email, request.wp_site_url inside the calling endpoint.
    # The background task itself will receive these as direct arguments.
    # So, the call from the endpoint needs to be updated.
    # For now, let's add them to the signature here.
    wp_user_email_from_request: Optional[str] = None, 
    wp_site_url_from_request: Optional[str] = None
):
    """Process PDF2Audio generation in the background"""
    is_self_hosted_call = bool(openai_api_key) # If user provides their own key, it's self-hosted
    actual_openai_api_key_to_use = openai_api_key # Key from the request (for self-hosted)
    user_identifier = None

    try:
        if not is_self_hosted_call:
            # This is a service user, check free credits
            if not wp_user_email_from_request or not wp_site_url_from_request:
                logger.error(f"Job {job_id}: Missing wp_user_email or wp_site_url for service tier audio generation.")
                jobs[job_id].update({"status": "failed", "message": "User identification missing for service tier.", "error": "User identification missing."})
                return

            user_identifier = f"{wp_user_email_from_request.lower().strip()}:{wp_site_url_from_request.lower().strip().rstrip('/')}"
            
            # Check for active subscription first
            active_subscription = db_utils.get_subscription_status_by_identifier(user_identifier)

            if active_subscription:
                logger.info(f"Job {job_id}: User {user_identifier} has an active subscription. Bypassing free credit check.")
                # User is subscribed and active, proceed with server's key
                actual_openai_api_key_to_use = os.environ.get("OPENAI_API_KEY")
            else:
                # Not actively subscribed (or no record), check free credits
                credits_used = db_utils.get_free_credits_used(user_identifier)
                MAX_FREE_CREDITS = 3 # Define this constant
                if credits_used >= MAX_FREE_CREDITS:
                    logger.warning(f"Job {job_id}: User {user_identifier} has exhausted free credits ({credits_used}/{MAX_FREE_CREDITS} used) and no active subscription.")
                    jobs[job_id].update({
                        "status": "failed", 
                        "message": "Free credits exhausted.", 
                        "error": f"You have used all your {MAX_FREE_CREDITS} free audio generations. Please subscribe for more."
                    })
                    return
                # If service call and free credits available, use server's OpenAI key
                actual_openai_api_key_to_use = os.environ.get("OPENAI_API_KEY")
                logger.info(f"Job {job_id}: User {user_identifier} using free credit {credits_used + 1}/{MAX_FREE_CREDITS}.")


            if not actual_openai_api_key_to_use: # This covers both subscribed and free tier needing the server key
                logger.error(f"Job {job_id}: Server's OPENAI_API_KEY is not configured for service tier generation.")
                jobs[job_id].update({"status": "failed", "message": "Service configuration error.", "error": "Service configuration error."})
                return
        
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Starting PDF2Audio generation"
        
        # Generate audio
        audio_filepath, transcript, _, audio_duration_seconds = generate_audio_from_text(
            files=[],
            openai_api_key=actual_openai_api_key_to_use, # Use the determined API key
            text_model=text_model,
            reasoning_effort=reasoning_effort,
            audio_model=audio_model,
            speaker_1_voice=speaker_1_voice,
            speaker_2_voice=speaker_2_voice,
            speaker_1_instructions=speaker_1_instructions,
            speaker_2_instructions=speaker_2_instructions,
            api_base=api_base,
            intro_instructions=intro_instructions,
            text_instructions=text_instructions,
            scratch_pad_instructions=scratch_pad_instructions,
            prelude_dialog=prelude_dialog,
            podcast_dialog_instructions=podcast_dialog_instructions,
            edited_transcript=edited_transcript,
            user_feedback=user_feedback,
            original_text=original_text
        )
        
        # Get audio filename from path
        audio_id = os.path.basename(audio_filepath)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "PDF2Audio generation completed"
        jobs[job_id]["audio_id"] = audio_id
        jobs[job_id]["transcript"] = transcript
        jobs[job_id]["audio_duration"] = audio_duration_seconds # Store duration
        jobs[job_id]["completion_time"] = time.time()
        
        log_user_identifier = user_identifier if user_identifier else 'self-hosted'
        logger.info(f"PDF2Audio job {job_id} completed successfully for user_identifier='{log_user_identifier}', Duration: {audio_duration_seconds:.2f}s")

        # After successful (or attempted, depending on policy) generation for a service user on free tier:
        if not is_self_hosted_call and user_identifier and not active_subscription: 
            # Only increment free credits if they are not actively subscribed
            db_utils.increment_free_credit_usage(user_identifier)
            logger.info(f"Job {job_id}: Incremented free credit usage for {user_identifier}.")
            
    except Exception as e:
        log_user_identifier_on_error = 'self-hosted'
        if user_identifier: # If it was set (i.e., service call attempt)
            log_user_identifier_on_error = user_identifier
        elif wp_user_email_from_request and wp_site_url_from_request: # If it was a service call but failed before user_identifier was set
             log_user_identifier_on_error = f"{wp_user_email_from_request.lower().strip()}:{wp_site_url_from_request.lower().strip().rstrip('/')}"
        
        logger.error(f"PDF2Audio job {job_id} failed for user_identifier='{log_user_identifier_on_error}': {str(e)}")
        # Update job status with error
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = "PDF2Audio generation failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completion_time"] = time.time()
        
        logger.error(f"PDF2Audio job {job_id} failed: {str(e)}")

async def process_audio_generation(
    job_id: str,
    file_id: str,
    voice1: str,
    voice2: str,
    speaker1_instructions: str,
    speaker2_instructions: str,
    openai_api_key: str # Key from WordPress request
):
    """Process audio generation in the background"""
    try:
        # Determine the API key to use
        # If WordPress sends a key (self-hosted mode ON in WP), use that.
        # Otherwise (self-hosted mode OFF in WP), use the server's environment variable.
        actual_openai_api_key = openai_api_key
        if not actual_openai_api_key:
            logger.info("OpenAI API key not provided in request for /generate, attempting to use server environment variable.")
            actual_openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not actual_openai_api_key:
                logger.error("OpenAI API key is not configured on the server (OPENAI_API_KEY env var) and was not provided in the request.")
                raise Exception("OpenAI API key is not configured on the server and was not provided in the request.")
        
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Reading text file"
        
        # Get the text file
        file_path = os.path.join(UPLOAD_DIR, file_id)
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_id}")
        
        # Read the content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Update job status
        jobs[job_id]["message"] = "Formatting content"
        
        # Format the content
        formatted_content = format_content(content, speaker1_instructions, speaker2_instructions)
        
        # Update job status
        jobs[job_id]["message"] = "Generating script"
        
        # Generate script
        script = generate_script(formatted_content, actual_openai_api_key)
        
        # Update job status
        jobs[job_id]["message"] = "Converting text to speech"
        
        # Generate audio
        audio_id = f"podcast_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_id)
        
        # Convert to speech with OpenAI API key
        generated_audio_path, audio_duration_seconds = text_to_speech(script, audio_path, voice1, voice2, actual_openai_api_key)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Audio generation completed"
        jobs[job_id]["audio_id"] = audio_id
        jobs[job_id]["audio_duration"] = audio_duration_seconds # Store duration
        jobs[job_id]["completion_time"] = time.time()
        
        logger.info(f"Job {job_id} completed successfully, Duration: {audio_duration_seconds:.2f}s")
    
    except Exception as e:
        # Update job status with error
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = "Audio generation failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completion_time"] = time.time()
        
        logger.error(f"Job {job_id} failed: {str(e)}")

# Attempt to import and include Stripe webhooks if the module exists
# All Stripe routes (webhooks, create-checkout-session) are now in stripe_webhooks.py
# and should be prefixed, e.g., with /stripe
try:
    from . import stripe_webhooks
    app.include_router(stripe_webhooks.router, prefix="/stripe", tags=["Stripe"])
    logger.info("Stripe webhooks module loaded and router included under /stripe prefix.")
except ImportError:
    logger.info("Stripe webhooks module (stripe_webhooks.py) not found. Skipping Stripe routes.")
except AttributeError as e:
    logger.error(f"Stripe webhooks module found, but 'router' attribute is missing or other error: {e}. Stripe routes not loaded.")

@app.on_event("startup")
async def startup_event():
    db_utils.init_db()
    logger.info("Application startup: Database initialized.")
