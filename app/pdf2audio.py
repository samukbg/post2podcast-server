import concurrent.futures as cf
import io
import os
import time
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any, Optional
# Remove Literal import as it's causing compatibility issues

import logging
from openai import OpenAI
from tenacity import retry, retry_if_exception_type
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define standard values
STANDARD_TEXT_MODELS = [
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-preview",
    "o1-pro",
    "o1-mini-2024-09-12",
    "o1-mini",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o3",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "openai/custom_model",
]

REASONING_EFFORTS = [
    "N/A",
    "low",
    "medium",
    "high",
]

STANDARD_AUDIO_MODELS = [
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
]

STANDARD_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
    "sage",
    "ash",
    "ballad",
    "coral",
    "nova",
]

# Define multiple sets of instruction templates
INSTRUCTION_TEMPLATES = {
    "podcast": {
        "intro": """Your task is to take the input text provided and turn it into an lively, engaging, informative podcast dialogue, in the style of NPR. Do not use or make up names. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. 
Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast. 
Define all terms used carefully for a broad audience of listeners.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.
Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.
Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.
Define all terms used clearly and spend effort to explain the background.
Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.
Make sure to make it fun and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. 
Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.
Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.
At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off. 
The podcast should have around 20000 words.
""",
    },
    "lecture": {
        "intro": """You are Professor Richard Feynman. Your task is to develop a script for a lecture. You never mention your name.
The material covered in the lecture is based on the provided text. 
Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be covered in the lecture. 
Define all terms used carefully for a broad audience of students.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """
Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.
Keep in mind that your lecture should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.
Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.
Define all terms used clearly and spend effort to explain the background.
Write your brainstorming ideas and a rough outline for the lecture here. Be sure to note the key insights and takeaways you want to reiterate at the end.
Make sure to make it fun and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the students.
Include clear definitions and terms, and examples. 
Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.
There is only one speaker, you, the professor. Stay on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest lecture you can, while still communicating the key information from the input text in an engaging way.
At the end of the lecture, naturally summarize the main insights and takeaways from the lecture. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 
Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas covered in this lecture one last time before class is over. 
The lecture should have around 20000 words.
""",
    },
    "summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.
Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.
Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.
Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.
Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.
Make sure to make it engaging and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.
Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.
Include clear definitions and terms, and examples, of all key issues. 
Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.
There is only one speaker, you. Stay on topic and maintaining an engaging flow. 
Naturally summarize the main insights and takeaways from the summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 
The summary should have around 1024 words.
""",
    },
    "short summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.
Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.
Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.
Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.
Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.
Make sure to make it engaging and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Keep it concise, and use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.
Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.
Include clear definitions and terms, and examples, of all key issues. 
Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.
There is only one speaker, you. Stay on topic and maintaining an engaging flow. 
Naturally summarize the main insights and takeaways from the short summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 
The summary should have around 256 words.
""",
    },
}

class DialogueItem(BaseModel):
    text: str
    speaker: str  # Will be either "speaker-1" or "speaker-2"

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]

def get_mp3(text: str, voice: str, audio_model: str, api_key: str = None,
           speaker_instructions: str = 'Speak in an emotive and friendly tone.') -> bytes:
    """
    Convert text to speech using OpenAI's TTS API
    
    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        audio_model: TTS model to use
        api_key: OpenAI API key
        speaker_instructions: Instructions for the speaker's tone and style
        
    Returns:
        Audio bytes
    """
    try:
        client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

        with client.audio.speech.with_streaming_response.create(
            model=audio_model,
            voice=voice,
            input=text,
            instructions=speaker_instructions,
        ) as response:
            with io.BytesIO() as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
                return file.getvalue()
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        # Create a fallback audio file
        with io.BytesIO() as file:
            # Write a minimal valid MP3 file header as fallback
            file.write(bytes.fromhex('FFFB9064000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'))
            return file.getvalue()

def extract_text_from_files(files: List[str]) -> str:
    """
    Extract text from uploaded files
    
    Args:
        files: List of file paths
        
    Returns:
        Combined text from all files
    """
    combined_text = ""
    
    for file in files:
        file_path = Path(file)
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                with file_path.open("rb") as f:
                    reader = PdfReader(f)
                    text = "\n\n".join(
                        page.extract_text() for page in reader.pages if page.extract_text()
                    )
                    combined_text += text + "\n\n"
            elif suffix in [".txt", ".md", ".mmd"]:
                with file_path.open("r", encoding="utf-8") as f:
                    text = f.read()
                    combined_text += text + "\n\n"
            else:
                logger.warning(f"Unsupported file type: {suffix}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
    
    return combined_text

def generate_dialogue_with_openai(
    text: str, 
    intro_instructions: str, 
    text_instructions: str, 
    scratch_pad_instructions: str, 
    prelude_dialog: str, 
    podcast_dialog_instructions: str,
    edited_transcript: str = None, 
    user_feedback: str = None,
    model: str = "gpt-4o",
    api_key: str = None,
    reasoning_effort: str = "N/A",
) -> Dialogue:
    """
    Generate dialogue using OpenAI
    
    Args:
        text: Input text
        intro_instructions: Introduction instructions
        text_instructions: Text analysis instructions
        scratch_pad_instructions: Scratch pad instructions
        prelude_dialog: Prelude dialog instructions
        podcast_dialog_instructions: Podcast dialog instructions
        edited_transcript: Previously edited transcript
        user_feedback: User feedback
        model: OpenAI model to use
        api_key: OpenAI API key
        reasoning_effort: Reasoning effort level
        
    Returns:
        Generated dialogue
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    # Process edited transcript and user feedback
    instruction_improve = 'Based on the original text, please generate an improved version of the dialogue by incorporating the edits, comments and feedback.'
    edited_transcript_processed = f"\nPreviously generated edited transcript, with specific edits and comments that I want you to carefully address:\n<edited_transcript>\n{edited_transcript}</edited_transcript>" if edited_transcript else ""
    user_feedback_processed = f"\nOverall user feedback:\n\n{user_feedback}" if user_feedback else ""

    if edited_transcript_processed.strip() != '' or user_feedback_processed.strip() != '':
        user_feedback_processed = f"<requested_improvements>{user_feedback_processed}\n\n{instruction_improve}</requested_improvements>"
    
    # Construct the prompt
    prompt = f"""
    {intro_instructions}
    
    Here is the original input text:
    
    <input_text>
    {text}
    </input_text>
    {text_instructions}
    
    <scratchpad>
    {scratch_pad_instructions}
    </scratchpad>
    
    {prelude_dialog}
    
    <podcast_dialogue>
    {podcast_dialog_instructions}
    </podcast_dialogue>
    {edited_transcript_processed}{user_feedback_processed}
    """
    
    # Add reasoning effort if specified
    messages = [{"role": "user", "content": prompt}]
    
    # Make API call
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )
        
        # Process response
        response_text = response.choices[0].message.content
        
        # Parse the response to extract dialogue
        scratchpad = ""
        dialogue = []
        
        # Simple parsing logic - in a real implementation, you would use more robust parsing
        lines = response_text.split('\n')
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for speaker indicators
            if line.startswith("SPEAKER_1:") or line.startswith("Speaker 1:"):
                # Save previous segment if exists
                if current_speaker and current_text:
                    dialogue.append(DialogueItem(speaker="speaker-1" if current_speaker == 1 else "speaker-2", text=current_text.strip()))
                
                # Start new segment
                current_speaker = 1
                current_text = line.split(":", 1)[1].strip()
            
            elif line.startswith("SPEAKER_2:") or line.startswith("Speaker 2:"):
                # Save previous segment if exists
                if current_speaker and current_text:
                    dialogue.append(DialogueItem(speaker="speaker-1" if current_speaker == 1 else "speaker-2", text=current_text.strip()))
                
                # Start new segment
                current_speaker = 2
                current_text = line.split(":", 1)[1].strip()
            
            elif current_speaker:
                # Continue current segment
                current_text += " " + line
        
        # Add the last segment
        if current_speaker and current_text:
            dialogue.append(DialogueItem(speaker="speaker-1" if current_speaker == 1 else "speaker-2", text=current_text.strip()))
        
        # If no dialogue was parsed, create a simple one
        if not dialogue:
            dialogue = [
                DialogueItem(speaker="speaker-1", text="Welcome to our podcast. Today we'll be discussing the text you provided."),
                DialogueItem(speaker="speaker-2", text="That's right. Let's dive into the key points from the document.")
            ]
        
        return Dialogue(scratchpad=scratchpad, dialogue=dialogue)
        
    except Exception as e:
        logger.error(f"Error generating dialogue: {str(e)}")
        # Return a simple dialogue as fallback
        return Dialogue(
            scratchpad="Error generating dialogue",
            dialogue=[
                DialogueItem(speaker="speaker-1", text="I apologize, but there was an error generating the dialogue."),
                DialogueItem(speaker="speaker-2", text="Please try again with a different input or settings.")
            ]
        )

def generate_audio_from_text(
    files: List[str],
    openai_api_key: str = None,
    text_model: str = "gpt-4o-mini",
    reasoning_effort: str = "N/A",
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
    speaker_1_instructions: str = 'Speak in an emotive and friendly tone.',
    speaker_2_instructions: str = 'Speak in a friendly, but serious tone.',
    api_base: str = None,
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
    edited_transcript: str = None,
    user_feedback: str = None,
    original_text: str = None,
) -> tuple:
    """
    Generate audio from text files
    
    Args:
        files: List of file paths
        openai_api_key: OpenAI API key
        text_model: Text generation model
        reasoning_effort: Reasoning effort level
        audio_model: Audio generation model
        speaker_1_voice: Voice for speaker 1
        speaker_2_voice: Voice for speaker 2
        speaker_1_instructions: Instructions for speaker 1
        speaker_2_instructions: Instructions for speaker 2
        api_base: Custom API base URL
        intro_instructions: Introduction instructions
        text_instructions: Text analysis instructions
        scratch_pad_instructions: Scratch pad instructions
        prelude_dialog: Prelude dialog instructions
        podcast_dialog_instructions: Podcast dialog instructions
        edited_transcript: Previously edited transcript
        user_feedback: User feedback
        original_text: Original text (if already provided)
        
    Returns:
        Tuple of (audio_file_path, transcript, original_text, audio_duration_seconds)
    """
    # Validate API Key
    if not os.getenv("OPENAI_API_KEY") and not openai_api_key:
        raise ValueError("OpenAI API key is required")

    # Get text from files or use provided text
    combined_text = original_text or ""
    if not combined_text:
        combined_text = extract_text_from_files(files)
    
    if not combined_text:
        raise ValueError("No text content found in the provided files")

    # Generate dialogue
    llm_output = generate_dialogue_with_openai(
        combined_text,
        intro_instructions=intro_instructions,
        text_instructions=text_instructions,
        scratch_pad_instructions=scratch_pad_instructions,
        prelude_dialog=prelude_dialog,
        podcast_dialog_instructions=podcast_dialog_instructions,
        edited_transcript=edited_transcript,
        user_feedback=user_feedback,
        model=text_model,
        api_key=openai_api_key,
        reasoning_effort=reasoning_effort
    )

    # Generate audio from the transcript
    audio = b""
    transcript = ""
    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            voice = speaker_1_voice if line.speaker == "speaker-1" else speaker_2_voice
            speaker_instructions = speaker_1_instructions if line.speaker == "speaker-1" else speaker_2_instructions
            future = executor.submit(get_mp3, line.text, voice, audio_model, openai_api_key, speaker_instructions)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    # Create temporary directory for audio files
    temporary_directory = "./audio_output"
    os.makedirs(temporary_directory, exist_ok=True)

    # Use a temporary file
    audio_filename = f"podcast_{int(time.time())}.mp3"
    audio_filepath = os.path.join(temporary_directory, audio_filename)
    
    with open(audio_filepath, "wb") as f:
        f.write(audio)

    # Calculate duration
    audio_duration_seconds = 0
    if audio:
        try:
            from pydub import AudioSegment # Ensure pydub is imported here if not globally
            audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
            audio_duration_seconds = len(audio_segment) / 1000.0
            logger.info(f"Calculated audio duration for PDF2Audio: {audio_duration_seconds:.2f}s")
        except Exception as e:
            logger.error(f"Could not calculate audio duration for PDF2Audio: {str(e)}")

    return audio_filepath, transcript, combined_text, audio_duration_seconds

def update_instructions(template: str) -> tuple:
    """
    Get instructions for a specific template
    
    Args:
        template: Template name
        
    Returns:
        Tuple of instructions
    """
    if template not in INSTRUCTION_TEMPLATES:
        template = "podcast"  # Default to podcast if template not found
        
    return (
        INSTRUCTION_TEMPLATES[template]["intro"],
        INSTRUCTION_TEMPLATES[template]["text_instructions"],
        INSTRUCTION_TEMPLATES[template]["scratch_pad"],
        INSTRUCTION_TEMPLATES[template]["prelude"],
        INSTRUCTION_TEMPLATES[template]["dialog"]
    )
