import re
import os
import logging
import openai
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def format_content(content: str, speaker1_instructions: str = "", speaker2_instructions: str = "") -> str:
    """
    Format content with speaker instructions and alternating speakers
    
    Args:
        content: The raw text content
        speaker1_instructions: Instructions for speaker 1
        speaker2_instructions: Instructions for speaker 2
        
    Returns:
        Formatted content with speaker tags
    """
    # Clean up the content
    content = content.strip()
    
    # Split content into paragraphs/sentences
    paragraphs = re.split(r'(?<=[.!?])\s+', content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    formatted_content = ""
    
    # Add speaker instructions
    formatted_content += f"SPEAKER_1_INSTRUCTION: {speaker1_instructions}\n"
    formatted_content += f"SPEAKER_2_INSTRUCTION: {speaker2_instructions}\n\n"
    
    # Alternate between speakers for each paragraph
    current_speaker = 1
    for paragraph in paragraphs:
        speaker = "SPEAKER_1" if current_speaker % 2 == 1 else "SPEAKER_2"
        formatted_content += f"{speaker}: {paragraph}\n\n"
        current_speaker += 1
    
    return formatted_content.strip()

def generate_script(formatted_content: str, api_key: str) -> str:
    """
    Generate a conversational script using OpenAI
    
    Args:
        formatted_content: Formatted content with speaker tags
        api_key: OpenAI API key
        
    Returns:
        Generated conversational script
    """
    try:
        # Create an OpenAI client with the API key
        client = openai.OpenAI(api_key=api_key)
        
        # Create system message with instructions
        system_message = """
        You are an expert podcast script writer. Your task is to transform the provided content 
        into a natural, engaging conversation between two speakers.
        
        Follow these guidelines:
        1. Maintain the alternating speaker format (SPEAKER_1 and SPEAKER_2)
        2. Follow the speaker instructions for tone and style
        3. Make the conversation flow naturally
        4. Keep the core information intact
        5. Add appropriate transitions between topics
        6. Keep each speaker's turn concise and engaging
        
        The input will include speaker instructions and content with speaker tags.
        Your output should maintain the same format with SPEAKER_1 and SPEAKER_2 tags.
        """
        
        # Call the OpenAI API using the new format (1.0.0+)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": formatted_content}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        # Extract and return the generated script
        script = response.choices[0].message.content
        return script
    
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        raise Exception(f"Failed to generate script: {str(e)}")

def extract_speaker_segments(script: str) -> List[Dict[str, str]]:
    """
    Extract speaker segments from the script
    
    Args:
        script: The generated script with speaker tags
        
    Returns:
        List of dictionaries with speaker and text
    """
    segments = []
    
    # Split the script by speaker tags
    lines = script.split('\n')
    current_speaker = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for speaker tags
        if line.startswith("SPEAKER_1:"):
            # Save previous segment if exists
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": current_text.strip()
                })
            
            # Start new segment
            current_speaker = "speaker1"
            current_text = line[len("SPEAKER_1:"):].strip()
        
        elif line.startswith("SPEAKER_2:"):
            # Save previous segment if exists
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": current_text.strip()
                })
            
            # Start new segment
            current_speaker = "speaker2"
            current_text = line[len("SPEAKER_2:"):].strip()
        
        elif current_speaker:
            # Continue current segment
            current_text += " " + line
    
    # Add the last segment
    if current_speaker and current_text:
        segments.append({
            "speaker": current_speaker,
            "text": current_text.strip()
        })
    
    return segments
