import os
import re
import time
import logging
import tempfile
import subprocess
import shutil
import sys
from typing import List, Dict, Any

# Check if ffmpeg is available
ffmpeg_available = shutil.which('ffmpeg') is not None
if not ffmpeg_available:
    logging.warning("ffmpeg not found in PATH. Audio generation will use fallback methods.")

# Import pydub only if ffmpeg is available
try:
    from pydub import AudioSegment
    pydub_available = True
except ImportError:
    logging.warning("pydub not installed. Audio generation will use fallback methods.")
    pydub_available = False
except Exception as e:
    logging.warning(f"Error importing pydub: {str(e)}. Audio generation will use fallback methods.")
    pydub_available = False

from .text_processor import extract_speaker_segments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mock TTS function for development - in production, use a real TTS service
def tts_mock(text: str, voice: str, output_path: str) -> str:
    """
    Mock text-to-speech function that creates a silent audio file
    
    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        output_path: Path to save the audio file
        
    Returns:
        Path to the generated audio file
    """
    # Check if pydub is available
    if not pydub_available:
        # Create a minimal MP3 file if pydub is not available
        with open(output_path, 'wb') as f:
            # Write a minimal valid MP3 file header
            f.write(bytes.fromhex('FFFB9064000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'))
        return output_path
    
    # Create a silent audio segment (1 second per 10 characters)
    duration = max(1000, len(text) * 100)  # At least 1 second
    audio = AudioSegment.silent(duration=duration)
    audio.export(output_path, format="mp3")
    return output_path

def tts_azure(text: str, voice: str, output_path: str, api_key: str = None, region: str = None) -> str:
    """
    Convert text to speech using Azure TTS
    
    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        output_path: Path to save the audio file
        api_key: Azure Speech API key
        region: Azure region
        
    Returns:
        Path to the generated audio file
    """
    # This is a placeholder for Azure TTS implementation
    # In a real implementation, you would use the Azure SDK or API
    
    # For now, use the mock implementation
    return tts_mock(text, voice, output_path)

def text_to_speech(script: str, output_path: str, voice1: str = "alloy", voice2: str = "nova", openai_api_key: str = None) -> str:
    """
    Convert script to speech with alternating voices
    
    Args:
        script: The script with speaker tags
        output_path: Path to save the final audio file
        voice1: Voice ID for speaker 1
        voice2: Voice ID for speaker 2
        
    Returns:
        Path to the generated audio file
        Duration of the audio in seconds
    """
    try:
        # Extract speaker segments
        segments = extract_speaker_segments(script)
        
        # Check if we can use pydub and ffmpeg
        if not ffmpeg_available or not pydub_available:
            logger.warning("Using fallback audio generation method (no ffmpeg or pydub available)")
            return fallback_audio_generation(script, output_path)
        
        # Create a temporary directory for segment audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            segment_files = []
            
            # Process each segment
            for i, segment in enumerate(segments):
                # Determine voice
                voice = voice1 if segment["speaker"] == "speaker1" else voice2
                
                # Create output path for this segment
                segment_path = os.path.join(temp_dir, f"segment_{i}.mp3")
                
                # Convert text to speech using OpenAI TTS
                try:
                    import openai
                    # Configure OpenAI with the API key
                    if openai_api_key:
                        openai.api_key = openai_api_key
                    
                    # Use OpenAI TTS API
                    with open(segment_path, "wb") as audio_file:
                        response = openai.audio.speech.create(
                            model="tts-1",
                            voice=voice,
                            input=segment["text"]
                        )
                        response.stream_to_file(segment_path)
                    logger.info(f"Generated audio segment with OpenAI TTS using voice: {voice}")
                except Exception as e:
                    logger.error(f"OpenAI TTS failed: {str(e)}. Falling back to mock TTS.")
                    # Fallback to mock if OpenAI fails
                    tts_mock(segment["text"], voice, segment_path)
                
                # Add to list of segment files
                segment_files.append(segment_path)
            
            # Combine all segments
            combined = AudioSegment.empty()
            for segment_file in segment_files:
                audio_segment = AudioSegment.from_mp3(segment_file)
                combined += audio_segment
                
                # Add a short pause between segments
                combined += AudioSegment.silent(duration=500)
            
            # Export the combined audio
            combined.export(output_path, format="mp3")
            
            duration_seconds = len(combined) / 1000.0 # pydub duration is in milliseconds
            logger.info(f"Audio generated successfully: {output_path}, Duration: {duration_seconds:.2f}s")
            return output_path, duration_seconds
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise Exception(f"Failed to generate audio: {str(e)}")

def fallback_audio_generation(script: str, output_path: str) -> str:
    """
    Fallback method for audio generation when ffmpeg is not available
    
    Args:
        script: The script with speaker tags
        output_path: Path to save the final audio file
        
    Returns:
        Path to the generated audio file
        Duration of the audio in seconds (0 for fallback)
    """
    try:
        # Create a simple text file with the script content
        # This is just a placeholder - in a real implementation, you would use a different TTS method
        # that doesn't rely on ffmpeg, or generate a simple audio file with a warning message
        
        # For now, we'll create a dummy MP3 file
        with open(output_path, 'wb') as f:
            # Write a minimal valid MP3 file header
            # This is just a placeholder and won't play actual audio
            f.write(bytes.fromhex('FFFB9064000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'))
        
        # Also save the script as a text file alongside the MP3
        text_path = output_path.replace('.mp3', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        logger.info(f"Created fallback audio file: {output_path}")
        logger.info(f"Script saved to text file: {text_path}")
        
        return output_path, 0 # Fallback duration is 0
    except Exception as e:
        logger.error(f"Error in fallback audio generation: {str(e)}")
        raise Exception(f"Failed to generate audio even with fallback method: {str(e)}")

def get_available_voices() -> Dict[str, List[Dict[str, str]]]:
    """
    Get available voices for TTS
    
    Returns:
        Dictionary of available voices by language
    """
    # This would typically come from the TTS service API
    # For now, return a static list
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
