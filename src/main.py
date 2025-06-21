import os
import argparse
import time
from typing import List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class FlaggedText(BaseModel):
    flagged_text: str = Field(
        ...,
        description="The sentence of text that was flagged.",
    )
    reason: Literal["profanity", "personal_information", "specific_events", "controversial_statements", "other_inappropriate_content"] = Field(
        ...,
        description="The reason for the flagged text.",
    )

SYSTEM_PROMPT = f"""
You are a world-class content moderator.
Given a transcript of a lecture, flag aggressively ALL of the following:
1. any profanity (words like God, shit, fuck, etc) 
2. personal information related to the professor,
3. specific incidents, 
4. controversial statements that show certain places or people in bad light, 
5. and any other information that is not appropriate for a public platform.
"""

google_client = instructor.from_provider("google/gemini-2.5-flash-preview-04-17")

openrouter_client = instructor.from_openai(
    client=OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    ),
    mode=instructor.Mode.JSON, # patch for most providers @SEE: https://github.com/567-labs/instructor/issues/676
)

def analyze(transcript: str) -> List[FlaggedText]:
    result = openrouter_client.chat.completions.create(
        model="google/gemini-2.5-flash-preview-05-20",
        temperature=0.0, # no randomness
        response_model=List[FlaggedText],
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Flag inappropriate content in the following transcript: <BEGIN TRANSCRIPT>{transcript}<END TRANSCRIPT>"
            }
        ],
    )
    # Instructor handles and returns a final result, we don't need to await
    return result # type: ignore

def write_analysis_to_file(video_path: str, transcript: str, analysis: str) -> str:
    # Create a filename based on the input video name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"analysis/{base_name}_analysis.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(analysis)
    return output_filename

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_transcript(transcript: str) -> List[str]:
    """
    Splits the transcript into chunks of ~TARGET_WORDS words.
    Sentences should not be split across chunks => we cut off that last sentence such that a chunk may have >TARGET_WORDS words.
    As such, there may be a remainder very small chunk at the end.
    """
    TARGET_WORDS = 300 # experimental value

    # Split transcript into sentences using common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_word_count = 0
    
    for sentence in sentences:
        # 1) Count words in the current sentence
        sentence_word_count = len(sentence.split())
        
        # 2) If adding this sentence would exceed 500 words, start a new chunk
        if current_word_count + sentence_word_count > TARGET_WORDS and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
    
    # 3) Add the last (remainder) chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def main(video_path: str) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
    
    # transcript = transcribe_video(video_path)
    transcript = read_file(r"src/prompts/transcript_1017.txt")
    
    start_time = time.time()
    chunks = chunk_transcript(transcript)

    print(f"Starting analysis with {len(chunks)} chunks...")

    # Analyze each chunk in parallel
    analysis: List[FlaggedText] = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze, chunk) for chunk in chunks]
        for future in as_completed(futures):
            analysis.extend(future.result())

    end_time = time.time()

    # Convert the analysis to a JSON-like string to be written to a file
    # This is temporary until we setup the proper pipeline
    analysis_to_str = "[\n"
    for i, item in enumerate(analysis):
        analysis_to_str += f'  {{\n    "flagged_text": "{item.flagged_text}",\n    "reason": "{item.reason}"\n  }}'
        if i < len(analysis) - 1:
            analysis_to_str += ","
        analysis_to_str += "\n"
    analysis_to_str += "]"

    analysis_time = end_time - start_time
    print(f"Analysis done in {analysis_time:.2f} seconds")
    
    write_analysis_to_file(video_path, transcript, analysis_to_str)
    print(f"Analysis written to analysis/{video_path}_analysis.txt")

def cli():
    parser = argparse.ArgumentParser(description="Transcribe and analyze a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    main(args.video_path)

if __name__ == "__main__":
    cli()