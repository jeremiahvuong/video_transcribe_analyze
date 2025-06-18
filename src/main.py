import os
import argparse
import time
from dotenv import load_dotenv
from google import genai # type: ignore

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def analyze_content(transcript: str) -> str:
    example_flags = read_file(r"src/prompts/human_flags_1017.txt")
    claude_flags = read_file(r"src/prompts/claude_flags_1017.txt")
    example_transcript = read_file(r"src/prompts/transcript_1017.txt")

    prompt1 = f"""
    Go through the entire transcript and point out 
    1. any profanity (words like God, shit, fuck, etc) 
    2. personal information related to the professor,
    3. specific incidents, 
    4. controversial statements that show certain places or people in bad light, 
    
    and flag out everything you can find.
    """

    prompt2 = f"""
    Here is an example transcript:
    {example_transcript}
    and here are just of the flags found by a human:
    {example_flags}
    """

    prompt3 = f"""
    Here are some additional flags found by me which match the criteria of the prompt:
    {claude_flags}
    """

    prompt4 = f"""
    Now go through this entire transcript and get similar flags.

    Be very sensitive, flag anything if you think that meets the criteria above.

    For the transcript below, return the specific sentences in quotes, and the reason you flagged them:

    The main transcript:

    {transcript}.
    """

    prompt = f"{prompt1}\n\n{prompt2}\n\n{prompt3}\n\n{prompt4}"

    return analyze_with_gemini(prompt)

def analyze_with_gemini(prompt: str) -> str:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(model="models/gemini-2.5-flash-preview-05-20", contents=prompt)
    if response.text:
        return response.text
    else:
        raise ValueError("Error generating analysis.")

def write_analysis_to_file(video_path: str, transcript: str, analysis: str) -> str:
    # Create a filename based on the input video name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"analysis/{base_name}_analysis.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"Analysis for video: {video_path}\n\n")
        f.write("\n\nContent Analysis:\n")
        f.write(analysis)
    
    return output_filename

def main(video_path: str) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
    
    # transcript = transcribe_video(video_path)
    transcript = read_file(r"src/prompts/transcript_1017.txt")
    print(f"Generating analysis...")
    
    start_time = time.time()
    analysis = analyze_content(transcript)
    end_time = time.time()
    
    analysis_time = end_time - start_time
    print(f"Analysis done in {analysis_time:.2f} seconds")
    
    write_analysis_to_file(video_path, transcript, analysis)
    print(f"Analysis written to analysis/{video_path}_analysis.txt")

def cli():
    parser = argparse.ArgumentParser(description="Transcribe and analyze a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    main(args.video_path)

if __name__ == "__main__":
    cli()