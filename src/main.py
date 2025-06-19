import os
import argparse
import time
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import instructor

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = instructor.from_provider("google/gemini-2.5-flash-preview-04-17")

class FlaggedText(BaseModel):
    flagged_text: str = Field(
        description="""The exact text verbatim from the transcript that should be flagged. 
        This should be a direct quote, not a paraphrase or summary. 
        Include enough context to identify the problematic content clearly.
        """,
    )
    reason: str = Field(
        description="""A clear, specific explanation of why this text is flagged. 
        Should reference the specific category (profanity, personal information, 
        controversial content, etc.) and explain the potential risk or concern."""
    )
    category: str = Field(
        description="""The primary category this flag falls into: 
        'profanity', 'personal_information', 'specific_events', 
        'controversial_statements', or 'inappropriate_academic_content'"""
    )

def split_transcript_into_chunks(transcript: str, num_chunks: int) -> List[str]:
    """Split transcript into approximately equal chunks."""
    words = transcript.split()
    chunk_size = len(words) // num_chunks
    remainder = len(words) % num_chunks
    
    chunks: List[str] = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Add one extra word to the first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk_words = words[start_idx:end_idx]
        chunks.append(' '.join(chunk_words))
        start_idx = end_idx
    
    return chunks

def analyze_chunk_with_gemini(chunk: str) -> List[FlaggedText]:
    """Analyze a single chunk and return flagged content."""
    system_prompt = """
    You are an expert content analyst specializing in educational material review. Your task is to analyze academic lecture transcripts and identify content that may be inappropriate for public platforms or academic distribution.

    CONTEXT:
    - This is a transcript from an academic lecture (likely economics/public policy)
    - The content will be made available on public platforms
    - We need to maintain academic integrity while protecting privacy and avoiding controversy

    ANALYSIS CRITERIA:
    You should flag text that falls into these categories:

    1. PROFANITY & OFFENSIVE LANGUAGE
    - Any profanity, swear words, or crude language
    - Examples: "shit", "fuck", "damn", "holy shit", etc.
    - Include mild profanity that might be inappropriate in formal academic settings

    2. PERSONAL INFORMATION & PRIVACY
    - Names of specific individuals (students, colleagues, family members)
    - Personal anecdotes that reveal private information about the speaker
    - Specific details about personal activities, travel, or private life
    - Information about family members or personal relationships

    3. SPECIFIC IDENTIFIABLE EVENTS & LOCATIONS
    - Precise dates, times, or locations of specific incidents
    - Detailed descriptions of specific cases or individuals that could be identified
    - References to specific institutions' internal matters (unless already public)
    - Confidential institutional information

    4. POTENTIALLY CONTROVERSIAL STATEMENTS
    - Content that portrays specific places, institutions, or groups negatively
    - Statements that could be seen as discriminatory or offensive
    - Unsubstantiated claims about specific organizations or individuals
    - Content that might create legal liability

    5. INAPPROPRIATE ACADEMIC CONTENT
    - Information that violates confidentiality (like specific enrollment numbers)
    - Content that might compromise professional relationships
    - Statements that could reflect poorly on the institution

    IMPORTANT GUIDELINES:
    - Focus on content that would be problematic if made public
    - Consider the context - academic discussions may include sensitive topics appropriately
    - Be thorough but not overly restrictive - don't flag general academic concepts
    - Provide clear, specific reasons for each flag
    - Extract the exact problematic text, not paraphrases

    OUTPUT FORMAT:
    For each flagged item, provide:
    - The exact text that should be flagged (quote directly from transcript)
    - A clear reason explaining why it's problematic and which category it falls into
    """

    try:
        result = client.chat.completions.create(
            response_model=List[FlaggedText],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""
                    Please analyze the following academic lecture transcript chunk and identify any content that should be flagged according to the criteria provided:
                    <transcript_chunk>
                    {chunk}
                    </transcript_chunk>
                    
                    Provide a comprehensive analysis, ensuring you capture all instances that meet the flagging criteria. Be thorough in your review.
                    """
                }
            ]
        )
        return result # type: ignore
    except Exception as e:
        print(f"Error analyzing chunk: {e}")
        return []

def remove_flagged_content(text: str, flagged_items: List[FlaggedText]) -> str:
    """Remove flagged content from the text."""
    cleaned_text = text
    
    # Sort flagged items by length (longest first) to avoid partial replacements
    sorted_flags = sorted(flagged_items, key=lambda x: len(x.flagged_text), reverse=True)
    
    for flag in sorted_flags:
        # Remove the flagged text completely
        flagged_content = flag.flagged_text.strip()
        if flagged_content in cleaned_text:
            cleaned_text = cleaned_text.replace(flagged_content, "")

    return cleaned_text

def combine_chunks(chunks: List[str]) -> List[str]:
    """Combine chunks in pairs."""
    combined: List[str] = []
    for i in range(0, len(chunks), 2):
        if i + 1 < len(chunks):
            # Combine two chunks
            combined_chunk = chunks[i] + " " + chunks[i + 1]
            combined.append(combined_chunk)
        else:
            # Odd number of chunks, add the last one as is
            combined.append(chunks[i])
    return combined

def remove_duplicate_flags(flagged_items: List[FlaggedText]) -> List[FlaggedText]:
    """Remove duplicate flagged items based on flagged_text content."""
    seen_texts: set[str] = set()
    unique_flags: List[FlaggedText] = []
    
    for flag in flagged_items:
        if flag.flagged_text not in seen_texts:
            seen_texts.add(flag.flagged_text)
            unique_flags.append(flag)
    
    return unique_flags

def hierarchical_analysis(transcript: str, initial_chunks: int = 8) -> List[FlaggedText]:
    """
    Perform hierarchical analysis of the transcript:
    1. Split into chunks
    2. Analyze and clean each chunk
    3. Combine chunks in pairs
    4. Repeat until one chunk remains
    5. Return final analysis
    """
    print(f"Starting hierarchical analysis with {initial_chunks} initial chunks...")
    print(f"Original transcript length: {len(transcript.split())} words")
    
    # Split transcript into initial chunks
    chunks = split_transcript_into_chunks(transcript, initial_chunks)
    print(f"Split transcript into {len(chunks)} chunks")
    
    all_flagged_content: List[FlaggedText] = []
    level = 1
    
    while len(chunks) > 1:
        print(f"\nLevel {level}: Processing {len(chunks)} chunks...")
        
        # Analyze each chunk and remove flagged content
        cleaned_chunks: List[str] = []
        level_flagged_count = 0
        
        for i, chunk in enumerate(chunks):
            print(f"  Analyzing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words)...")
            flagged_items = analyze_chunk_with_gemini(chunk)
            print(f"    Flagged {len(flagged_items)} pieces of text in chunk {i+1}")
            print(f"    Flagged items: {flagged_items}")
            level_flagged_count += len(flagged_items)
            all_flagged_content.extend(flagged_items)
            
            # Remove flagged content from chunk
            cleaned_chunk = remove_flagged_content(chunk, flagged_items)
            cleaned_chunks.append(cleaned_chunk)
        
        print(f"  Found {level_flagged_count} flagged items at level {level}")
        
        # Combine chunks in pairs
        chunks = combine_chunks(cleaned_chunks)
        print(f"  Combined into {len(chunks)} chunks")
        level += 1
    
    # Final analysis on the last remaining chunk
    print(f"\nLevel {level}: Final analysis of combined content...")
    print(f"  Final chunk length: {len(chunks[0].split())} words")
    final_flagged_items = analyze_chunk_with_gemini(chunks[0])
    print(f"    Flagged {len(final_flagged_items)} pieces of text in final chunk")
    all_flagged_content.extend(final_flagged_items)
    print(f"  Found {len(final_flagged_items)} flagged items in final analysis")
    
    # Remove duplicates
    unique_flagged_content = remove_duplicate_flags(all_flagged_content)
    
    print(f"\nHierarchical analysis complete!")
    print(f"Total flagged items found: {len(all_flagged_content)}")
    print(f"Unique flagged items: {len(unique_flagged_content)}")
    return unique_flagged_content

def analyze_with_gemini(transcript: str) -> List[FlaggedText]:
    """Main analysis function using hierarchical approach."""
    return hierarchical_analysis(transcript)

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

def main(video_path: str, initial_chunks: int = 8) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
    
    # transcript = transcribe_video(video_path)
    transcript = read_file(r"src/prompts/transcript_1017.txt")
    print(f"Generating hierarchical analysis with {initial_chunks} initial chunks...")
    
    start_time = time.time()
    analysis = hierarchical_analysis(transcript, initial_chunks)
    end_time = time.time()

    # Convert the analysis to a JSON-like string
    # This is temporary until we setup the proper pipeline
    analysis_to_str = "[\n"
    for i, item in enumerate(analysis):
        analysis_to_str += f'  {{\n    "flagged_text": "{item.flagged_text}",\n    "reason": "{item.reason}",\n    "category": "{item.category}"\n  }}'
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
    parser.add_argument("--chunks", type=int, default=8, 
                        help="Number of initial chunks to split the transcript into (default: 8)")
    args = parser.parse_args()
    main(args.video_path, args.chunks)

if __name__ == "__main__":
    cli()