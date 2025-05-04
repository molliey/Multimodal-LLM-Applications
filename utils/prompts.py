def build_prompt(transcript, slide_text):
    return f"""
Transcript:
{transcript}

Slide Content:
{slide_text}

Please generate:
1. Bullet-point summary of lecture
2. Clarification of difficult terms
3. Suggested follow-up questions for students
"""