# agent3_vlm/agent3.py
# Agent 3: VLM Auditor using LLaVA via Ollama.
# Supports single image OR two images (before + after) for richer analysis.
#
# Color legend in frames (matches agent2.py drawing):
#   RED zone    = person's assigned home zone
#   BLUE zones  = other zones
#   GREEN box   = person being analyzed
#   White on dark red = violation label
#   White on black    = duration text

import ollama

def run_agent3_auditor(image_path, person_id, assigned_zone):
    """
    Analyze a violation frame using LLaVA.
    
    image_path   — frame when violation was detected (person leaving / absent)
    """

        # Single image prompt
    prompt = (
            f"You are a workplace security auditor reviewing a CCTV frame. "
            f"COLOR GUIDE: "
            f"RED outlined zone = Person {person_id}'s assigned desk (Zone {assigned_zone}). "
            f"BLUE outlined zones = other people's Zones. "
            f"BRIGHT GREEN rectangle = Person {person_id} being analyzed , if Theres no green rectangle, the person is not visible in the frame which means he's either afk or left the area depending on whether he was seen in Future Frames or not. "
            f"Classify Person {person_id}'s current status: "
            f"'OTHER_ZONE' = person is inside a blue zone (someone else's desk). "
            f"'LOITERING' = person with the green frame is present and in an open space not inside any zones, with no zone around them. "
            f"'AFK' = the person assigned to the RED zone is not sitting in their designated chair, even if the desk/laptop is still there. "
            f"Output ONLY one word. No explanation."
        )
    images = [image_path]

    try:
        response = ollama.chat(
            model='llava',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': images
            }]
        )
        answer = response['message']['content'].strip().upper()
        valid  = {"INSIDE", "OTHER_ZONE", "LOITERING", "AFK"}
        return answer if answer in valid else "UNKNOWN"
    except Exception as e:
        print(f"  [Agent3 ERROR] {e}")
        return "UNKNOWN"