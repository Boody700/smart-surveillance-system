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

def run_agent3_auditor(image_path, person_id, assigned_zone, image_path_2=None):
    """
    Analyze a violation frame using LLaVA.
    
    image_path   — frame when violation was detected (person leaving / absent)
    image_path_2 — optional: frame when person returned (for AFK/LEFT violations)
    """

    if image_path_2:
        # Two-image prompt — compare before and after
        prompt = (
            f"You are a workplace security auditor reviewing two CCTV frames. "
            f"COLOR GUIDE: "
            f"RED outlined zone = Person {person_id}'s assigned desk (Zone {assigned_zone}). "
            f"BLUE outlined zones = other people's desks. "
            f"BRIGHT GREEN rectangle = Person {person_id} being analyzed. "
            f"FIRST IMAGE = when the person left or was absent. "
            f"SECOND IMAGE = when the person returned or was last seen. "
            f"Analyze both images and classify what happened: "
            f"'AFK' = person was absent from their zone and returned. "
            f"'LEFT' = person left and did not return. "
            f"'OTHER_ZONE' = person moved to someone else's blue zone. "
            f"'LOITERING' = person was in open space between zones. "
            f"'INSIDE' = person was actually in their correct zone (false alarm). "
            f"Output ONLY one word. No explanation."
        )
        images = [image_path, image_path_2]
    else:
        # Single image prompt
        prompt = (
            f"You are a workplace security auditor reviewing a CCTV frame. "
            f"COLOR GUIDE: "
            f"RED outlined zone = Person {person_id}'s assigned desk (Zone {assigned_zone}). "
            f"BLUE outlined zones = other people's Zones. "
            f"BRIGHT GREEN rectangle = Person {person_id} being analyzed , if Theres no green rectangle, the person is not visible in the frame which means he's either afk or left the area depending on whether he was seen in Future Frames or not. "
            f"Classify Person {person_id}'s current status: "
            f"'OTHER_ZONE' = person is inside a blue zone (someone else's desk). "
            f"'LOITERING' = person is present and in an open space with no zone around them. "
            f"'AFK' = the person assigned to the RED zone is not sitting in their designated chair, even if the desk/laptop is still there. "
            f"'LEFT' = person has left the area and is no longer visible. "
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
        valid  = {"INSIDE", "OTHER_ZONE", "LOITERING", "AFK", "LEFT"}
        return answer if answer in valid else "UNKNOWN"
    except Exception as e:
        print(f"  [Agent3 ERROR] {e}")
        return "UNKNOWN"