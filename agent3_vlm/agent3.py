import ollama

VLM_MODEL = "qwen2.5vl:7b"


def run_agent3_auditor(image_path, person_id, assigned_zone):
    """
    Analyze a violation frame using the VLM.

    NOTE:
    Agent 2 already detects the violation deterministically.
    Agent 3 is only used as a visual confirmation.
    """

    prompt = f"""
You are analysing an annotated workplace CCTV frame.

VISUAL LEGEND
-------------
RED outlined zone:
    Person {person_id}'s assigned workstation (Zone {assigned_zone}).

BLUE outlined zones:
    Other employees' assigned workstations.

BRIGHT GREEN bounding box:
    Person {person_id}.
    There is NEVER more than one green box.

TASK
----
Follow these steps EXACTLY.

STEP 1
Check whether a BRIGHT GREEN bounding box exists anywhere in the image.

If NO green box exists,
respond with exactly:

AFK

Do NOT guess where the employee went.
Do NOT analyse any other people.
Do NOT continue to Step 2.

STEP 2
If a GREEN box exists:

Determine where the GREEN box is located.

If the GREEN box is inside any BLUE zone:

OTHER_ZONE

If the GREEN box is outside ALL coloured zones:

LOITERING

OUTPUT RULES
------------
Return ONLY ONE of these words.

AFK
OTHER_ZONE
LOITERING

Do not explain.
Do not use punctuation.
Do not return any other text.
"""

    try:
        response = ollama.chat(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }]
        )

        answer = response["message"]["content"].strip().upper()

        valid = {"AFK", "OTHER_ZONE", "LOITERING"}

        if answer in valid:
            return answer

        return "UNKNOWN"

    except Exception as e:
        print(f"[Agent3 ERROR] {e}")
        return "UNKNOWN"