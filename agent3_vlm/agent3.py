import ollama

def run_agent3_auditor(image_path, person_id, assigned_zone):
    # Enforce strict labels
    prompt = (
        f"You are a security auditor. The image shows a person in a green box. "
        f"The red zone is their assigned area; blue zones are others. "
        f"Classify the person's status: "
        f"1. 'INSIDE' if they are within their assigned red zone. "
        f"2. 'OTHER' if they are inside a blue zone. "
        f"3. 'LOITERING' if they are in an area with no zone lines. "
        f"4. 'AFK' if the person is missing. "
        f"Output ONLY one word from: INSIDE, OTHER, LOITERING, AFK."
    )
    try:
        response = ollama.chat(model='llava', messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}])
        answer = response['message']['content'].strip().upper()
        # Filter for allowed words
        valid_options = ["INSIDE", "OTHER", "LOITERING", "AFK"]
        return answer if answer in valid_options else "UNKNOWN"
    except:
        return "UNKNOWN"