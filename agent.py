import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. LOAD ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# 2. CONNECT TO GROQ
# ==========================================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"


# ==========================================
# 3. BLITZ RESEARCH SYSTEM
# ==========================================

SYSTEM_PROMPT = """
You are Blitz Research, an AI research assistant
designed for MBBS students.

Your job is NOT to behave like a normal chatbot.

You are an agent.

You must:

1. Understand the user's medical research question.
2. Decide what information is required.
3. Use available tools when necessary.
4. Examine tool results.
5. Decide what to do next.
6. Continue researching when evidence is insufficient.
7. Never invent medical sources or evidence.
8. Clearly separate established information from uncertainty.
9. If the available information is insufficient, say what
   additional information is needed.

You are primarily an educational and research assistant
for medical students, not a replacement for a clinician.

For clinical cases:
- Do not claim certainty from symptoms alone.
- Consider differential diagnoses.
- Distinguish evidence from inference.
- Do not promise a guaranteed cure.
- Do not manufacture a confidence percentage.
- A high evidence threshold means stronger supporting evidence,
  not a guarantee that a diagnosis or treatment is correct.

Your goal is evidence-backed medical research.
"""


# ==========================================
# 4. TEST TOOL
# ==========================================

def get_research_status():
    """
    Returns the current status of the Blitz Research
    research engine.
    """

    return {
        "system": "Blitz Research",
        "status": "operational",
        "research_engine": "ready",
        "message": "Medical research agent is ready."
    }


# ==========================================
# 5. TOOL DEFINITIONS
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_research_status",
            "description": (
                "Check whether the Blitz Research "
                "research engine is operational."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ==========================================
# 6. TOOL REGISTRY
# ==========================================

available_functions = {
    "get_research_status": get_research_status
}


# ==========================================
# 7. EXECUTE TOOL
# ==========================================

def execute_tool(tool_call):

    function_name = tool_call.function.name

    arguments = json.loads(
        tool_call.function.arguments or "{}"
    )

    if function_name not in available_functions:
        return {
            "error": f"Unknown tool: {function_name}"
        }

    function = available_functions[function_name]

    try:
        result = function(**arguments)

        return result

    except Exception as e:

        return {
            "error": str(e)
        }


# ==========================================
# 8. AGENT LOOP
# ==========================================

def run_agent(user_input):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_input
        }

    ]

    MAX_ITERATIONS = 8

    for iteration in range(MAX_ITERATIONS):

        print()
        print("=" * 60)
        print(f"AGENT STEP {iteration + 1}")
        print("=" * 60)

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            tools=tools,

            tool_choice="auto"

        )

        assistant_message = response.choices[0].message

        # --------------------------------------
        # Add assistant response to conversation
        # --------------------------------------

        messages.append(assistant_message)

        # --------------------------------------
        # If there are NO tool calls,
        # the agent has finished.
        # --------------------------------------

        if not assistant_message.tool_calls:

            print("\nFINAL ANSWER:\n")

            print(assistant_message.content)

            return assistant_message.content

        # --------------------------------------
        # Execute requested tools
        # --------------------------------------

        for tool_call in assistant_message.tool_calls:

            print("\nTOOL REQUEST:")
            print("Tool:", tool_call.function.name)
            print(
                "Arguments:",
                tool_call.function.arguments
            )

            result = execute_tool(tool_call)

            print("\nTOOL RESULT:")
            print(result)

            # ----------------------------------
            # Send tool result back to AI
            # ----------------------------------

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "name": tool_call.function.name,

                "content": json.dumps(result)

            })

    return (
        "The research process reached the maximum "
        "number of steps."
    )


# ==========================================
# 9. START BLITZ RESEARCH
# ==========================================

if __name__ == "__main__":

    user_question = input(
        "\nEnter your research problem:\n> "
    )

    run_agent(user_question)