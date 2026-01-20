"""
Use Case: As a company we would like to improve the quality of our emails. In particular, we would like replace the internal abbreviation and acronyms with wording more easily understood.
Approach: Use a the BaseTool package to create a ReplaceJargonTool() function to scan an replace jargon where needed. We will also instruct an Agent with a task to clean up the email and make it sound more professional.
"""

from dotenv import load_dotenv
load_dotenv()
#%%
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1
)
#%%
original_email = """
Yo! In talking with the SSC, this student claims they filled out all the empty boxes in the application. They sounded pretty good on the phone and they were certainly juiced up with excitement to fill out the application. However, it seems like a WIP. Can U tell me wassup with with this? Thx!
"""

from crewai.tools import BaseTool

class ReplaceJargonsTool(BaseTool):
    name: str = "Replace Jargons"
    description: str = "Replace jargons with more specific terms"

    def _run(self, email: str) -> str:
        replacements = {
            "WIP": "work in progress ",
            "ARB": "Architectural Review Board",
            "LD": "Learning Design",
            "POC": "proof of concept",
            "SSC": "student success coordinator",
            "Thx": "Thanks"
        }
        suggestions = []
        email_lower = email.lower()
        for jargon, replacement in replacements.items():
            if jargon.lower() in email_lower:
                suggestions.append(f"Consider replacing '{jargon}' with '{replacement}'")

        return "\n".join(suggestions) if suggestions else "No jargon or internal abbreviations detected."

jt = ReplaceJargonsTool()
jt.run(original_email)

#%%
# import from CrewAI to get agent setup

from crewai import Agent, Task, Crew

email_assistant = Agent(
    role="Email Assistant Agent",
    goal="Improve Everspring emails and make them sound professional and clear",
    backstory="A highly experienced communication expert skilled in professional email writing for higher education universities",
    tools=[jt],
    verbose=True,
    llm=llm
)

email_task = Task(
    description=f"""Take the following rough email and rewrite it into a professional and polished version.
    Expand abbreviations:
    '''{original_email}'''""",
    agent=email_assistant,
    expected_output="A professional written email with proper formatting and content.",
)

crew = Crew(
    agents=[email_assistant],
    tasks=[email_task],
    verbose=True
)

result = crew.kickoff()
print(result)