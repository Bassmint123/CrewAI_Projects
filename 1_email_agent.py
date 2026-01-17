from dotenv import load_dotenv
load_dotenv()
#%%
from crewai import LLM
"""
# Google Gemini
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1
)
"""

# OpenAI ChatGPT
llm = LLM(
    model="openai/gpt-4o",
    temperature=0.1,
)

"""
# Anthropic Claude
llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    base_url="https://api.anthropic.com",  # optional; default is correct for Anthropic
    temperature=0.7,
    max_tokens=4096,  # required for Anthropic
)
"""

llm.call("Who won the 2025 super bowl?")
#%%
from crewai import Agent, Task, Crew

email_assistant = Agent(
    role="Email Assistant Agent",
    goal="Improve Everspring emails and make them sound professional and clear",
    backstory="A highly experienced communication expert skilled in professional email writing for higher education universities",
    verbose=True,
    llm=llm
)

original_email = """
Yo! This student claims they filled out all the empty boxes in the application. They sounded pretty good on the phone and they were certainly juiced up with excitement to fill out the application. Can U tell me wassup with with this? Thx!
"""

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