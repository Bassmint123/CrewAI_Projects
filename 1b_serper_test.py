from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1
)
llm.call("Who is the current US President?")

#%%
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

email_assistant = Agent(
    role="Email Assistant Agent",
    goal="Search Google",
    backstory="A highly experienced communication expert skilled in professional search engines",
    tools=[SerperDevTool()],
    verbose=True,
    llm=llm
)

email_task = Task(
    description=f"""Find out who is the currently elected US President of the United States?""",
    agent=email_assistant,
    expected_output="The name of the current US President.",
)

crew = Crew(
    agents=[email_assistant],
    tasks=[email_task],
    verbose=True
)

result = crew.kickoff()
print(result)