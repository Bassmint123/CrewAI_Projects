from dotenv import load_dotenv

load_dotenv()
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1
)

#%%
from crewai import Agent, Task, Crew

# Setup the agents with their roles, goals, and backstories.
research_agent = Agent(
    role="Research Specialist",
    goal="Research interesting facts about the topic: {topic}",
    backstory="You are an expert in finding relevant and factual data",
    verbose=True,
    llm=llm
)

writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research",
    backstory="You are skilled at writing engaging summaries based upon the provided content",
    verbose=True,
    llm=llm
)
#%%
# Setup the tasks for the Agents.
task1 = Task(
    # Keep in mind the { bracket around topic means it is dynamic.
    description="Find 3 to 5 interesting and recent facts about {topic}.",
    agent=research_agent,
    expected_output="A bullet list of 3 to 5 facts",
)
task2 = Task(
    description="Write a 100-word blog post summary about {topic} using the facts from the research.",
    agent=writer_agent,
    expected_output="A blog post summary",
    context=[task1]
)

# Setup the crew of agents with corresponding tasks.
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[task1, task2],
    verbose=True,
)

crew.kickoff(inputs={"topic": "the future of student enrollment in Texas A&M's University School of Law Juris Doctor (J.D.) degree program"})