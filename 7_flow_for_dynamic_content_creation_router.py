from dotenv import load_dotenv
load_dotenv()
#%%
# Define the State Model for the Flow

from typing import Dict, Any
from pydantic import BaseModel

class ContentState(BaseModel):
  """
  State model that tracks information throughout the content creation workflow.

  Think of this as a form that gets filled out as the workflow progresses:
  - Start: Only URL is filled
  - After routing: Content type is determined
  - After processing: Final content is ready
  - Throughout: Metadata can be added by any step
  """
  url: str = "" # The source website we're analyzing
  content_type: str = ""  # "blog", "newsletter", or "linkedin"
  final_content: str = ""  # The final content after processing
  metadata: Dict[str, Any] = {}  # Arbitrary metadata
#%%
# 1. Blog content creation agents

from crewai import Agent, LLM
from crewai_tools import SerperDevTool

def create_blog_agents():
    """Create agents specialized for blog content"""

    blog_researcher = Agent(
        role="Blog Content Researcher",
        goal="Extract and analyze web content to identify key insights for blog posts",
        backstory="""You are an expert content researcher who specializes in analyzing
        web content and identifying the most valuable insights for creating engaging blog posts.
        You excel at understanding complex topics and breaking them down into digestible content.""",
        verbose=True,
        tool=[SerperDevTool()],
        llm=LLM(model="gemini/gemini-2.0-flash"),
        max_iter=5
    )

    blog_writer = Agent(
        role="Blog Content Writer",
        goal="Transform research into engaging, well-structured blog posts",
        backstory="""You are a skilled blog writer with expertise in creating compelling content
        that engages readers and drives meaningful discussions. You excel at taking complex
        information and making it accessible and interesting.""",
        verbose=True,
        llm=LLM(model="gemini/gemini-2.0-flash")
    )

    return blog_researcher, blog_writer
#%%
# 2. Newsletter content creation agents

def create_newsletter_agents():
    """Create agents specialized for newsletter content"""

    newsletter_researcher = Agent(
        role="Newsletter Content Researcher",
        goal="Extract key insights from web content for newsletter format",
        backstory="""You are an expert at identifying the most newsworthy and actionable
        insights from web content. You understand what makes content valuable for newsletter
        subscribers and how to present information concisely.""",
        verbose=True,
        tool=[SerperDevTool()],
        llm=LLM(model="gemini/gemini-2.0-flash"),
        max_iter=5
    )

    newsletter_writer = Agent(
        role="Newsletter Writer",
        goal="Create engaging newsletter content that provides immediate value",
        backstory="""You are a newsletter specialist who knows how to craft content that
        busy professionals want to read. You excel at creating scannable, actionable content
        with clear takeaways.""",
        verbose=True,
        llm=LLM(model="gemini/gemini-2.0-flash")
    )

    return newsletter_researcher, newsletter_writer
#%%
# 3. LinkedIn content creation agents
def create_linkedin_agents():
    """Create agents specialized for LinkedIn content"""

    linkedin_researcher = Agent(
        role="LinkedIn Content Researcher",
        goal="Extract professional insights suitable for LinkedIn audience",
        backstory="""You are an expert at identifying professional insights and industry
        trends that resonate with LinkedIn's professional audience. You understand what
        content drives engagement on professional networks.""",
        verbose=True,
        tool=[SerperDevTool()],
        llm=LLM(model="gemini/gemini-2.0-flash"),
        max_iter=5
    )

    linkedin_writer = Agent(
        role="LinkedIn Content Writer",
        goal="Create engaging LinkedIn posts that drive professional engagement",
        backstory="""You are a LinkedIn content specialist who knows how to craft posts
        that get noticed in the professional feed. You excel at creating content that
        sparks meaningful professional discussions.""",
        verbose=True,
        llm=LLM(model="gemini/gemini-2.0-flash")
    )

    return linkedin_researcher, linkedin_writer
#%%
# 1. Blog writing Task

from crewai import Task

def create_blog_tasks(researcher, writer, url):
    """Create tasks for blog content generation"""

    research_task = Task(
        description=f"""
        Analyze the content from {url} and extract key insights for a blog post.
        Your analysis should identify:
        1. Main themes and key points
        2. Interesting insights or data points
        3. Potential angles for blog content
        4. Target audience considerations
        5. SEO-worthy topics and keywords

        Provide a comprehensive research summary that will guide blog writing.
        """,
        expected_output="A detailed research summary with key insights, themes, and recommendations for blog content",
        agent=researcher
    )

    writing_task = Task(
        description="""
        Create an engaging blog post based on the research findings.

        Requirements:
        - 800-1200 words
        - Engaging headline
        - Clear introduction with hook
        - Well-structured body with subheadings
        - Actionable insights or takeaways
        - Strong conclusion
        - SEO-optimized content
        - Professional yet accessible tone

        Format the output in markdown.
        """,
        expected_output="A complete, well-structured blog post in markdown format",
        agent=writer,
        context=[research_task]
    )

    return [research_task, writing_task]
#%%
# 2. Newsletter writing Task

def create_newsletter_tasks(researcher, writer, url):
    """Create tasks for newsletter content generation"""

    research_task = Task(
        description=f"""
        Analyze the content from {url} and extract the most newsworthy insights for a newsletter.
        Focus on:
        1. Most important news or updates
        2. Actionable insights subscribers can use immediately
        3. Key statistics or data points
        4. Industry implications
        5. Quick takeaways for busy professionals

        Prioritize information that provides immediate value.
        """,
        expected_output="A focused research summary highlighting the most valuable and actionable insights",
        agent=researcher
    )

    writing_task = Task(
        description="""
        Create a compelling newsletter section based on the research.

        Requirements:
        - 400-600 words
        - Catchy subject line
        - Scannable format with bullet points
        - Clear action items or takeaways
        - Conversational yet professional tone
        - Include relevant links or resources
        - End with a clear call-to-action

        Format for easy reading in email.
        """,
        expected_output="A complete newsletter section with subject line and formatted content",
        agent=writer,
        context=[research_task]
    )

    return [research_task, writing_task]
#%%
# 3. LinkedIn post writing Task
def create_linkedin_tasks(researcher, writer, url):
    """Create tasks for LinkedIn content generation"""

    research_task = Task(
        description=f"""
        Analyze the content from {url} and extract insights suitable for LinkedIn audience.
        Consider what would engage LinkedIn's professional audience based on the content.
        """,
        expected_output="Research summary focused on professional insights and engagement opportunities",
        agent=researcher
    )

    writing_task = Task(
        description="""
        Create an engaging LinkedIn post based on the research.

        Requirements:
        - 150-300 words (optimal LinkedIn length)
        - Professional yet conversational tone
        - Include relevant hashtags (3-5)
        - Pose a question to encourage engagement
        - Share a key insight or lesson learned from the content
        - Use line breaks for readability
        - Include a call-to-action for comments

        Make it shareable and discussion-worthy.
        """,
        expected_output="A complete LinkedIn post with hashtags and engagement elements",
        agent=writer,
        context=[research_task]
    )

    return [research_task, writing_task]
#%%
# ContentRouterFlow Class Analysis

"""This class orchestrates the entire content creation pipeline.

Class Overview

ContentRouterFlow is a subclass of Flow[ContentState] that implements an event-driven workflow for dynamic content creation routing. It demonstrates CrewAI's Flow architecture with state management and specialized crew routing.
Key Decorators & Their Roles
💾 @persist()

Purpose: Enables automatic state persistence across workflow execution
Behavior: Saves flow state to SQLite database by default
Benefits: Workflow recovery, state tracking, and debugging capabilities
Scope: Applied at class level to persist all method states
🚀 @start()

Purpose: Marks the workflow entry point
Function: Collects user input (URL + content type) and initializes state
Triggers: Automatic when flow begins
🧭 @router(get_user_input)

Purpose: Creates decision point for workflow routing
Logic: Returns self.state.content_type to determine next path
Output: Routes to "blog", "newsletter", or "linkedin" listeners
👂 @listen("event_name")

Purpose: Creates event listeners for specific triggers
Behavior: Methods execute when their specified event occurs
Examples: @listen("blog"), @listen("newsletter"), @listen("linkedin")
Workflow:
The workflow starts with @start() on get_user_input().
The result of get_user_input() is routed by @router() to one of the process_*_content methods based on the content_type.
Whichever process_*_content method runs (e.g., process_blog_content), it will eventually complete and save the result to the flow's state."""

from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.persistence import persist
from crewai import Crew

@persist(verbose=True)
class ContentRouterFlow(Flow[ContentState]):
    """
    A dynamic workflow that routes content creation to specialized crews.

    Flow Overview:
    1. START: Get user input (URL + content type)
    2. ROUTE: Direct to appropriate content crew
    3. PROCESS: Execute specialized content creation
    4. FINISH: Return the final content

    This flow demonstrates:
    - Event-driven architecture with decorators
    - State management across workflow steps
    - Dynamic routing based on user input
    - Parallel processing capabilities
    """

    @start()
    def get_user_input(self):
        """Get URL and desired content type from user"""
        # Enter your target URL here
        url = input("Enter your target URL: ") # example: https://www.everspringpartners.com/news

        # Enter your content type: blog, newsletter, or LinkedIn
        content_type = input("Enter your desired content type (blog, newsletter, or linkedin): ")

        # Store in state
        self.state.url = url
        self.state.content_type = content_type

        return "Input collected"

    @router(get_user_input)
    def route_to_crew(self, previous_result):
        """Route to appropriate crew based on content type"""
        return self.state.content_type

    @listen("blog")
    def process_blog_content(self):
        """Process content using blog crew"""
        # Create blog agents
        researcher, writer = create_blog_agents()

        # Create blog tasks
        tasks = create_blog_tasks(researcher, writer, self.state.url)

        # Create and run crew
        blog_crew = Crew(
            agents=[researcher, writer],
            tasks=tasks,
            verbose=True
        )

        result = blog_crew.kickoff()
        self.state.final_content = result.raw

        return "Blog content created"

    @listen("newsletter")
    def process_newsletter_content(self):
        """Process content using newsletter crew"""

        # Create newsletter agents
        researcher, writer = create_newsletter_agents()

        # Create newsletter tasks
        tasks = create_newsletter_tasks(researcher, writer, self.state.url)

        # Create and run crew
        newsletter_crew = Crew(
            agents=[researcher, writer],
            tasks=tasks,
            verbose=True
        )

        result = newsletter_crew.kickoff()
        self.state.final_content = result.raw

        return "Newsletter content created"

    @listen("linkedin")
    def process_linkedin_content(self):
        """Process content using LinkedIn crew"""

        # Create LinkedIn agents
        researcher, writer = create_linkedin_agents()

        # Create LinkedIn tasks
        tasks = create_linkedin_tasks(researcher, writer, self.state.url)

        # Create and run crew
        linkedin_crew = Crew(
            agents=[researcher, writer],
            tasks=tasks,
            verbose=True
        )

        result = linkedin_crew.kickoff()
        self.state.final_content = result.raw

        return "LinkedIn content created"
#%%
# Run the flow

import asyncio

# Create and run the flow
flow = ContentRouterFlow()
result = await flow.kickoff_async()

#%%
# Access the final content saved in our state via flow.state.final_content
flow.state
from IPython.display import display, Markdown

# Display the actual content
display(Markdown("---"))

# Format the content nicely for notebook
content = str(flow.state.final_content)
display(Markdown(content))