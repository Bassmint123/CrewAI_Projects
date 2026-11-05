"""This is a sample of CrewAI agent code for a “writer agent” whose task is to post a message saying “Hello Everspring!” on Slack using a JSON-RPC tool request to AWS AgentCore. This shows how you can set up the agent, define the task, and leverage AgentCore’s tool routing (assuming the BedrockInvokeAgentTool/AgentCore integration is set up for Slack posting).

1. This sets up a tool-based integration for CrewAI to AgentCore, specifying a JSON-RPC endpoint and Slack post parameters.

2. The agent uses the BedrockInvokeAgentTool to securely route the Slack API call through AgentCore, leveraging AWS identity and session handling.

3. The agent’s only goal is to post “Hello World!” to the specified Slack channel; AgentCore handles the actual API invocation

4. Outside of Crew, the RPC request would look like this:
{
  "jsonrpc": "2.0",
  "method": "slack_send_message",
  "params": {
    "channel": "#tech-general",
    "text": "Hello Everspring!"
  },
  "id": "writer-agent-001"
}
"""
#%%
from dotenv import load_dotenv
load_dotenv()
#%%
from crewai import Agent, Task, Crew

# This is the specific tool being imported. It acts as a bridge, allowing a CrewAI agent to delegate tasks to an autonomous, managed agent on Amazon Bedrock.
from crewai_tools import BedrockInvokeAgentTool

# Configure the JSON-RPC tool for AWS AgentCore with Slack
slack_post_tool = BedrockInvokeAgentTool(
    agentcore_endpoint="https://whatever-agentcore-endpoint.amazonaws.com/jsonrpc",
    agentcore_api_key="OUR_AGENTCORE_API_KEY",
# Tool specification for the Slack post
    tool_action="slack_send_message",
    tool_params={
        "channel": "#tech-general",
        "text": "Hello Everspring!"
    }
)

# Define the writer agent
writer_agent = Agent(
    role="Writer",
    goal="Post a greeting message to Slack.",
    backstory="An AI agent specializing in quick team updates through Slack.",
    tools=[slack_post_tool]  # Attach the Slack RPC posting tool
)

# Define the task the agent must complete
post_message_task = Task(
    description="Use AgentCore JSON-RPC to send 'Hello World!' to #general on Slack.",
    agent=writer_agent,
    expected_output="Confirmation that the message was posted in Slack."
)

# Set up and run the Crew
crew = Crew(
    agents=[writer_agent],
    tasks=[post_message_task]
)
crew.kickoff()