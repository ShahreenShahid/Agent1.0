"""
AutoGen Agent with NVIDIA Nemotron Model Client
"""

import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._model_info import ModelInfo

# Load environment variables
load_dotenv()

# Configuration from environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://openrouter.ai/api/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openrouter/free")

if not NVIDIA_API_KEY or NVIDIA_API_KEY == "your_nvidia_api_key_here":
    raise ValueError("Please set NVIDIA_API_KEY in .env file")

def create_model_client() -> OpenAIChatCompletionClient:
    """Create OpenAI-compatible model client for NVIDIA Nemotron."""
    model_info = ModelInfo(
        model_id=NVIDIA_MODEL,
        max_tokens=4096,
        temperature=0.7,
        vision=False,
        function_calling=True,
        json_output=False,
        family="text",
    )
    
    return OpenAIChatCompletionClient(
        model=NVIDIA_MODEL,
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        model_info=model_info,
        temperature=0.7,
        max_tokens=4096,
    )

def create_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """Create an AssistantAgent with the NVIDIA Nemotron model."""
    return AssistantAgent(
        name="nemotron_assistant",
        model_client=model_client,
        system_message=(
            "You are a helpful AI assistant powered by NVIDIA Nemotron. "
            "Provide clear, accurate, and concise responses."
        ),
    )

async def main():
    print("Connecting to Nemotron Agent...")
    model_client = create_model_client()
    agent = create_agent(model_client)
    print(f"✓ Agent created: {agent.name}")
    
    # Live task execution
    task_prompt = input("\nYou: ").strip()
    if not task_prompt:
        task_prompt = "Hello! Who are you and how can you help me today?"
    print()
    print("Waiting for Nemotron response...")
    
    result = await agent.run(task=task_prompt)
    
    print("\n--- AGENT RESPONSE ---")
    print(result.messages[-1].content)
    print("----------------------\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nExecution error: {type(e).__name__}")
        print(f"Details: {str(e)}")