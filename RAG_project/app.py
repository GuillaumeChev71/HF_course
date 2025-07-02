import datasets
# Import necessary libraries
import random
from smolagents import CodeAgent, InferenceClientModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from langchain.docstore.document import Document
from retriever import GuestInfoRetrieverTool
from smolagents import CodeAgent, LiteLLMModel

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)

# Initialize the Hugging Face model
model = LiteLLMModel(
        model_id="ollama_chat/qwen2:7b",  # le modèle tel que listé par ollama
        api_key="ollama"                 # clé spéciale pour identificaton locale
)


# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()


# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)

query = "Tell me about 'Lady Ada Lovelace'"
response = alfred.run(query)

print("Alfred's Response:")
print(response)


