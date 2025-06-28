import datasets
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




