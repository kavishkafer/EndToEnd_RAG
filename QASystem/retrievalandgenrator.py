import time
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack import Pipeline
from QASystem.ingestion import ingest
from QASystem.utility import pinecone_config
import os
from dotenv import load_dotenv
from huggingface_hub.utils._errors import HfHubHTTPError

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

def get_result(query, retries=3, wait_time=60):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=pinecone_config()))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    query_pipeline.add_component("llm", HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1", token=Secret.from_token("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")))

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "llm")

    for attempt in range(retries):
        try:
            results = query_pipeline.run(
                {
                    "text_embedder": {"text": query},
                    "prompt_builder": {"query": query},
                }
            )
            return results['llm']['replies'][0]
        except HfHubHTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            else:
                print(f"HTTP error occurred: {e}")
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    raise Exception("Max retries exceeded")

if __name__ == '__main__':
    result = get_result("what is requirement for housing loan")
    print(result)
