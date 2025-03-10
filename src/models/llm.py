from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama


def get_response(gpt_model: str, context: str, prompt_template: str):
    """
    Generate a response to a given question based on the provided document."
    """
    try:
        llm = Ollama(
            model=gpt_model,  # Or your desired model
            base_url="http://10.203.13.225:11434",  # Replace with the remote host's IP address
        )

        # Create a prompt template for unstructured markdown output
        prompt_template = PromptTemplate(f"{prompt_template}")
        prompt = prompt_template.format(context=context)

        # Get the response from the model
        return llm.complete(prompt=prompt)
    except Exception as e:
        # Handle any errors that may occur during context generation
        raise RuntimeError(f"Error during context generation: {str(e)}")
