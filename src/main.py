import yaml

from data.preprocess import clean_context
from data.readers import read_json
from models.llm import get_response
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("skip", "read_json_data") as check, check():
    data_path = "data/data.json"
    data = read_json(data_path, key="FactualNarrative")


with skip_run("run", "llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    gpt_model = "qwen2.5:72b"

    io_prompts = yaml.load(open(str("prompts/io.yaml")), Loader=yaml.SafeLoader)
    for context in contexts[0:3]:
        for prompt in io_prompts:
            context = clean_context(context)
            output = get_response(gpt_model, context, io_prompts[prompt])
            print(output)
