import os
import re

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from data.preprocess import clean_context
from data.readers import read_json
from models.llm import get_response
from utils import skip_run

# The configuration file
with open("./configs/config.yaml") as f:
    general_config = yaml.load(f, Loader=yaml.SafeLoader)
    # Configure API keys
    os.environ["OPENAI_API_KEY"] = general_config["openai_api_key"]


with skip_run("skip", "read_json_data") as check, check():
    data_path = "data/data.json"
    data = read_json(data_path, key="FactualNarrative")


with skip_run("skip", "input_output_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    # gpt_model = "qwen2.5:32b-instruct"
    gpt_model = "gpt-4o-mini"

    io_prompts = yaml.load(open(str("prompts/io.yaml")), Loader=yaml.SafeLoader)
    output = pd.DataFrame(columns=["document_id", "prompt", "result"])

    # Reports to drop:
    reports_to_drop = []

    for i, context in enumerate(tqdm(contexts)):
        try:
            context = clean_context(context)
            for prompt in io_prompts:
                result = get_response(
                    gpt_model, context, io_prompts[prompt], model_type="gpt"
                ).text
                if np.isnan(output.index.max() + 1):
                    output.loc[0] = [i, prompt, result]
                else:
                    output.loc[output.index.max() + 1] = [i, prompt, result]
        except Exception:
            if str(i) not in reports_to_drop:
                reports_to_drop.append(str(i))

            if i % 10 == 0:
                # Save the reports to drop
                with open("data/io_reports_to_drop.txt", "w") as outfile:
                    outfile.write("\n".join(reports_to_drop))

                # Save the dictionary
                output.to_csv("data/io_results.csv")

    # Save the dictionary
    output.to_csv("data/io_results.csv")


with skip_run("skip", "input_output_merged_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    # gpt_model = "qwen2.5:32b-instruct"
    gpt_model = "gpt-4o-mini"

    io_prompts = yaml.load(open(str("prompts/io_merged.yaml")), Loader=yaml.SafeLoader)
    output = pd.DataFrame(columns=["document_id", "prompt", "result"])

    # Reports to drop:
    reports_to_drop = []
    prompt = "merged_queries"

    for i, context in enumerate(tqdm(contexts)):
        try:
            context = clean_context(context)
            result = get_response(
                gpt_model, context, io_prompts[prompt], model_type="gpt"
            ).text
            if np.isnan(output.index.max() + 1):
                output.loc[0] = [i, prompt, result]
            else:
                output.loc[output.index.max() + 1] = [i, prompt, result]
        except Exception:
            if str(i) not in reports_to_drop:
                reports_to_drop.append(str(i))

            # Save the dictionary
            output.to_csv("data/io_merged_results.csv")

    # Save the dictionary
    output.to_csv("data/io_merged_results.csv")


with skip_run("skip", "input_output_expanded_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    # gpt_model = "qwen2.5:32b-instruct"
    gpt_model = "gpt-4o-mini"

    io_prompts = yaml.load(
        open(str("prompts/io_expanded.yaml")), Loader=yaml.SafeLoader
    )
    output = pd.DataFrame(columns=["document_id", "prompt", "result"])

    # Reports to drop:
    reports_to_drop = []

    for i, context in enumerate(tqdm(contexts)):
        try:
            context = clean_context(context)
            for prompt in io_prompts:
                result = get_response(
                    gpt_model, context, io_prompts[prompt], model_type="gpt"
                ).text
                if np.isnan(output.index.max() + 1):
                    output.loc[0] = [i, prompt, result]
                else:
                    output.loc[output.index.max() + 1] = [i, prompt, result]
        except Exception:
            if str(i) not in reports_to_drop:
                reports_to_drop.append(str(i))

            if i % 10 == 0:
                # Save the reports to drop
                with open("data/io_expanded_reports_to_drop.txt", "w") as outfile:
                    outfile.write("\n".join(reports_to_drop))

                # Save the dictionary
                output.to_csv("data/io_expanded_results.csv")

    # Save the dictionary
    output.to_csv("data/io_expanded_results.csv")


with skip_run("skip", "input_output_expanded_merged_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    # gpt_model = "qwen2.5:32b-instruct"
    gpt_model = "gpt-4o-mini"

    io_prompts = yaml.load(
        open(str("prompts/io_expanded_merged.yaml")), Loader=yaml.SafeLoader
    )
    output = pd.DataFrame(columns=["document_id", "prompt", "result"])

    # Reports to drop:
    reports_to_drop = []
    prompt = "merged_queries"

    for i, context in enumerate(tqdm(contexts)):
        try:
            context = clean_context(context)
            result = get_response(
                gpt_model, context, io_prompts[prompt], model_type="gpt"
            ).text
            if np.isnan(output.index.max() + 1):
                output.loc[0] = [i, prompt, result]
            else:
                output.loc[output.index.max() + 1] = [i, prompt, result]
        except Exception:
            if str(i) not in reports_to_drop:
                reports_to_drop.append(str(i))

            # Save the dictionary
            output.to_csv("data/io_expanded_merged_results.csv")

    # Save the dictionary
    output.to_csv("data/io_expanded_merged_results.csv")


with skip_run("skip", "cot_llm_query") as check, check():
    data_path = "data/data.json"
    contexts = read_json(data_path, key="FactualNarrative")

    # Set the GPT model to use
    # gpt_model = "qwen2.5:32b-instruct"
    gpt_model = "gpt-4o-mini"

    io_prompts = yaml.load(open(str("prompts/cot.yaml")), Loader=yaml.SafeLoader)
    output = pd.DataFrame(columns=["document_id", "prompt", "result"])

    # Reports to drop:
    reports_to_drop = []

    for i, context in enumerate(tqdm(contexts)):
        try:
            context = clean_context(context)
            for prompt in io_prompts:
                result = get_response(
                    gpt_model, context, io_prompts[prompt], model_type="gpt"
                ).text
                if np.isnan(output.index.max() + 1):
                    output.loc[0] = [i, prompt, result]
                else:
                    output.loc[output.index.max() + 1] = [i, prompt, result]
        except Exception:
            if str(i) not in reports_to_drop:
                reports_to_drop.append(str(i))

        # Save the reports to drop
        with open("data/cot_reports_to_drop.txt", "w") as outfile:
            outfile.write("\n".join(reports_to_drop))

        # Save the dictionary
        output.to_csv("data/cot_results.csv")

    # Save the dictionary
    output.to_csv("data/cot_results.csv")


with skip_run("skip", "consolidate_data_io") as check, check():
    df = pd.read_csv("data/raw/io_results.csv")[["prompt", "result"]]
    df["result"] = df["result"].replace({"YES": 1, "NO": 0})
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.groupby(["prompt"]).sum()

    # Load manual data
    manual_df = pd.read_csv("data/raw/manual.csv")

    final_df = pd.merge(df, manual_df, on="prompt")
    print(final_df)


with skip_run("skip", "consolidate_data_io_expanded") as check, check():
    df = pd.read_csv("data/raw/io_expanded_results.csv")[["prompt", "result"]]
    df["result"] = df["result"].replace({"YES": 1, "NO": 0})
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.groupby(["prompt"]).sum()

    # Load manual data
    manual_df = pd.read_csv("data/raw/manual.csv")

    final_df = pd.merge(df, manual_df, on="prompt")
    print(final_df)


with skip_run("skip", "consolidate_data_cot") as check, check():
    df = pd.read_csv("data/raw/cot_results.csv")[["prompt", "result"]]
    df = df[df["prompt"].str.contains("detailed")]

    factors = {
        "supervisory_factors_detailed": [
            "inadequate_supervision",
            "planned_inappropriate_operations",
            "failure_to_correct_known_problems",
            "supervisory_violation",
        ],
        "preconditions_for_unsafe_acts_detailed": [
            "physical_environment_factors",
            "tools_and_technology_issues",
            "operational_process_failures",
            "communication_coordination_planning_failures",
            "fit_for_duty",
            "mental_problems",
            "physiological_state",
            "physical_mental_limitations",
        ],
        "unsafe_acts_detailed": [
            "decision_error",
            "skill_based_errors",
            "perceptual_error",
            "routine_violation",
            "exceptional_violation",
        ],
    }

    output_rows = []

    for index, row in df.iterrows():
        detailed_values = [
            re.sub(r"^\d+\. \s*", "", output).replace(" ", "")
            for output in row["result"].split("\n")
        ]
        for factor, value in zip(factors[row["prompt"]], detailed_values):
            output_rows.append({"prompt": factor, "result": value})
    df = pd.DataFrame(output_rows)

    df["result"] = df["result"].replace({"YES": 1, "NO": 0})
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.groupby(["prompt"]).sum()

    # Load manual data
    manual_df = pd.read_csv("data/raw/manual.csv")

    final_df = pd.merge(df, manual_df, on="prompt")
    print(final_df)


with skip_run("skip", "consolidate_data_io_merged") as check, check():
    df = pd.read_csv("data/raw/io_expanded_merged_results.csv")[["prompt", "result"]]

    factors = [
        "inadequate_supervision",
        "planned_inappropriate_operations",
        "failure_to_correct_known_problems",
        "supervisory_violation",
        "physical_environment_factors",
        "tools_and_technology_issues",
        # "operational_process_failures",
        "communication_coordination_planning_failures",
        "fit_for_duty",
        "mental_problems",
        "physiological_state",
        "physical_mental_limitations",
        "decision_error",
        "skill_based_errors",
        "perceptual_error",
        "routine_violation",
        "exceptional_violation",
    ]

    output_rows = []

    for index, row in df.iterrows():
        detailed_values = [
            re.sub(r"^\d+\. \s*", "", output).replace(" ", "")
            for output in row["result"].split("\n")
        ]
        for factor, value in zip(factors, detailed_values):
            output_rows.append({"prompt": factor, "result": value})

    df = pd.DataFrame(output_rows)

    df["result"] = df["result"].replace({"YES": 1, "NO": 0})
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.groupby(["prompt"]).sum().astype(int)

    # Load manual data
    manual_df = pd.read_csv("data/raw/manual.csv")

    final_df = pd.merge(df, manual_df, on="prompt")
    print(final_df)


with skip_run("skip", "consolidate_data_with_cot") as check, check():
    data = pd.read_csv("data/cot_explanation_llm_manual.csv")

    print(data)


with skip_run("skip", "chi_sqaure_test") as check, check():
    data = pd.read_csv("data/no_explanation_llm_manual.csv")

    from scipy.stats import chisquare

    print(data["Manual"].sum())
    print(data["LLM"].sum())

    chi2_statistic, p_value = chisquare(f_exp=data["Manual"], f_obs=data["LLM"])

    print("Chi-squared statistic:", chi2_statistic)
    print("P-value:", p_value)
