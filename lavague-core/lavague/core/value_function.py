from __future__ import annotations
import os
from abc import ABC
from llama_index.core import PromptTemplate
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.legacy.readers.file.base import SimpleDirectoryReader
from lavague.core.context import Context, get_default_context
from lavague.core.logger import AgentLogger, Loggable
from functools import lru_cache
from PIL import Image
from lavague.core.utilities.model_utils import get_model_name
import time
import yaml
from lavague.core.utilities.profiling import time_profiler

import re
import numpy as np

VALUE_FUNCTION_PROMPT_TEMPLATE = PromptTemplate(
    """
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's objective, the previous instructions the agent was given, the current state of the webpage, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

Your inputs are:
- objective ('str'): a high level description of the goal to achieve.
- previous_instructions ('str'): a list of previous steps taken to reach the objective.
- last_engine ('str'): the engine used in the previous step.
- current_state ('dict'): the state of the environment in YAML to use to perform the next step.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"

Here is the next objective:
Objective: {objective}
Previous instructions:
{previous_instructions}
Last engine: {last_engine}
Current state:
{current_state}
{tab_info}

Thought:
"""
)


def clean_directory(path):
    # Get all the file names in the directory
    file_names = os.listdir(path)

    # Iterate over the file names and remove each file
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


class ValueFunction(ABC, Loggable):
    """Abstract class for ValueFunction"""

    def __init__(
        self,
        mm_llm: MultiModalLLM = None,
        prompt_template: PromptTemplate = VALUE_FUNCTION_PROMPT_TEMPLATE,
        logger: AgentLogger = None,
        n: int = 20,
    ):
        if mm_llm is None:
            mm_llm = get_default_context().mm_llm
        self.mm_llm: MultiModalLLM = mm_llm
        self.prompt_template = prompt_template
        self.logger: AgentLogger = logger
        self.n = n

    def get_score(
        self,
        objective: str,
        current_state: dict,
        past: dict,
        observations: dict,
    ) -> str:
        """Use GPT*V to generate a score for the current state and objective."""
        mm_llm = self.mm_llm
        logger = self.logger

        previous_instructions = past["previous_instructions"]
        last_engine = past["last_engine"]

        tab_info = observations["tab_info"]

        try:
            current_state_str = yaml.dump(current_state, default_flow_style=False)
        except:
            raise Exception("Could not convert current state to YAML")

        screenshots_path: str = observations["screenshots_path"]
        image_documents = SimpleDirectoryReader(screenshots_path).load_data()

        prompt = self.prompt_template.format(
            objective=objective,
            previous_instructions=previous_instructions,
            last_engine=last_engine,
            current_state=current_state_str,
            tab_info=tab_info,
        )

        start = time.time()

        with time_profiler("Value Function Inference", prompt_size=len(prompt)):
            all_responses = []
            for _ in range(self.n):
                mm_llm_output = mm_llm.complete(
                    prompt, image_documents=image_documents
                ).text
                all_responses.append(mm_llm_output)

        end = time.time()
        value_function_inference_time = end - start

        all_scores = []
        for r in all_responses:
            try:
                pred = re.search(r'Status: "?(.+)"?', r).group(1)
                if 'success' in pred.lower():
                    score = 1.0
                else:
                    # Check if it's on the path to success
                    on_path = re.search(r'On the right track to success: "?(.+)"?', r).group(1)
                    if 'yes' in on_path.lower():
                        score = 0.5
                    else:
                        score = 0.0
            except Exception as e:
                print(f"Error parsing response: {e}")
                score = 0.0
            
            all_scores.append(score)
        
        score = np.mean(all_scores).item()

        if logger:
            log = {
                "value_function_prompt": prompt,
                "Final score": score,
                "value_function_inference_time": value_function_inference_time,
                "screenshots": [
                    Image.open(image_document.image_path)
                    for image_document in image_documents
                ],
            }
            logger.add_log(log)

        return score

    def get_mm_llm_name(self):
        return get_model_name(self.mm_llm)