from io import BytesIO
import logging
import os
import copy
import heapq
import shutil
from typing import Any, Optional

from lavague.core.action_engine import ActionEngine
from lavague.core.world_model import WorldModel
from lavague.core.value_function import ValueFunction

from lavague.core.utilities.format_utils import (
    extract_before_next_engine,
    extract_next_engine,
    extract_world_model_instruction,
    replace_hyphens,
)
from lavague.core.logger import AgentLogger, LocalDBLogger
from lavague.core.action_memory import Action
from lavague.core.base_driver import BaseDriver
from lavague.core.base_engine import ActionResult
from lavague.core.utilities.telemetry import send_telemetry
from PIL import Image
from IPython.display import display, HTML, Code
from lavague.core.token_counter import TokenCounter
from lavague.core.utilities.config import is_flag_true

from lavague.core.utilities.profiling import (
    ChartGenerator,
    time_profiler,
    start_new_step,
    clear_profiling_data,
)

logging_print = logging.getLogger(__name__)
logging_print.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logging_print.addHandler(ch)
logging_print.propagate = False


class TreeAgent:
    """
    Web agent class, for now only works with selenium.
    """

    def __init__(
        self,
        world_model: WorldModel,
        action_engine: ActionEngine,
        value_function: ValueFunction,
        token_counter: Optional[TokenCounter] = None,
        n_steps: int = 10,
        branching_factor: int = 3,
        clean_screenshot_folder: bool = True,
        logger: AgentLogger = None,
    ):
        self.driver: BaseDriver = action_engine.driver
        self.action_engine: ActionEngine = action_engine
        self.world_model: WorldModel = world_model
        self.value_function: ValueFunction = value_function
        self.token_counter = token_counter
        self.interrupted = False

        self.n_steps = n_steps
        self.branching_factor = branching_factor

        self.output = ""

        self.clean_screenshot_folder = clean_screenshot_folder

        if logger is None:
            self.logger: AgentLogger = AgentLogger()
        else:
            self.logger = logger

        self.action_engine.set_logger_all(self.logger)
        self.world_model.set_logger(self.logger)
        self.value_function.set_logger(self.logger)
        
        init_action = Action()
        init_action.set_logger(self.logger)
        self.action_queue = []
        heapq.heappush(self.action_queue, init_action)
        self.url = ""

        if self.clean_screenshot_folder:
            try:
                if os.path.isdir("screenshots"):
                    shutil.rmtree("screenshots")
                logging_print.info("Screenshot folder cleared")
            except:
                pass

        self.result = ActionResult(
            instruction=None,
            code=self.driver.code_for_init(),
            success=False,
            output=None,
            total_estimated_tokens=0,
            total_estimated_cost=0.0,
        )

    def get(self, url):
        self.url = url
        self.driver.get(url)
        self.driver.wait_for_idle()
        self.result.code += self.driver.code_for_get(url) + "\n"


    def add_next_actions(self, objective: str, action: Action) -> Optional[ActionResult]:
        obs = self.driver.get_obs()
        current_state, past = action.get_state()

        for _ in range(self.branching_factor):
            world_model_output = self.world_model.get_instruction(
                objective, current_state, past, obs
            )
            logging_print.info(world_model_output)
            next_engine_name = extract_next_engine(world_model_output)
            instruction = extract_world_model_instruction(world_model_output)
            new_action = copy.deepcopy(action)
            new_action.update_last_engine(next_engine_name)
            new_action.update_instruction(instruction)
            if new_action.get_last_engine() == "COMPLETE" or new_action.get_last_engine() == "SUCCESS":
                return self.termination(new_action)
            heapq.heappush(self.action_queue, new_action)

    def termination(self, action: Action) -> ActionResult:
        self.result.success = True
        self.result.output = action.get_last_instruction()
        self.result.code += action.get_code()
        logging_print.info("Objective reached. Stopping...")
        return self.result
    
    def take_action_and_score(self, objective: str, action: Action) -> Optional[Action]:            
        next_engine_name = action.get_last_engine()
        instruction = action.get_last_instruction()
        action_result = self.action_engine.dispatch_instruction(
            next_engine_name, instruction
        )
        if action_result.success:
            action.update_current_state(action_result.output)
            action.update_code(action_result.code)

            obs = self.driver.get_obs()
            current_state, past = action.get_state()
            score = self.value_function.get_score(objective, current_state, past, obs)
            action.update_value(score)

            return action
        
        return None

    def run_step(self, objective: str) -> Optional[ActionResult]:
        # Select best action to do:
        best_action = heapq.heappop(self.action_queue)

        # Navigate to be in the right state to perform the action
        self.driver.get(self.url)
        if best_action.get_code() != "":
            self.driver.exec_code(best_action.get_code()) 

        # Take action and score
        best_action = self.take_action_and_score(objective, best_action)
        if best_action == None:
            return None

        # Add next actions to the queue
        obs = self.driver.get_obs()
        result = self.add_next_actions(objective, best_action)
        
        self.logger.add_log(obs)

        self.process_token_usage()
        self.logger.end_step()
        return result

    def prepare_run(self, display: bool = False, user_data=None):
        self.action_engine.set_display_all(display)
        if user_data:
            init_action = heapq.heappop(self.action_queue)
            init_action.set_user_data(user_data)
            heapq.heappush(self.action_queue, init_action)
        self.logger.new_run()

    def run(
        self,
        objective: str,
        user_data=None,
        display: bool = False,
        log_to_db: bool = is_flag_true("LAVAGUE_LOG_TO_DB"),
        step_by_step=False,
    ) -> ActionResult:
        self.interrupted = False
        self.prepare_run(display=display, user_data=user_data)
        # Add next actions to the root queue
        init_action = heapq.heappop(self.action_queue)
        self.add_next_actions(objective, init_action)
        try:
            for _ in range(self.n_steps):
                start_new_step()
                with time_profiler("Run step", full_step_profiling=True):
                    result = self.run_step(objective)

                if result is not None:
                    break

                if step_by_step:
                    input("Press ENTER to continue")

        except KeyboardInterrupt:
            logging_print.warning("The agent was interrupted.")
            self.interrupted = True
            pass
        except Exception as e:
            logging_print.error(f"Error while running the agent: {e}")
            self.interrupted = True
            raise e
        finally:
            origin = self.origin if hasattr(self, "origin") else "lavague"
            send_telemetry(self.logger.return_pandas(), origin=origin)
            if log_to_db:
                local_db_logger = LocalDBLogger()
                local_db_logger.insert_logs(self)
        return self.result

    def process_token_usage(self):
        if self.token_counter is not None:
            token_counts, token_costs = self.token_counter.process_token_usage(
                self.world_model, self.action_engine, result_to_update=self.result
            ) # self.value_function could be added
            self.logger.add_log(token_counts)
            self.logger.add_log(token_costs)

    def display_previous_nodes(self, steps: int) -> None:
        """prints out all nodes per each sub-instruction for given steps"""
        dflogs = self.logger.return_pandas()
        # check if dflogs are not null and not empty and engine_log is present in dflogs columns
        if (
            dflogs is not None
            and dflogs.empty is False
            and "engine_log" in dflogs.columns
        ):
            if steps > len(dflogs):
                print(
                    f"Previous steps: {len(dflogs)}\nrequested steps: {steps}\nshowing available steps"
                )
            steps = len(dflogs) if steps > len(dflogs) else steps
            for step in range(steps):
                print(f"Step: {step}")
                sub_ins = 0
                if isinstance(dflogs.at[step, "engine_log"], list):
                    for subinst in dflogs.at[step, "engine_log"]:
                        print(f"Sub-Instruction: {sub_ins}")
                        sub_ins += 1
                        x = 0
                        for node in subinst["retrieved_html"]:
                            print(f"Node {x}")
                            x = x + 1
                            display(HTML(node))  # Display node as visual element
                            display(Code(node, language="html"))  # Display code
        else:
            print(
                f"No previous nodes available. Please run the agent atleast once to view previous steps"
            )

    def display_all_nodes(self) -> None:
        """prints out all nodes per each sub-instruction"""
        dflogs = self.logger.return_pandas()
        # check if dflogs are not null and not empty and engine_log is present in dflogs columns
        if (
            dflogs is not None
            and dflogs.empty is False
            and "engine_log" in dflogs.columns
        ):
            print(f"Number of steps: {len(dflogs)}")
            steps = len(dflogs)
            for step in range(steps):
                print(f"Step: {step}")
                sub_ins = 0
                if isinstance(dflogs.at[step, "engine_log"], list):
                    for subinst in dflogs.at[step, "engine_log"]:
                        print(f"Sub-Instruction: {sub_ins}")
                        sub_ins += 1
                        x = 0
                        for node in subinst["retrieved_html"]:
                            print(f"Node: {x}")
                            x = x + 1
                            display(HTML(node))  # Display node as visual element
                            display(Code(node, language="html"))  # Display code
        else:
            print(
                f"No previous nodes available. Please run the agent atleast once to view previous steps"
            )

    def set_origin(self, origin: str):
        self.origin = origin

    def get_summary(self):
        from lavague.core.utilities.profiling import agent_events, agent_steps

        chart_generator = ChartGenerator(
            agent_events=agent_events, agent_steps=agent_steps
        )
        plot = chart_generator.plot_waterfall()
        table = chart_generator.get_summary_df()

        clear_profiling_data()

        return plot, table