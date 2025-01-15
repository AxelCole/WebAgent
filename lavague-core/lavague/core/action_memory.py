"""TODO: Make this class generalizable"""

from lavague.core.logger import AgentLogger, Loggable


class Action(Loggable):
    """
    Short term memory module of the agent.
    """

    def __init__(self, user_data=None, logger: AgentLogger = None) -> None:
        current_state = {
            "external_observations": {
                "vision": "[SCREENSHOT]",
            },
            "internal_state": {
                "user_inputs": [],
                "agent_outputs": [],
            },
        }
        self.logger = logger

        if user_data:
            current_state["internal_state"]["user_inputs"].append(user_data)

        self.current_state = current_state

        self.previous_instructions: str = "[NONE]"
        self.last_instruction: str = ""
        self.last_engine: str = "[NONE]"
        self.code_trajectory: str = ""
        self.value: int = 0.5
    
    def __lt__(self, other):
        return self.value < other.value

    def set_user_data(self, user_data=None):
        if user_data:
            self.current_state["internal_state"]["user_inputs"].append(user_data)

    def get_state(self):
        current_state = self.current_state
        past = {
            "previous_instructions": str(self.previous_instructions),
            "last_engine": self.last_engine,
        }

        logger = self.logger

        if logger:
            log = {
                "current_state": current_state,
                "past": past,
            }
            logger.add_log(log)
        return current_state, past
    
    def get_last_instruction(self):
        return self.last_instruction
    def get_last_engine(self):
        return self.last_engine
    def get_code(self):
        return self.code_trajectory
    def get_value(self):
        return self.value


    def update_instruction(self, instruction):
        if self.previous_instructions == "[NONE]":
            self.previous_instructions = f"""
- {instruction}"""
        else:
            self.previous_instructions += f"""
- {instruction}"""
        self.last_instruction = instruction
    def update_current_state(self, output):
        if output:
            self.current_state["internal_state"]["agent_outputs"].append(output)
    def update_last_engine(self, last_engine):
        self.last_engine = last_engine      
    def update_code(self, last_code):
        self.code_trajectory += last_code
    def update_value(self, value):
        self.value = value
      


    