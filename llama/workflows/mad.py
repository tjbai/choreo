from typing import Tuple

from llama import Llama, Workflow

moderator_system_prompt = '''
You are a moderator in a debate between
'''

def mad_cached(
    workflow: Workflow,
    agents: Tuple[str, str], # (prompt, name)
    max_rounds: int,
):
    pass

def mad_baseline(
    llama: Llama,
    agents: Tuple[str, str],
    max_rounds: int,
):
    pass
