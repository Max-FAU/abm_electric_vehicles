from mesa.time import SimultaneousActivation
# mypy
from typing import Iterator, Union
import mesa


TimeT = Union[float, int]


class AgentDualStep(mesa.Agent):
    def pre_step(self) -> None:
        """A single step of the agent."""
        pass


class DualStepScheduler(SimultaneousActivation):
    def __init__(self):
        self.pre_step = 0
        self.pre_time: TimeT = 0

    def pre_step(self) -> None:
        """ Calculate potential attributes before taking the final step."""

        # To be able to remove and/or add agents during stepping
        # it's necessary to cast the keys view to a list.
        agent_keys = list(self._agents.keys())
        for agent_key in agent_keys:
            self._agents[agent_key].pre_step()
        # We recompute the keys because some agents might have been removed in
        # the previous loop.
        agent_keys = list(self._agents.keys())
        for agent_key in agent_keys:
            self._agents[agent_key].advance()
        self.pre_step += 1
        self.time += 1
