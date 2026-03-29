"""Mission planning state and junction decision logic.

Tracks visited AprilTag nodes and resolves branch choices based on
a deterministic exploration policy.
"""

from enum import Enum

class TurnDecision(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"

class MissionPlanner:
    def __init__(self, target_countries: list[int]):
        self.targets = [c for c in target_countries if c != 0]
        self.visited_tags = set()
        
        # In a completely unknown graph without odometry, the Left-Hand Rule 
        # (always turning left at a junction) mathematically guarantees you 
        # will explore every connected edge of the maze.
        self.default_routing_rule = TurnDecision.LEFT

    def on_tag_reached(self, tag_id: int, country_code: int, is_landable: bool) -> bool:
        """Mark tag as visited and evaluate landing condition.

        Args:
            tag_id: Unique AprilTag identifier.
            country_code: First digit code from tag payload.
            is_landable: True when airport is declared safe.

        Returns:
            True if this node is still a pending mission target and can land.
        """
        self.visited_tags.add(tag_id)

        return country_code in self.targets and is_landable

    def on_target_landed(self, country_code: int):
        """Removes the country from the target list after a successful landing."""
        if country_code in self.targets:
            self.targets.remove(country_code)

    def is_mission_complete(self) -> bool:
        return len(self.targets) == 0

    def get_junction_decision(self, branch_count: int) -> str:
        """Determine trajectory bias at detected junction.

        Args:
            branch_count: number of active line branches in current view.

        Returns:
            Direction string from policy: left/right/straight.
        """
        if branch_count > 1:
            return self.default_routing_rule.value
        return TurnDecision.STRAIGHT.value