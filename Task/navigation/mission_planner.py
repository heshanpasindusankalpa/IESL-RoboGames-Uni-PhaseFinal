"""
navigation/mission_planner.py

Handles graph traversal logic, tracking visited airports (nodes),
and making routing decisions at junctions (edges).
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
        """
        Registers a visited node in the graph.
        Returns True if the drone should land here.
        """
        self.visited_tags.add(tag_id)
        
        if country_code in self.targets and is_landable:
            return True
        return False

    def on_target_landed(self, country_code: int):
        """Removes the country from the target list after a successful landing."""
        if country_code in self.targets:
            self.targets.remove(country_code)

    def is_mission_complete(self) -> bool:
        return len(self.targets) == 0

    def get_junction_decision(self, branch_count: int) -> str:
        """
        Called when the vision system detects a fork in the road.
        Returns the routing bias ("left", "right", or "straight").
        """
        if branch_count > 1:
            # We hit a junction. Apply the maze solving rule to pick a path.
            return self.default_routing_rule.value
        return TurnDecision.STRAIGHT.value