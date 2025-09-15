from enum import Enum, auto

class Action(Enum):
    NEXT = auto()   # Move to next reel (scroll down)
    PREV = auto()   # Move to previous reel (scroll up)
    PAUSE = auto()  # (Future) pause / play toggle

__all__ = ["Action"]
