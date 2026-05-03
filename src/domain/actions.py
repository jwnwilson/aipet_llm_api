from enum import Enum


class Action(str, Enum):
    EAT = "EAT"
    DRINK = "DRINK"
    PLAY = "PLAY"
    FETCH = "FETCH"
    SLEEP = "SLEEP"
    SOCIAL = "SOCIAL"
    FOLLOW = "FOLLOW"
    TOILET = "TOILET"
    IDLE = "IDLE"
    EXPLORE = "EXPLORE"
