from enum import Enum


class RetrievalTechniques(Enum):
    ReverseLogitLens = 1
    LogitLens = 2
    Patchscopes = 3


class MultiTokenKind(Enum):
    Split = 1
    Typo = 2
    Natural = 3
