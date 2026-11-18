from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    id: str
    speaker: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Passage:
    turn: str
    start: int
    end: int


class Transcript:
    """保留一次转写中的说话人、音频区间和原文位置。"""

    def __init__(self, call, version, turns):
        self.call = call
        self.version = version
        self.turns = {}
        for turn in turns:
            if turn.id in self.turns or not 0 <= turn.start_ms < turn.end_ms:
                raise ValueError(f'发言编号或录音区间有误：{turn.id}')
            self.turns[turn.id] = turn

    def locate(self, passage):
        turn = self.turns[passage.turn]
        if not 0 <= passage.start < passage.end <= len(turn.text):
            raise ValueError('摘录范围超出发言')
        return turn, turn.text[passage.start:passage.end]
