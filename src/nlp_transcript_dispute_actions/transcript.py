"""转写版本、发言和原文区间。

``Passage`` 始终指向某一版转写中的发言编号和该发言原文的字符范围
（含左端、不含右端）。跨版本定位由 :mod:`revision` 完成。
"""

from dataclasses import dataclass

from .text import normalized_stream


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


@dataclass(frozen=True)
class Segment:
    """交付材料中的一段原文，带说话人、字符位置和该发言的音频区间。"""

    turn: str
    speaker: str
    start: int
    end: int
    text: str
    start_ms: int
    end_ms: int

    def to_dict(self):
        return {
            'turn': self.turn, 'speaker': self.speaker,
            'start': self.start, 'end': self.end, 'text': self.text,
            'audio_ms': [self.start_ms, self.end_ms],
        }


class Transcript:
    """保留一次转写中的说话人、音频区间和原文位置。"""

    def __init__(self, call, version, turns):
        self.call = call
        self.version = version
        turns = list(turns)
        self.turns = {}
        for turn in turns:
            if turn.id in self.turns or not 0 <= turn.start_ms < turn.end_ms:
                raise ValueError(f'发言编号或录音区间有误：{turn.id}')
            self.turns[turn.id] = turn
        if not self.turns:
            raise ValueError('转写至少需要一轮发言')
        ordered = [turn.id for turn in turns]
        if len(set(ordered)) != len(ordered):
            raise ValueError('发言编号不能重复')
        self.order = tuple(ordered)

    @classmethod
    def from_dict(cls, value):
        return cls(value['call'], value['version'], [Turn(**turn) for turn in value['turns']])

    def locate(self, passage):
        try:
            turn = self.turns[passage.turn]
        except KeyError:
            raise ValueError(f'未知发言：{passage.turn}') from None
        if not 0 <= passage.start < passage.end <= len(turn.text):
            raise ValueError('摘录范围超出发言')
        return turn, turn.text[passage.start:passage.end]

    def speaker(self, turn_id):
        return self.turns[turn_id].speaker

    def audio(self, passage):
        turn = self.turns[passage.turn]
        return [turn.start_ms, turn.end_ms]

    def segment(self, passage):
        turn, text = self.locate(passage)
        return Segment(turn.id, turn.speaker, passage.start, passage.end, text,
                       turn.start_ms, turn.end_ms)

    def segments(self, passages):
        """按原文顺序合并区间，同一发言内相邻或重叠的区间会合并。"""
        ordered = sorted(passages, key=lambda item: (self.order.index(item.turn), item.start, item.end))
        merged = []
        for passage in ordered:
            self.locate(passage)
            if merged and merged[-1].turn == passage.turn and passage.start <= merged[-1].end:
                last = merged[-1]
                merged[-1] = Passage(last.turn, last.start, max(last.end, passage.end))
            else:
                merged.append(passage)
        return [self.segment(passage) for passage in merged]

    def neighbors(self, turn_id):
        index = self.order.index(turn_id)
        before = self.order[index - 1] if index > 0 else None
        after = self.order[index + 1] if index + 1 < len(self.order) else None
        return before, after

    def stream(self):
        """整篇规范化字符流及每个字符的 (发言编号, 原文字符下标)。

        标点、空白和口语填充词被移除。跨发言拼接使校订后拆到多轮的同一段
        发言仍能被找到。
        """
        pieces = []
        mapping = []
        for turn_id in self.order:
            turn = self.turns[turn_id]
            normalized, positions = normalized_stream(turn.text)
            pieces.append(normalized)
            mapping.extend((turn_id, position) for position in positions)
        return ''.join(pieces), mapping

    def to_dict(self):
        return {
            'call': self.call, 'version': self.version,
            'turns': [
                {'id': turn.id, 'speaker': turn.speaker, 'start_ms': turn.start_ms,
                 'end_ms': turn.end_ms, 'text': turn.text}
                for turn in (self.turns[turn_id] for turn_id in self.order)
            ],
        }


def supports(finding):
    """认定的完整支撑区间：摘录本身加上判读必需的片段。"""
    passages = [finding.passage, *finding.essential]
    unique = []
    for passage in passages:
        if passage not in unique:
            unique.append(passage)
    return tuple(unique)
