"""同一通话的人工校订版与旧摘录之间的对照。

定位只看字面内容和忽略的标点、断句、语气词；认定是否改变则由
:func:`text.semantic_differences` 判断。这样“把一句话拆成两轮”、“修正
交叠时段的说话人”或“只改标点”的情况都能被区分开。
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .text import normalized_stream, semantic_differences
from .transcript import Passage


# 字面高度相似但并非同一句时，不强行对应。
FUZZY_THRESHOLD = 0.75

NEEDS_REVIEW = ('多个对应', '无法对应', '已改变认定')


@dataclass(frozen=True)
class AlignedTurn:
    turn: str
    speaker: str
    start: int
    end: int
    text: str
    start_ms: int
    end_ms: int

    def to_dict(self):
        return {
            'turn': self.turn, 'speaker': self.speaker, 'start': self.start,
            'end': self.end, 'text': self.text,
            'audio_ms': [self.start_ms, self.end_ms],
        }


@dataclass(frozen=True)
class Alignment:
    passage: Passage
    status: str
    old_text: str
    old_speaker: str
    old_audio_ms: tuple[int, int]
    matches: tuple[tuple[AlignedTurn, ...], ...] = field(default_factory=tuple)
    differences: tuple[str, ...] = ()
    reason: str = ''

    @property
    def reusable(self):
        return self.status == '同一段发言'

    def to_dict(self):
        return {
            'passage': {'turn': self.passage.turn, 'start': self.passage.start,
                        'end': self.passage.end},
            'status': self.status,
            'old_text': self.old_text,
            'old_speaker': self.old_speaker,
            'old_audio_ms': list(self.old_audio_ms),
            'matches': [[segment.to_dict() for segment in match] for match in self.matches],
            'differences': list(self.differences),
            'reason': self.reason,
            'reusable': self.reusable,
        }


class Revision:
    def __init__(self, previous, current):
        if previous.call != current.call:
            raise ValueError('只能对照同一通话的转写版本')
        if previous.version == current.version:
            raise ValueError('对照需要两个不同的转写版本')
        self.previous = previous
        self.current = current
        self.stream, self.mapping = current.stream()

    def _segments(self, start, end):
        """把字符流区间还原为按发言分组的原文区间。"""
        groups = []
        for turn_id, position in self.mapping[start:end]:
            if groups and groups[-1][0] == turn_id and position >= groups[-1][1]:
                group = groups[-1]
                groups[-1] = (turn_id, group[1], max(group[2], position + 1))
            else:
                groups.append((turn_id, position, position + 1))
        segments = []
        for turn_id, left, right in groups:
            turn = self.current.turns[turn_id]
            segments.append(AlignedTurn(
                turn_id, turn.speaker, left, right, turn.text[left:right],
                turn.start_ms, turn.end_ms,
            ))
        return tuple(segments)

    def _occurrences(self, needle):
        starts = []
        index = self.stream.find(needle)
        while index >= 0:
            starts.append((index, index + len(needle)))
            index = self.stream.find(needle, index + 1)
        return starts

    def _fuzzy(self, needle):
        """措辞有改动时，找出新版中整段对应但字面不同的发言。

        只在整轮发言范围内比较，相似度够高才作为对应；差异交由后续语义
        对比判断，因此“还没结算”改成“已经结算”会报出认定改变，而不会
        被当成找不到。
        """
        if len(needle) < 4:
            return []
        groups = []
        for index, (turn_id, _) in enumerate(self.mapping):
            if groups and groups[-1][0] == turn_id:
                groups[-1][2] = index + 1
            else:
                groups.append([turn_id, index, index + 1])
        candidates = []
        for turn_id, left, right in groups:
            similarity = SequenceMatcher(
                None, needle, self.stream[left:right], autojunk=False
            ).ratio()
            if similarity >= FUZZY_THRESHOLD:
                candidates.append((left, right))
        return candidates

    def align(self, passage):
        turn, old_text = self.previous.locate(passage)
        needle, _ = normalized_stream(old_text)
        if not needle:
            return Alignment(passage, '无法对应', old_text, turn.speaker,
                             (turn.start_ms, turn.end_ms), reason='摘录只有标点或空白')
        starts = self._occurrences(needle)
        if not starts:
            starts = self._fuzzy(needle)
        if not starts:
            return Alignment(
                passage, '无法对应', old_text, turn.speaker,
                (turn.start_ms, turn.end_ms),
                reason='新版转写中找不到对应文字，需审核员按录音判断',
            )
        matches = tuple(
            self._segments(start, end) for start, end in starts
        )
        if len(matches) > 1:
            return Alignment(
                passage, '多个对应', old_text, turn.speaker,
                (turn.start_ms, turn.end_ms), matches=matches,
                reason='新版有多处文字相同，需审核员按录音选择',
            )

        segments = matches[0]
        new_text = ''.join(segment.text for segment in segments)
        differences = list(semantic_differences(old_text, new_text))
        speakers = {segment.speaker for segment in segments}
        if speakers != {turn.speaker}:
            differences.insert(0, '说话者变更')
        if differences:
            return Alignment(
                passage, '已改变认定', old_text, turn.speaker,
                (turn.start_ms, turn.end_ms), matches=matches,
                differences=tuple(differences),
                reason='定位到同一段发言，但认定所依据的内容已变化，不能沿用旧结论',
            )
        return Alignment(
            passage, '同一段发言', old_text, turn.speaker,
            (turn.start_ms, turn.end_ms), matches=matches,
            reason='标点、断句或措辞变化不影响认定',
        )

    def align_finding(self, finding):
        """把一条认定的全部依据放到新版中核对。

        逐段对照所有依据；任何一段改变说话者、退款对象或限制条件，整条
        结论都不能继续作为本次争议的依据。
        """
        aligned = [self.align(passage) for passage in finding.all_passages()]
        blocked = [item for item in aligned if item.status in NEEDS_REVIEW]
        if blocked:
            status = '已改变认定' if any(
                item.status == '已改变认定' for item in blocked
            ) else ('无法对应' if any(
                item.status == '无法对应' for item in blocked
            ) else '多个对应')
            reusable = False
            reason = '；'.join(dict.fromkeys(item.reason for item in blocked))
        else:
            status = '同一段发言'
            reusable = True
            reason = '新版可定位到同一段发言，且认定依据未变'
        return {
            'finding': finding.id, 'verdict': finding.verdict,
            'status': status, 'reusable': reusable, 'reason': reason,
            'segments': [item.to_dict() for item in aligned],
            'disagreements': [
                {
                    'passage': {'turn': item.passage.turn, 'start': item.passage.start,
                                'end': item.passage.end},
                    'old_text': item.old_text,
                    'old_speaker': item.old_speaker,
                    'old_audio_ms': list(item.old_audio_ms),
                    'status': item.status,
                    'differences': list(item.differences),
                    'reason': item.reason,
                    'candidates': [[segment.to_dict() for segment in match]
                                   for match in item.matches],
                }
                for item in blocked
            ],
        }
