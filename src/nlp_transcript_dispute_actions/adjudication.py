from dataclasses import dataclass

from .transcript import Passage


@dataclass(frozen=True)
class Finding:
    id: str
    verdict: str
    reviewer: str
    passage: Passage
    essential: tuple[Passage, ...]


class DisputeReview:
    def __init__(self, staff_roles):
        self.staff_roles = staff_roles

    def accept(self, transcript, finding):
        if self.staff_roles.get(finding.reviewer) != '内审':
            raise ValueError('此人没有认定权限')
        if finding.verdict not in {'承诺', '仅查询', '无法认定'}:
            raise ValueError('未知的承诺认定')
        turn, text = transcript.locate(finding.passage)
        if finding.verdict == '承诺' and turn.speaker != '坐席':
            raise ValueError('不能把客户发言直接认定为坐席承诺')
        for required in finding.essential:
            transcript.locate(required)
            if (required.turn != turn.id or required.start < finding.passage.start
                    or required.end > finding.passage.end):
                raise ValueError('摘录缺少判读所需内容')
        return turn, text
