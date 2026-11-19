"""争议事项候选摘录的发现，以及认定的复核。"""

from dataclasses import dataclass, field

from .text import classify_text, clause_spans, cues, verdict_for_category
from .transcript import Passage


VERDICTS = ('承诺', '附条件答复', '仅查询', '无法认定')
PROMISE_VERDICTS = ('承诺', '附条件答复')

# 默认争议事项关键词：退款是否得到承诺。
DEFAULT_MATTER_KEYWORDS = ('退款', '退', '到账', '订单', '结算', '承诺', '查询')

_CUE_REASONS = (
    ('negated_promise', '含否定表述，承诺被否定'),
    ('conditional', '含条件或限制表述'),
    ('retraction', '含改口或更正表述'),
    ('relay', '含转述他人说法'),
    ('interruption', '含插话或打断表述'),
    ('other_subject', '指向另一笔退款或另一订单'),
    ('same_subject', '明确指向本笔退款'),
    ('query', '含查询或跟进动作'),
    ('promise', '含退款、到账或答应类表述'),
)


@dataclass(frozen=True)
class Candidate:
    id: str
    turn: str
    speaker: str
    start: int
    end: int
    text: str
    audio_ms: tuple[int, int]
    category: str
    verdict: str
    reasons: tuple[str, ...]
    context: tuple[dict, ...] = ()
    suggested_support: tuple[Passage, ...] = ()
    requires_context: bool = False
    matching_matter: bool = True

    def to_dict(self):
        return {
            'id': self.id, 'turn': self.turn, 'speaker': self.speaker,
            'start': self.start, 'end': self.end, 'text': self.text,
            'audio_ms': list(self.audio_ms), 'category': self.category,
            'verdict': self.verdict, 'reasons': list(self.reasons),
            'context': [dict(item) for item in self.context],
            'suggested_support': [
                {'turn': item.turn, 'start': item.start, 'end': item.end}
                for item in self.suggested_support
            ],
            'requires_context': self.requires_context,
            'matching_matter': self.matching_matter,
        }


def _clauses(transcript, turn_id):
    """把一轮发言按句读切成保留原文位置的区间。"""
    turn = transcript.turns[turn_id]
    return [Passage(turn.id, left, right) for left, right in clause_spans(turn.text)]


def _context_segment(transcript, passage, role):
    segment = transcript.segment(passage)
    return {
        'role': role, 'turn': segment.turn, 'speaker': segment.speaker,
        'start': segment.start, 'end': segment.end, 'text': segment.text,
        'audio_ms': [segment.start_ms, segment.end_ms],
    }


def find_candidates(transcript, keywords=DEFAULT_MATTER_KEYWORDS):
    """按争议事项关键词列出可核对的原文片段。

    每个候选带说话人、音频位置、识别类型和理由。条件、否定或改口如果落在
    本轮的后半句，或者隔着一轮客户插话落在坐席后一轮里，会通过
    ``suggested_support`` 和 ``requires_context`` 一并给出，避免专员只选到
    “可以退”而漏掉“前提是…”。
    """
    candidates = []
    order = transcript.order
    lowered_keywords = tuple(keyword.casefold() for keyword in keywords if keyword)

    for index, turn_id in enumerate(order):
        turn = transcript.turns[turn_id]
        clauses = _clauses(transcript, turn_id)
        for clause in clauses:
            text = turn.text[clause.start:clause.end]
            if not any(keyword in text.casefold() for keyword in lowered_keywords):
                continue
            value = cues(text)
            category = classify_text(text, turn.speaker)
            reasons = [reason for attribute, reason in _CUE_REASONS if getattr(value, attribute)]

            context = _neighbor_context(transcript, order, index)
            support = []
            needs_context = False

            if turn.speaker == '坐席' and value.positive_promise:
                support, needs_context = _qualifier_support(
                    transcript, order, index, clause, clauses
                )
                if needs_context:
                    reasons.append('条件、否定或改口不在本句，需一并摘录')
                    if not value.qualifier:
                        reasons.append('单看本句可能误认为无条件承诺')
            elif turn.speaker == '坐席' and value.qualifier:
                support, needs_context = _interrupted_promise(
                    transcript, order, index, clauses
                )
                if needs_context:
                    reasons.append('本句条件或否定隔着一轮客户插话，需与前面的答复一并摘录')

            candidates.append(Candidate(
                id=f'{turn_id}:{clause.start}-{clause.end}',
                turn=turn_id, speaker=turn.speaker,
                start=clause.start, end=clause.end, text=text,
                audio_ms=(turn.start_ms, turn.end_ms),
                category=category, verdict=verdict_for_category(category),
                reasons=tuple(dict.fromkeys(reasons)),
                context=context,
                suggested_support=tuple(support),
                requires_context=needs_context,
            ))
    return candidates


def _neighbor_context(transcript, order, index):
    """摘录前后各一轮的首尾句，供专员按上下文核对。"""
    context = []
    if index > 0:
        previous = _clauses(transcript, order[index - 1])
        if previous:
            context.append(_context_segment(transcript, previous[-1], 'preceding'))
    if index + 1 < len(order):
        following = _clauses(transcript, order[index + 1])
        if following:
            context.append(_context_segment(transcript, following[0], 'following'))
    return tuple(context)


def _qualifier_support(transcript, order, index, clause, clauses):
    """找出坐在答复之后、决定其含义的条件或否定。"""
    support = []
    # 同一轮的后半句：“可以退，前提是还没结算”。
    for later in clauses:
        if later.start >= clause.end and cues(
                transcript.turns[later.turn].text[later.start:later.end]).qualifier:
            support.append(later)
    # 隔着一轮客户插话的下一轮：“等一下……”“前提是还没结算”。
    if not support and index + 1 < len(order):
        next_turn = order[index + 1]
        if transcript.speaker(next_turn) != '坐席':
            if index + 2 < len(order) and transcript.speaker(order[index + 2]) == '坐席':
                resumed = _clauses(transcript, order[index + 2])
                for item in resumed:
                    text = transcript.turns[item.turn].text[item.start:item.end]
                    if cues(text).qualifier:
                        support.append(item)
    return support, bool(support)


def _interrupted_promise(transcript, order, index, clauses):
    """条件句前面是否隔着客户插话的坐席答复，需一并摘录。"""
    if index < 2 or transcript.speaker(order[index - 1]) == '坐席':
        return [], False
    earlier = _clauses(transcript, order[index - 2])
    if not earlier or transcript.speaker(order[index - 2]) != '坐席':
        return [], False
    promise = [
        item for item in earlier
        if cues(transcript.turns[item.turn].text[item.start:item.end]).positive_promise
    ]
    return promise, bool(promise)


@dataclass(frozen=True)
class Finding:
    id: str
    verdict: str
    reviewer: str
    passage: Passage
    essential: tuple[Passage, ...] = ()
    # 决定含义但隔着插话的片段，可以位于别的发言；会随摘录一并定位和发放。
    support: tuple[Passage, ...] = ()
    # 争议事项标识，用于把可办理项和等待复核项分开。
    matter: str | None = None
    # 需要保护的账户代号明文，供遮盖和泄漏检查使用。
    sensitive: tuple[str, ...] = ()
    # 随摘录一并外发的说明文字；同样不能拼回账户代号。
    notes: tuple[str, ...] = ()

    def all_passages(self):
        unique = [self.passage, *self.essential, *self.support]
        return tuple(dict.fromkeys(unique))


@dataclass(frozen=True)
class Assessment:
    verdict: str
    category: str
    excerpt: str
    combined: str
    reasons: tuple[str, ...]
    support_texts: tuple[str, ...] = field(default_factory=tuple)


class DisputeReview:
    def __init__(self, staff_roles):
        self.staff_roles = staff_roles

    def _authorize(self, finding):
        if self.staff_roles.get(finding.reviewer) != '内审':
            raise ValueError('此人没有认定权限')
        if finding.verdict not in VERDICTS:
            raise ValueError('未知的承诺认定')

    def assess(self, transcript, finding):
        self._authorize(finding)
        turn, text = transcript.locate(finding.passage)
        if finding.verdict in PROMISE_VERDICTS and turn.speaker != '坐席':
            raise ValueError('不能把客户发言直接认定为坐席承诺')
        for required in finding.essential:
            required_turn, _ = transcript.locate(required)
            if (required.turn != turn.id or required.start < finding.passage.start
                    or required.end > finding.passage.end):
                raise ValueError('摘录缺少判读所需内容')
        support_texts = []
        for extra in getattr(finding, 'support', ()):
            _, support_text = transcript.locate(extra)
            support_texts.append(support_text)

        # 客户插话里也可能带“另一笔订单”等指代；决定坐席答复含义的是坐席自己
        # 的话。取发言全文，使同一轮中位于摘录之外的条件或否定也能被发现。
        staff_parts = [turn.text]
        for extra in getattr(finding, 'support', ()):
            if transcript.speaker(extra.turn) == turn.speaker:
                staff_parts.append(transcript.locate(extra)[1])
        own_text = ''.join(staff_parts)
        combined = ''.join([text, *support_texts])

        relay_only = cues(own_text).relay and not _own_statement(own_text)
        if relay_only and finding.verdict in PROMISE_VERDICTS:
            raise ValueError('转述他人的说法不能直接认定为坐席本人承诺')
        if _other_subject(own_text):
            raise ValueError('摘录指向另一笔订单，不能作为本次争议的依据')
        category = classify_text(own_text, turn.speaker)
        # 承诺类认定的摘录本身必须含该承诺，不能只靠补充片段补足；能否算
        # 无条件承诺则要看坐席同一段话里有没有条件或否定。
        if finding.verdict in PROMISE_VERDICTS:
            if classify_text(text, turn.speaker) not in PROMISE_VERDICTS:
                raise ValueError(f'摘录本身不含{finding.verdict}，需重新选择摘录')
        if finding.verdict != verdict_for_category(category):
            if finding.verdict in PROMISE_VERDICTS and category == '附条件答复':
                raise ValueError('条件或限制未纳入摘录，不能作为无条件承诺')
            raise ValueError(f'摘录与认定不一致：识别为{category}')
        reasons = [
            reason for attribute, reason in _CUE_REASONS if getattr(cues(combined), attribute)
        ]
        return Assessment(finding.verdict, category, text, combined, tuple(reasons),
                          tuple(support_texts))

    def accept(self, transcript, finding):
        self.assess(transcript, finding)
        return transcript.locate(finding.passage)

    def explain(self, transcript, finding):
        assessment = self.assess(transcript, finding)
        return {
            'finding': finding.id, 'verdict': assessment.verdict,
            'category': assessment.category, 'reasons': list(assessment.reasons),
            'excerpt': assessment.excerpt,
            'support': [segment.to_dict() for segment in transcript.segments(
                getattr(finding, 'support', ())
            )],
        }


def _own_statement(text):
    """去掉转述他人话语的句子后，是否仍含坐席自己的实质性表态。"""
    from .text import clause_spans
    clauses = [text[left:right] for left, right in clause_spans(text)]
    own = [clause for clause in clauses if not cues(clause).relay]
    return any(cues(clause).statement or cues(clause).qualifier for clause in own)


def _other_subject(text):
    """坐席自己的话是否明确把承诺或查询指向另一笔订单。"""
    return cues(text).other_subject and not cues(text).same_subject


__all__ = [
    'Assessment', 'Candidate', 'DisputeReview', 'Finding',
    'PROMISE_VERDICTS', 'VERDICTS', 'find_candidates',
]
