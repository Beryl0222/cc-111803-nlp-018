"""内审与外部复核材料的分级生成、遮盖和发放回执。

外部材料的安全底线：

* 账户代号即使分散在多个发言、被逐段复读或被校订拆开，也要整体遮盖，
  且不能从摘录和随附说明里拼回去；
* 遮盖不能抹掉判定所依据的否定、条件、改口或指代，否则先核实能否换用
  意思完整、无需遮盖的片段；
* 没有合宜片段时，外部只能看到该项等待复核；
* 已交付回执保留当时的文字和音频范围，并标明是否仍可用于本次争议。
"""

import re
from copy import deepcopy

from .text import (
    classify_text, clause_spans, normalize_for_match, normalized_stream,
    semantic_differences, semantic_signature,
)
from .transcript import Passage

MASK = '■'
REVIEW_REQUIRED = '须重新选择摘录'
WAITING = '等待复核'

# 认定文字与分类的对应，用于寻找可替代摘录。
_VERDICT_CATEGORY = {
    '承诺': '承诺', '附条件答复': '附条件答复',
    '仅查询': '仅代为查询', '无法认定': None,
}


def overlaps(left, right):
    return left.turn == right.turn and left.start < right.end and right.start < left.end


def _clip(mask, turn_id, length):
    """把遮盖项裁剪到某一发言的字符范围内；完全不相交时返回空。"""
    if mask.turn != turn_id:
        return []
    left = max(0, mask.start)
    right = min(length, mask.end)
    if left >= right:
        return []
    return list(range(left, right))


def _sensitive_identifiers(transcript, masks, extra=()):
    """从遮盖项本身和显式清单取得需要保护的账户代号。"""
    values = []
    for mask in masks:
        try:
            _, text = transcript.locate(mask)
        except ValueError:
            continue
        normalized = normalize_for_match(text)
        if normalized:
            values.append(normalized)
    for value in extra:
        normalized = normalize_for_match(str(value))
        if normalized:
            values.append(normalized)
    # 长代号优先，避免短片段把长代号的遮盖判成“已遮盖”。
    return tuple(dict.fromkeys(sorted(values, key=len, reverse=True)))


def _occurrences_in(text, identifier):
    """在原文中找出 identifier 的全部字符区间（忽略标点、空白和语气词）。"""
    stream, positions = normalized_stream(text)
    ranges = []
    start = stream.find(identifier)
    while start >= 0:
        end = start + len(identifier)
        if end <= len(positions):
            ranges.append((positions[start], positions[end - 1] + 1))
        start = stream.find(identifier, start + 1)
    return ranges


def _is_fragment(run, identifier):
    """一段字母数字是否是账户代号的一部分（含被逐段复读、被校订拆开）。

    代号常被坐席分段复读（“…88”“…2173”）。任何两字符以上的片段都可能
    帮助拼回完整代号，因此一并遮盖；若遮盖后判定所依据的含义受损，外发
    会改为等待复核，而不是留下片段。
    """
    run = run.casefold()
    identifier = identifier.casefold()
    return len(run) >= 2 and any(character.isdigit() for character in run) and run in identifier


def _mask_identifier_fragments(text, hidden, sensitive):
    """在 text 中遮盖所有属于账户代号的字母数字片段。"""
    for match in re.finditer(r'[A-Za-z0-9]+', text):
        if any(_is_fragment(match.group(), identifier) for identifier in sensitive):
            hidden.update(range(match.start(), match.end()))


def _contains_identifier(payload, sensitive):
    """材料（含随附说明、字段名）里是否还能读到完整账户代号。"""
    stream, _ = normalized_stream(str(payload).replace(MASK, ''))
    return any(len(identifier) >= 2 and identifier in stream for identifier in sensitive)


def _redact_segment(transcript, passage, masks, sensitive):
    """遮盖区间并返回 (文字, 生效遮盖项, 可见字符流, 遮盖位置)。

    可见字符流保留原文顺序，只去掉遮盖符号本身；这样跨发言、跨摘录与
    随附说明的复读都能在拼接后检查。账户代号即使被逐段复读、或因校订
    拆到不同位置，其每一段可识别片段也会被遮盖。
    """
    turn, text = transcript.locate(passage)
    hidden = set()
    applied = []
    # 遮盖项可以跨发言；与本次交付区间相交、可能把代号切成可见碎片的，
    # 都记为本次生效的遮盖。
    for mask in masks:
        fragment = _clip(mask, turn.id, len(turn.text))
        hidden.update(fragment)
        if fragment:
            applied.append(mask)
    # 账户代号无论出现在哪里都整段遮盖，包括被拆开复读的位置。
    for identifier in sensitive:
        for left, right in _occurrences_in(text, identifier):
            hidden.update(range(left, right))
            applied.extend(
                mask for mask in masks
                if mask.turn == turn.id and mask.start < right and left < mask.end
                and mask not in applied
            )
    # 单独出现的“88”“2173”等片段同样可能被拼回完整代号。
    _mask_identifier_fragments(text, hidden, sensitive)

    characters = [
        MASK if offset in hidden else character
        for offset, character in enumerate(text, start=passage.start)
    ]
    redacted = ''.join(characters)
    visible_stream, _ = normalized_stream(redacted.replace(MASK, ''))
    return redacted, applied, visible_stream, _contiguous(hidden)


def _contiguous(offsets):
    """把被遮盖的字符位置合并成连续的字符范围。"""
    ranges = []
    for offset in sorted(offsets):
        if ranges and offset == ranges[-1][1]:
            ranges[-1][1] = offset + 1
        else:
            ranges.append([offset, offset + 1])
    return [tuple(pair) for pair in ranges]


def _identifier_fragments(transcript, sensitive):
    """原文中被遮盖代号的所有可见片段，用于语义对比时忽略编号差异。"""
    values = set(sensitive)
    for turn in transcript.turns.values():
        for run in re.findall(r'[A-Za-z0-9]+', turn.text):
            if any(_is_fragment(run, identifier) for identifier in sensitive):
                values.add(run.casefold())
    return tuple(values)


def _fragment_stream(segments, attachments=()):
    """材料里所有明文字符（含随附说明）按顺序拼接后的字符流。"""
    pieces = []
    for segment in segments:
        stream, _ = normalized_stream(segment['text'].replace(MASK, ''))
        pieces.append(stream)
    for attachment in attachments:
        stream, _ = normalized_stream(str(attachment).replace(MASK, ''))
        pieces.append(stream)
    return ''.join(pieces)


def _public_label(label, sensitive):
    """争议事项或认定编号本身也不能带出账户代号。"""
    if not isinstance(label, str):
        return label
    for identifier in sensitive:
        if len(identifier) >= 2 and identifier in label.casefold():
            return label.casefold().replace(identifier, MASK * 3)
    return label


class ReviewPackets:
    """按认定结果和遮盖规则生成可分别发放的复核材料。"""

    def __init__(self, review):
        self.review = review
        self._issued = {}
        self._revision = {}

    # -------------------------------------------------------------- 内部材料
    def preview(self, transcript, findings, masks, audience, matter=None):
        """``findings`` 可以是单条认定，也可以是同一通话的多条认定。"""
        if isinstance(findings, (list, tuple)):
            return self.manifest(transcript, findings, masks, audience, matter)
        return self._preview_one(transcript, findings, masks, audience)

    def manifest(self, transcript, findings, masks, audience, matter=None):
        """按争议事项逐项给出材料；可办理项与等待复核项彼此独立。"""
        items = []
        for finding in findings:
            if matter is not None and getattr(finding, 'matter', None) not in (None, matter):
                continue
            items.append(self._preview_one(transcript, finding, masks, audience))
        return {
            'call': transcript.call,
            'version': transcript.version,
            'audience': audience,
            'items': items,
            'review_required': [
                item['finding'] for item in items
                if item.get('review_status') == REVIEW_REQUIRED
            ],
        }

    def _preview_one(self, transcript, finding, masks, audience):
        if audience not in {'内审', '外部'}:
            raise ValueError('未知的材料接收方')
        assessment = self.review.assess(transcript, finding)
        segments = transcript.segments(finding.all_passages())
        turn, _ = transcript.locate(finding.passage)
        packet = {
            'call': transcript.call,
            'version': transcript.version,
            'finding': finding.id,
            'matter': getattr(finding, 'matter', None),
            'verdict': finding.verdict,
            'category': assessment.category,
            'speaker': turn.speaker,
            'audio_ms': [turn.start_ms, turn.end_ms],
            'reasons': list(assessment.reasons),
            'review_status': '可发放',
        }
        if audience == '内审':
            packet.update({
                'excerpt': assessment.excerpt,
                'segments': [segment.to_dict() for segment in segments],
                'support': [segment.to_dict() for segment in segments[1:]],
                'masks': [],
            })
            return packet
        return self._external(transcript, finding, assessment, segments, masks, packet)

    # -------------------------------------------------------------- 外部材料
    def _external(self, transcript, finding, assessment, segments, masks, packet):
        for mask in masks:
            transcript.locate(mask)
        sensitive = _sensitive_identifiers(
            transcript, masks, getattr(finding, 'sensitive', ())
        )
        ignored_numbers = _identifier_fragments(transcript, sensitive)

        redacted_segments = []
        applied_masks = []
        for segment in segments:
            text, applied, _, covered = _redact_segment(
                transcript, Passage(segment.turn, segment.start, segment.end),
                masks, sensitive,
            )
            applied_masks.extend(mask for mask in applied if mask not in applied_masks)
            redacted_segments.append({
                'turn': segment.turn, 'speaker': segment.speaker,
                'start': segment.start, 'end': segment.end,
                'text': text, 'audio_ms': [segment.start_ms, segment.end_ms],
                'masks': [
                    {'turn': mask.turn, 'start': mask.start, 'end': mask.end}
                    for mask in applied
                ],
                'redacted_ranges': [
                    {'start': segment.start + left, 'end': segment.start + right}
                    for left, right in covered
                ],
            })

        redacted_combined = ''.join(segment['text'] for segment in redacted_segments)

        # 遮盖如果抹掉了判定所需的否定、条件、改口或对象指代，就不能发放。
        changes = semantic_differences(
            assessment.combined, redacted_combined, ignore_numbers=ignored_numbers,
        )
        if changes:
            alternatives = self._alternatives(
                transcript, finding, masks, sensitive, assessment.combined,
                redacted_segments, ignored_numbers=ignored_numbers,
            )
            return self._blocked(
                packet,
                '遮盖会抹掉判定所依据的' + '、'.join(changes),
                alternatives=alternatives, sensitive=sensitive,
            )

        # 复读片段可能分散在多个发言、甚至随附说明里：拼接后仍含代号即视为泄漏。
        attachments = tuple(getattr(finding, 'notes', ()) or ())
        fragments = _fragment_stream(redacted_segments, attachments)
        if _contains_identifier(fragments, sensitive):
            alternatives = self._alternatives(
                transcript, finding, masks, sensitive, assessment.combined,
                redacted_segments, ignored_numbers=ignored_numbers,
            )
            return self._blocked(
                packet,
                '遮盖后仍能从摘录、支撑片段或随附说明拼回账户代号',
                alternatives=alternatives, sensitive=sensitive,
            )

        packet.update({
            'excerpt': redacted_segments[0]['text'],
            'segments': redacted_segments,
            'support': redacted_segments[1:],
            'masks': [
                {'turn': mask.turn, 'start': mask.start, 'end': mask.end,
                 'field': 'account_id',
                 'audio_ms': list(transcript.audio(mask))}
                for mask in applied_masks
            ],
            # 只登记字段类型，不登记被遮盖的原值。
            'redacted_fields': ['account_id'] if any(
                segment['redacted_ranges'] for segment in redacted_segments
            ) else [],
            'redaction_checked': True,
            'review_status': '可发放',
        })
        packet['matter'] = _public_label(packet.get('matter'), sensitive)
        packet['finding'] = _public_label(packet.get('finding'), sensitive)
        return packet

    def _blocked(self, packet, reason, alternatives=(), sensitive=()):
        # 外部只能看见该项等待复核；原文、替代片段和判定理由都不随之外发。
        blocked = {
            'call': packet.get('call'), 'version': packet.get('version'),
            'finding': _public_label(packet.get('finding'), sensitive),
            'matter': _public_label(packet.get('matter'), sensitive),
            'speaker': packet.get('speaker'),
            'audio_ms': packet.get('audio_ms'),
            'review_status': REVIEW_REQUIRED,
            'waiting_for_review': True,
            'blocked_reason': _public_label(reason, sensitive),
            'can_switch_excerpt': bool(alternatives),
            'alternative_count': len(alternatives),
        }
        return blocked

    def redaction_options(self, transcript, finding, masks):
        """内部核实用：列出可否换用意思完整且允许公开的片段。"""
        assessment = self.review.assess(transcript, finding)
        for mask in masks:
            transcript.locate(mask)
        sensitive = _sensitive_identifiers(
            transcript, masks, getattr(finding, 'sensitive', ())
        )
        ignored_numbers = _identifier_fragments(transcript, sensitive)
        segments = transcript.segments(finding.all_passages())
        redacted = []
        for segment in segments:
            text, _, _, _ = _redact_segment(
                transcript, Passage(segment.turn, segment.start, segment.end),
                masks, sensitive,
            )
            redacted.append({'text': text})
        changes = semantic_differences(
            assessment.combined, ''.join(item['text'] for item in redacted),
            ignore_numbers=ignored_numbers,
        )
        alternatives = self._alternatives(
            transcript, finding, masks, sensitive, assessment.combined, redacted,
            ignored_numbers=ignored_numbers,
        )
        return {
            'finding': finding.id,
            'blocked': bool(changes) or bool(alternatives),
            'changes': changes,
            'alternatives': alternatives,
            'note': '替代片段需专员核对后方可替换，外部在确认前只看到等待复核',
        }

    def _alternatives(self, transcript, finding, masks, sensitive, original,
                      redacted_segments, ignored_numbers=()):
        """找出意思完整、无需遮盖即可公开的替代片段。

        只建议与认定结论同类、保留承诺/查询/条件/否定/改口/转述性质的片段；
        它仍需专员确认后才能替换。
        """
        expected = _VERDICT_CATEGORY.get(finding.verdict)
        original_signature = semantic_signature(original, ignore_numbers=ignored_numbers)
        nature = ('promise', 'query', 'conditional', 'negated_promise',
                  'retraction', 'relay')
        original_turn, _ = transcript.locate(finding.passage)
        alternatives = []
        for turn_id in transcript.order:
            turn = transcript.turns[turn_id]
            if turn.speaker != original_turn.speaker:
                continue
            for left, right in clause_spans(turn.text):
                text = turn.text[left:right]
                if any(identifier in normalize_for_match(text) for identifier in sensitive):
                    continue
                if any(
                    mask.turn == turn_id and mask.start < right and left < mask.end
                    for mask in masks
                ):
                    continue
                category = classify_text(text, turn.speaker)
                if expected is not None and category != expected:
                    continue
                signature = semantic_signature(text, ignore_numbers=ignored_numbers)
                if any(signature[key] != original_signature[key] for key in nature):
                    continue
                if any(
                    segment['text'].replace(MASK, '') == text
                    for segment in redacted_segments
                ):
                    continue
                alternatives.append({
                    'turn': turn_id, 'speaker': turn.speaker,
                    'start': left, 'end': right, 'text': text,
                    'audio_ms': [turn.start_ms, turn.end_ms],
                    'mask_free': True,
                })
        return alternatives

    # ---------------------------------------------------------------- 发放
    def issue(self, receipt, transcript, findings, masks,
              finding_id=None, version_id=None):
        if receipt in self._issued:
            raise ValueError('发放回执已存在')
        if isinstance(findings, dict):
            if finding_id is None:
                if len(findings) != 1:
                    raise ValueError('需要指明要发放的认定')
                finding = next(iter(findings.values()))
            else:
                finding = findings[finding_id]
        else:
            finding = findings
        packet = self._preview_one(transcript, finding, masks, '外部')
        if packet['review_status'] != '可发放':
            raise ValueError('摘录须重新复核')
        self._issued[receipt] = {
            **packet,
            'receipt': receipt,
            'version_id': version_id or f'v{transcript.version}',
            'masks': deepcopy(packet.get('masks', [])),
            'still_usable': True,
            'supporting_conclusion': True,
            'revision_status': '未重新对照',
            'revision_reason': None,
            'revision_disagreements': [],
        }
        return deepcopy(self._issued[receipt])

    def archive(self, receipt, record):
        """按交付当时的材料归档回执，不依赖当前转写能否重现。"""
        if receipt in self._issued:
            raise ValueError('发放回执已存在')
        self._issued[receipt] = {
            **deepcopy(record),
            'receipt': receipt,
            'still_usable': record.get('still_usable', True),
            'supporting_conclusion': record.get('supporting_conclusion', True),
            'revision_status': record.get('revision_status', '未重新对照'),
            'revision_reason': record.get('revision_reason'),
            'revision_disagreements': deepcopy(record.get('revision_disagreements', [])),
        }
        return deepcopy(self._issued[receipt])

    def receipt(self, receipt):
        if receipt not in self._issued:
            raise KeyError(receipt)
        return deepcopy(self._issued[receipt])

    def receipts(self):
        return [deepcopy(value) for value in self._issued.values()]

    def apply_revision(self, receipt, alignment, version=None):
        """新版转写出来后更新回执的可用性；已交付文字原样保留。"""
        if receipt not in self._issued:
            raise ValueError('未知的发放回执')
        record = self._issued[receipt]
        self._revision[receipt] = deepcopy(alignment)
        record['current_version'] = version if version is not None else record.get('version')
        record['revision_status'] = alignment.get('status', '无法对应')
        record['revision_reason'] = alignment.get('reason')
        record['revision_disagreements'] = deepcopy(alignment.get('disagreements', []))
        record['still_usable'] = bool(alignment.get('reusable'))
        record['supporting_conclusion'] = bool(alignment.get('reusable'))
        return deepcopy(record)

    def external_manifest(self, receipts=None, matter=None):
        """外部可见的回执清单：只含可公开材料或“等待复核”标记。"""
        selected = receipts if receipts is not None else list(self._issued)
        items = []
        for receipt in selected:
            record = self._issued.get(receipt)
            if record is None:
                raise ValueError(f'未知的发放回执：{receipt}')
            if matter is not None and record.get('matter') not in (None, matter):
                continue
            if record.get('review_status') != '可发放':
                items.append({
                    'receipt': receipt, 'finding': record['finding'],
                    'review_status': WAITING, 'waiting_for_review': True,
                })
                continue
            items.append({
                'receipt': receipt,
                'finding': record['finding'],
                'verdict': record['verdict'],
                'excerpt': record.get('excerpt'),
                'segments': deepcopy(record.get('segments', [])),
                'support': deepcopy(record.get('support', [])),
                'audio_ms': record.get('audio_ms'),
                'masks': deepcopy(record.get('masks', [])),
                'still_usable': record.get('still_usable', True),
                'supporting_conclusion': record.get('supporting_conclusion', True),
                'revision_status': record.get('revision_status'),
                'review_status': '可发放',
            })
        return items


__all__ = [
    'MASK', 'REVIEW_REQUIRED', 'WAITING', 'ReviewPackets', 'overlaps',
]
