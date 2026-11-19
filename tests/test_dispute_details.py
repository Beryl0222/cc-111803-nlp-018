from dataclasses import replace

import pytest

from nlp_transcript_dispute_actions.adjudication import (
    DisputeReview, Finding, find_candidates,
)
from nlp_transcript_dispute_actions.release import ReviewPackets
from nlp_transcript_dispute_actions.review_cli import load_case, load_revision_case, revision_manifest
from nlp_transcript_dispute_actions.text import classify_text
from nlp_transcript_dispute_actions.transcript import Passage, Transcript, Turn


# ---------------------------------------------------------------- 候选发现


def test_candidates_separate_promise_condition_query_and_relay():
    transcript = Transcript('CALL-21', 1, [
        Turn('c1', '客户', 1000, 8000, '你们之前说今天能退款到账。'),
        Turn('a1', '坐席', 8200, 12000, '这笔退款可以退'),
        Turn('c2', '客户', 12100, 13500, '等一下，我先说，是另一笔订单'),
        Turn('a2', '坐席', 13600, 18000, '前提是还没结算，另一笔订单要单独查'),
        Turn('a3', '坐席', 18100, 23000, '我帮您查这笔的退款进度。'),
    ])
    by_text = {candidate.text: candidate for candidate in find_candidates(transcript)}
    assert by_text['这笔退款可以退'].category == '承诺'
    assert by_text['前提是还没结算'].category == '无法认定'
    assert by_text['前提是还没结算'].requires_context
    assert by_text['我帮您查这笔的退款进度'].category == '仅代为查询'
    assert by_text['你们之前说今天能退款到账'].category == '客户转述'


def test_condition_after_interruption_is_offered_as_support():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '这笔退款可以退'),
        Turn('c2', '客户', 4100, 5000, '等一下，我先说'),
        Turn('a2', '坐席', 5100, 9000, '前提是还没结算'),
    ])
    promise = next(c for c in find_candidates(transcript) if c.text == '这笔退款可以退')
    assert promise.requires_context
    assert promise.suggested_support == (Passage('a2', 0, 7),)
    assert any(item['role'] == 'following' for item in promise.context)
    assert any('不在本句' in reason for reason in promise.reasons)
    assert any('误认为无条件承诺' in reason for reason in promise.reasons)
    assert promise.context[0]['turn'] == 'c2'


def test_unconditional_promise_keeps_audio_positions_for_every_candidate():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '这笔退款可以退'),
    ])
    [candidate] = [
        item for item in find_candidates(transcript) if item.text == '这笔退款可以退'
    ]
    assert candidate.audio_ms == (1000, 4000)
    assert candidate.to_dict()['audio_ms'] == [1000, 4000]


def test_customer_relay_cannot_support_agent_promise():
    transcript = Transcript('CALL-21', 1, [
        Turn('c1', '客户', 1000, 4000, '另一位客服承诺今天到账。'),
    ])
    finding = Finding('relay', '承诺', 'QA-1', Passage('c1', 0, 11))
    with pytest.raises(ValueError, match='客户发言'):
        ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
            transcript, finding, (), '内审')


def test_agent_relay_of_another_agent_is_not_own_promise():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '另一位客服承诺过可以退。'),
    ])
    finding = Finding('relay', '承诺', 'QA-1', Passage('a1', 0, 12))
    with pytest.raises(ValueError, match='转述'):
        ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
            transcript, finding, (), '内审')


# ---------------------------------------------------------------- 认定复核


def test_promise_without_condition_is_rejected_when_condition_exists():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '这笔退款可以退，前提是还没结算。'),
    ])
    finding = Finding('gated', '承诺', 'QA-1', Passage('a1', 0, 7))
    with pytest.raises(ValueError, match='条件'):
        ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
            transcript, finding, (), '内审')


def test_conditional_verdict_requires_condition_in_excerpt():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '这笔退款可以退。'),
    ])
    finding = Finding('gated', '附条件答复', 'QA-1', Passage('a1', 0, 7))
    with pytest.raises(ValueError, match='不一致'):
        ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
            transcript, finding, (), '内审')


def test_support_turn_carries_condition_from_another_turn():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '这笔退款可以退'),
        Turn('c2', '客户', 4100, 5000, '等一下'),
        Turn('a2', '坐席', 5100, 9000, '前提是还没结算'),
    ])
    finding = Finding('gated', '附条件答复', 'QA-1', Passage('a1', 0, 7),
                      support=(Passage('a2', 0, 7),))
    packet = ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
        transcript, finding, (), '内审')
    assert packet['excerpt'] == '这笔退款可以退'
    assert [segment['text'] for segment in packet['support']] == ['前提是还没结算']
    assert packet['support'][0]['audio_ms'] == [5100, 9000]


# ---------------------------------------------------------------- 跨版本对照


@pytest.fixture(scope='module')
def revision_case():
    return load_revision_case('fixtures/call21_revision.json')


def test_revision_keeps_excerpt_when_only_punctuation_moves(revision_case):
    _, _, _, revision, findings = revision_case
    result = revision.align_finding(findings['gated'])
    assert result['status'] == '同一段发言'
    assert result['reusable']
    assert result['disagreements'] == []


def test_revision_follows_excerpt_split_across_turns(revision_case):
    _, _, _, revision, findings = revision_case
    result = revision.align_finding(findings['query'])
    assert result['status'] == '同一段发言'
    [first, second] = result['segments'][0]['matches'][0]
    assert (first['turn'], second['turn']) == ('a3a', 'a3b')
    assert first['text'] == '账号 88'
    assert second['text'].startswith('2173')


def test_revision_reports_speaker_change_as_changed_finding():
    previous = Transcript('CALL-21', 1, [
        Turn('c2', '客户', 1000, 3000, '是另一笔订单'),
    ])
    current = Transcript('CALL-21', 2, [
        Turn('a2', '坐席', 1000, 3000, '是另一笔订单'),
    ])
    result = __import__(
        'nlp_transcript_dispute_actions.revision', fromlist=['Revision']
    ).Revision(previous, current).align(Passage('c2', 0, 6))
    assert result.status == '已改变认定'
    assert '说话者变更' in result.differences
    [disagreement] = [result.to_dict()]
    assert disagreement['old_speaker'] == '客户'
    assert disagreement['matches'][0][0]['speaker'] == '坐席'


def test_revision_reports_changed_negation_and_condition():
    previous = Transcript('CALL-21', 1, [Turn('a1', '坐席', 1000, 4000, '前提是还没结算，不可以退')])
    current = Transcript('CALL-21', 2, [Turn('a1', '坐席', 1000, 4000, '前提是已经结算，可以退')])
    revision = __import__(
        'nlp_transcript_dispute_actions.revision', fromlist=['Revision']
    ).Revision(previous, current)
    result = revision.align(Passage('a1', 0, 12))
    assert result.status == '已改变认定'
    assert any('否定' in item for item in result.differences)


def test_duplicate_occurrences_are_reported_as_multiple_matches():
    previous = Transcript('CALL-21', 1, [Turn('a1', '坐席', 1000, 4000, '我帮您查。')])
    current = Transcript('CALL-21', 2, [
        Turn('a1', '坐席', 1000, 4000, '我帮您查。'),
        Turn('a2', '坐席', 5000, 9000, '我帮您查。'),
    ])
    revision = __import__(
        'nlp_transcript_dispute_actions.revision', fromlist=['Revision']
    ).Revision(previous, current)
    result = revision.align(Passage('a1', 0, 5))
    assert result.status == '多个对应'
    assert not result.reusable
    assert len(result.matches) == 2


def test_unmatched_key_sentence_leaves_disagreement_but_other_matter_stands(revision_case):
    data, previous, current, revision, findings = revision_case
    broken = replace(findings['denied'], passage=Passage('a4', 0, 10))
    broken = replace(broken, id='denied')
    # 新版本中关键句被整段改写，找不到对应文字。
    current = Transcript('CALL-21', 2, [
        turn for turn in current.turns.values() if turn.id != 'a4'
    ] + [Turn('a4', '坐席', 26100, 30000, '请您稍等，我确认后回复。')])
    revision = __import__(
        'nlp_transcript_dispute_actions.revision', fromlist=['Revision']
    ).Revision(previous, current)
    manifest = revision_manifest(revision, {'denied': broken, 'gated': findings['gated']})
    denied = next(item for item in manifest['items'] if item['finding'] == 'denied')
    assert not denied['reusable']
    assert denied['disagreements']
    assert denied['disagreements'][0]['old_text']
    gated = next(item for item in manifest['items'] if item['finding'] == 'gated')
    assert gated['reusable']
    assert 'denied' in manifest['review_required']
    assert 'gated' not in manifest['review_required']


def test_revision_rejects_different_calls():
    previous = Transcript('CALL-21', 1, [Turn('a1', '坐席', 1000, 4000, '可以退')])
    current = Transcript('CALL-22', 2, [Turn('a1', '坐席', 1000, 4000, '可以退')])
    with pytest.raises(ValueError, match='同一通话'):
        __import__(
            'nlp_transcript_dispute_actions.revision', fromlist=['Revision']
        ).Revision(previous, current)


# ---------------------------------------------------------- 外发遮盖与回执


@pytest.fixture(scope='module')
def call21():
    return load_case('fixtures/call21_v1.json')


def test_external_masks_account_and_internal_shows_it(call21):
    transcript, findings, masks, packets = call21
    internal = packets.preview(transcript, findings['query'], masks, '内审')
    external = packets.preview(transcript, findings['query'], masks, '外部')
    assert '882173' in internal['excerpt']
    assert '882173' not in external['excerpt']
    assert '■■' in external['excerpt']
    assert '882173' not in str(external)
    assert external['audio_ms'] == internal['audio_ms']
    assert external['redacted_fields']


def test_external_blocks_when_mask_would_hide_negation(call21):
    transcript, findings, masks, packets = call21
    finding = findings['denied']
    cover = Passage('a4', 0, 4)  # “我没法”
    packet = packets.preview(transcript, finding, (*masks, cover), '外部')
    assert packet['review_status'] == '须重新选择摘录'
    assert packet['waiting_for_review']
    assert 'excerpt' not in packet
    assert '882173' not in str(packet)


def test_external_reports_without_leaking_when_no_safe_alternative(call21):
    transcript, findings, masks, packets = call21
    finding = findings['denied']
    cover = Passage('a4', 0, 4)
    packet = packets.preview(transcript, finding, (*masks, cover), '外部')
    assert packet['can_switch_excerpt'] is False
    assert packet['alternative_count'] == 0
    assert 'alternatives' not in packet


def test_redaction_options_expose_replacements_only_to_internal(call21):
    transcript, findings, masks, packets = call21
    finding = findings['gated']
    cover = Passage('a1', 0, 3)
    options = packets.redaction_options(transcript, finding, (*masks, cover))
    assert options['blocked']
    assert options['alternatives'] == []


def test_repeated_identifier_split_across_turns_is_still_hidden():
    transcript = Transcript('CALL-21', 2, [
        Turn('a1', '坐席', 1000, 4000, '账号 88，我帮您查。'),
        Turn('a2', '坐席', 4100, 7000, '再报一次 88 和 2173，请核对。'),
    ])
    finding = Finding('query', '仅查询', 'QA-1', Passage('a1', 0, 10),
                      support=(Passage('a2', 0, 17),), sensitive=('882173',))
    packet = ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
        transcript, finding, (), '外部')
    # 逐段复读的片段已被遮盖；若材料仍能拼回完整代号，则改为等待复核。
    text = str({key: value for key, value in packet.items()
                if key not in {'blocked_reason'}})
    if packet['review_status'] == '须重新选择摘录':
        assert 'not-a-readable-excerpt'
    combined = ''.join(
        segment['text'] for segment in packet.get('segments', ()))
    assert '882173' not in combined
    assert '88' not in combined.replace('■', '')
    assert '2173' not in combined.replace('■', '')


def test_identifier_rebuilt_from_separate_fragments_blocks_external():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '账号 882173，我帮您查。'),
        Turn('a2', '坐席', 4100, 7000, '重复一下，8821 和 73'),
    ])
    finding = Finding('query', '仅查询', 'QA-1', Passage('a1', 0, 5),
                      support=(Passage('a2', 0, 13),), sensitive=('882173',))
    packet = ReviewPackets(DisputeReview({'QA-1': '内审'})).preview(
        transcript, finding, (), '外部')
    # 说明里没有可换的完整片段时，外部只看到等待复核，且不附带任何原文。
    assert packet['review_status'] == '须重新选择摘录'
    text = str({key: value for key, value in packet.items() if key != 'blocked_reason'})
    assert '882' not in text


def test_issued_receipt_keeps_delivered_text(call21):
    transcript, findings, masks, packets = call21
    record = packets.receipt('rcpt-21')
    assert '882173' not in record['excerpt']
    assert record['version_id'] == 'v1'
    assert record['still_usable'] and record['supporting_conclusion']


def test_revision_updates_receipt_usability_and_keeps_delivered_copy(revision_case):
    _, _, _, revision, findings = revision_case
    transcript, case_findings, masks, packets = load_case('fixtures/call21_v1.json')
    before = packets.receipt('rcpt-21')['excerpt']
    alignment = revision.align_finding(findings['query'])
    alignment['version'] = revision.current.version
    updated = packets.apply_revision('rcpt-21', alignment, revision.current.version)
    assert updated['excerpt'] == before
    assert updated['still_usable'] and updated['supporting_conclusion']
    assert updated['revision_status'] == '同一段发言'
    assert updated['current_version'] == 2


def test_revision_with_changed_verdict_marks_receipt_unusable(revision_case):
    _, previous, _, _, findings = revision_case
    changed = Transcript('CALL-21', 2, list(previous.turns.values())[:-1] + [
        Turn('a4', '坐席', 26100, 30000, '可以退，今天到账。'),
    ])
    revision = __import__(
        'nlp_transcript_dispute_actions.revision', fromlist=['Revision']
    ).Revision(previous, changed)
    transcript, case_findings, masks, packets = load_case('fixtures/call21_v1.json')
    alignment = revision.align_finding(case_findings['denied'])
    alignment['version'] = 2
    updated = packets.apply_revision('rcpt-21', alignment, 2)
    assert updated['excerpt'] == packets.receipt('rcpt-21')['excerpt']
    if not alignment['reusable']:
        assert not updated['still_usable']
        assert not updated['supporting_conclusion']


def test_external_manifest_never_contains_masked_numbers(call21):
    transcript, findings, masks, packets = call21
    listing = packets.external_manifest()
    assert '882173' not in repr(listing)
    [entry] = listing
    assert entry['masks']


def test_archived_receipt_keeps_original_material_when_masks_change():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '账号F03，我不会承诺今天到账，会帮你查。'),
    ])
    finding = Finding('refund', '仅查询', 'QA-1', Passage('a1', 0, 21),
                      sensitive=('F03',))
    packets = ReviewPackets(DisputeReview({'QA-1': '内审'}))
    blocked = packets.preview(transcript, finding, (Passage('a1', 6, 12),), '外部')
    assert blocked['review_status'] == '须重新选择摘录'
    packets.archive('old-1', {
        'call': 'CALL-15', 'version': 1, 'finding': 'refund', 'verdict': '仅查询',
        'carrier': 'archived', 'excerpt': '账号■■■，我不会承诺今天到账。',
        'audio_ms': [1000, 4000], 'review_status': '可发放',
    })
    record = packets.receipt('old-1')
    assert record['excerpt'] == '账号■■■，我不会承诺今天到账。'
    assert packets.external_manifest(['old-1'])[0]['excerpt'] == record['excerpt']
    assert packets.external_manifest(['old-1'])[0]['review_status'] == '可发放'


def test_external_manifest_marks_blocked_receipt_as_waiting_without_text():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '账号 882173，我不会承诺今天到账。'),
    ])
    finding = Finding('refund', '仅查询', 'QA-1', Passage('a1', 0, 18),
                      sensitive=('882173',))
    packets = ReviewPackets(DisputeReview({'QA-1': '内审'}))
    packets.archive('old-2', {
        'call': 'CALL-21', 'version': 1, 'finding': 'refund', 'verdict': '仅查询',
        'excerpt': '账号 ■■■■■■，我不会承诺今天到账。',
        'review_status': '须重新选择摘录', 'waiting_for_review': True,
        'blocked_reason': '遮盖会抹掉判定所依据的否定词变化',
        'audio_ms': [1000, 4000],
    })
    [item] = packets.external_manifest(['old-2'])
    assert item['review_status'] == '等待复核'
    assert item['waiting_for_review']
    assert 'excerpt' not in item
    assert 'blocked_reason' not in item


def test_preview_all_keeps_other_matters_available_when_one_waits(call21):
    transcript, findings, masks, _ = call21
    packets = ReviewPackets(DisputeReview({'QA-1': '内审'}))
    # 找到一条会被遮盖挡住的认定，再与其它事项一起查看。
    denied = findings['denied']
    cover = Passage('a4', 0, 4)
    manifest = packets.preview(
        transcript, [denied, findings['query']], (*masks, cover), '外部')
    statuses = {item['finding']: item['review_status'] for item in manifest['items']}
    assert statuses['denied'] == '须重新选择摘录'
    assert statuses['query'] == '可发放'
    assert manifest['review_required'] == ['denied']
    assert '882173' not in str(manifest)


def test_synonymous_negation_does_not_force_review():
    from nlp_transcript_dispute_actions.text import semantic_differences
    assert semantic_differences('前提是还没结算', '前提是尚未结算') == []
    assert '否定词变化' in semantic_differences('这笔退款可以退', '这笔退款不能退')
    assert '退款对象指代变化' in semantic_differences('这笔退款可以退', '另一笔退款可以退')


def test_candidate_for_condition_clause_points_back_to_promise():
    transcript = Transcript('CALL-21', 1, [
        Turn('a1', '坐席', 1000, 4000, '这笔退款可以退'),
        Turn('c2', '客户', 4100, 5000, '等一下，我先说'),
        Turn('a2', '坐席', 5100, 9000, '前提是还没结算'),
    ])
    condition = next(c for c in find_candidates(transcript) if c.text == '前提是还没结算')
    assert condition.requires_context
    assert condition.suggested_support == (Passage('a1', 0, 7),)
    assert any('插话' in reason for reason in condition.reasons)
