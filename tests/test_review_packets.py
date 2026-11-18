from dataclasses import replace
from pathlib import Path

import pytest

from nlp_transcript_dispute_actions.review_cli import load_case
from nlp_transcript_dispute_actions.transcript import Passage


@pytest.fixture
def case():
    return load_case(Path(__file__).parents[1] / 'fixtures' / 'review_case.json')


def test_internal_and_external_excerpt_preserve_same_audio_window(case):
    transcript, findings, masks, packets = case
    internal = packets.preview(transcript, findings['refund'], masks, '内审')
    external = packets.receipt('packet-8')
    assert 'F03' in internal['excerpt']
    assert 'F03' not in external['excerpt']
    assert '不会承诺' in external['excerpt']
    assert external['audio_ms'] == internal['audio_ms'] == [19000, 24700]


def test_overlapping_redaction_blocks_external_only_and_preserves_issued_copy(case):
    transcript, findings, masks, packets = case
    mask = Passage('agent-01', 6, 11)
    packet = packets.preview(transcript, findings['refund'], (*masks, mask), '外部')
    assert packet['review_status'] == '须重新选择摘录'
    assert 'excerpt' not in packet
    assert '不会承诺' in packets.receipt('packet-8')['excerpt']


def test_customer_cannot_be_used_as_source_of_agent_promise(case):
    transcript, findings, masks, packets = case
    finding = replace(findings['refund'], verdict='承诺', passage=Passage('customer-01', 0, 11), essential=())
    with pytest.raises(ValueError, match='客户发言'):
        packets.preview(transcript, finding, masks, '内审')
