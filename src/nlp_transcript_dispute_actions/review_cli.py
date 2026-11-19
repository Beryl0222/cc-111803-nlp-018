import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .adjudication import DisputeReview, Finding, find_candidates
from .release import ReviewPackets
from .revision import Revision
from .transcript import Passage, Transcript


@dataclass(frozen=True)
class Case:
    """已载入的复核案例；可按四元组解包以兼容既有调用。"""

    transcript: Transcript
    findings: dict
    masks: tuple
    packets: ReviewPackets
    data: dict

    def __iter__(self):
        return iter((self.transcript, self.findings, self.masks, self.packets))


def _passages(values):
    return tuple(Passage(**value) for value in values or ())


def _finding(value):
    return Finding(
        value['id'], value['verdict'], value['reviewer'], Passage(**value['passage']),
        _passages(value.get('essential')), _passages(value.get('support')),
        value.get('matter'), tuple(value.get('sensitive', ())),
    )


def load_case(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    transcript = Transcript.from_dict(data)
    findings = {finding['id']: _finding(finding) for finding in data['findings']}
    masks = tuple(Passage(**value) for value in data.get('masks', ()))
    packets = ReviewPackets(DisputeReview(data['roles']))
    for issuance in data.get('issuance', ()):
        _issue(packets, transcript, findings, masks, issuance)
    return Case(transcript, findings, masks, packets, data)


def _issue(packets, transcript, findings, masks, issuance):
    if 'text' not in issuance:
        return packets.issue(
            issuance['receipt'], transcript, findings[issuance['finding']], masks,
        )
    # 历史回执按交付当时的文字归档，不因当前转写或遮盖重新生成而改变。
    finding = findings[issuance['finding']]
    turn, _ = transcript.locate(finding.passage)
    record = {
        'call': transcript.call,
        'version': issuance.get('version', transcript.version),
        'version_id': f"v{issuance.get('version', transcript.version)}",
        'finding': finding.id,
        'matter': finding.matter,
        'verdict': finding.verdict,
        'speaker': turn.speaker,
        'audio_ms': issuance.get('audio_ms', [turn.start_ms, turn.end_ms]),
        'review_status': issuance.get('review_status', '可发放'),
        'excerpt': issuance['text'],
        'delivered_text': issuance['text'],
        'masks': issuance.get('masks', []),
    }
    return packets.archive(issuance['receipt'], record)


def load_revision_case(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    previous = Transcript.from_dict(data['previous'])
    current = Transcript.from_dict(data['current'])
    revision = Revision(previous, current)
    findings = {value['id']: _finding(value) for value in data['findings']}
    return data, previous, current, revision, findings


def revision_manifest(revision, findings):
    """逐项对照；无法对应的具体分歧留给审核员，独立事项仍可办理。"""
    items = []
    for finding in findings.values():
        alignment = revision.align_finding(finding)
        alignment['version'] = revision.current.version
        alignment['matter'] = finding.matter
        items.append(alignment)
    return {
        'call': revision.current.call,
        'from_version': revision.previous.version,
        'to_version': revision.current.version,
        'items': items,
        'review_required': [item['finding'] for item in items if not item['reusable']],
        'open_matters': [
            item['finding'] for item in items
            if not item['reusable'] and not item['disagreements']
        ],
    }


def _run(action, args, parser):
    if action == 'revision':
        data, _, _, revision, findings = load_revision_case(args.case)
        return revision_manifest(revision, findings)

    case = load_case(args.case)
    transcript, findings, masks, packets = case
    data = case.data
    if action == 'candidates':
        keywords = tuple(args.keyword) if args.keyword else None
        candidates = find_candidates(transcript, keywords) if keywords else find_candidates(transcript)
        return {
            'call': transcript.call, 'version': transcript.version,
            'candidates': [candidate.to_dict() for candidate in candidates],
        }
    if action == 'explain':
        if args.finding not in findings:
            parser.error(f'未知的认定：{args.finding}')
        return DisputeReview(data['roles']).explain(transcript, findings[args.finding])
    if action == 'receipt':
        record = packets.receipt(args.number)
        if args.align:
            if not args.revision_file:
                parser.error('重新对照需要 --revision-file')
            _, _, _, revision, revision_findings = load_revision_case(args.revision_file)
            finding = revision_findings.get(record['finding'])
            if finding is None:
                parser.error(f'新版没有对应认定：{record["finding"]}')
            alignment = revision.align_finding(finding)
            alignment['version'] = revision.current.version
            return packets.apply_revision(args.number, alignment, revision.current.version)
        return record
    if action == 'external':
        return packets.external_manifest(args.receipt or None, args.matter)

    audience = args.audience
    if args.finding:
        if args.finding not in findings:
            parser.error(f'未知的认定：{args.finding}')
        selected = findings[args.finding]
        if args.all:
            selected = list(findings.values())
        return packets.preview(transcript, selected, masks, audience, args.matter)
    if not args.all:
        parser.error('需要指定认定编号，或使用 --all 查看全部争议事项')
    return packets.preview(transcript, list(findings.values()), masks, audience, args.matter)


def main(argv=None):
    parser = argparse.ArgumentParser(description='通话争议复核')
    parser.add_argument('--case', default='fixtures/review_case.json')
    commands = parser.add_subparsers(dest='action', required=True)
    preview = commands.add_parser('preview')
    preview.add_argument('finding', nargs='?')
    preview.add_argument('--audience', choices=['内审', '外部'], required=True)
    preview.add_argument('--all', action='store_true')
    preview.add_argument('--matter')
    candidates = commands.add_parser('candidates')
    candidates.add_argument('--keyword', action='append')
    explain = commands.add_parser('explain')
    explain.add_argument('finding')
    revision = commands.add_parser('revision')
    revision.add_argument('--case', dest='case', required=True)
    receipt = commands.add_parser('receipt')
    receipt.add_argument('number')
    receipt.add_argument('--revision-file')
    receipt.add_argument('--align', action='store_true')
    external = commands.add_parser('external')
    external.add_argument('--receipt', action='append')
    external.add_argument('--matter')
    args = parser.parse_args(argv)
    try:
        result = _run(args.action, args, parser)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
