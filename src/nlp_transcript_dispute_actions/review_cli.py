import argparse
import json
from pathlib import Path

from .adjudication import DisputeReview, Finding
from .release import ReviewPackets
from .transcript import Passage, Transcript, Turn


def load_case(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    transcript = Transcript(data['call'], data['version'], map(lambda turn: Turn(**turn), data['turns']))
    findings = {
        finding['id']: Finding(finding['id'], finding['verdict'], finding['reviewer'],
                               Passage(**finding['passage']), tuple(Passage(**part) for part in finding['essential']))
        for finding in data['findings']
    }
    masks = tuple(Passage(**value) for value in data['masks'])
    packets = ReviewPackets(DisputeReview(data['roles']))
    for issuance in data['issuance']:
        packets.issue(issuance['receipt'], transcript, findings[issuance['finding']], masks)
    return transcript, findings, masks, packets


def main(argv=None):
    parser = argparse.ArgumentParser(description='通话争议复核')
    parser.add_argument('--case', default='fixtures/review_case.json')
    commands = parser.add_subparsers(dest='action', required=True)
    preview = commands.add_parser('preview')
    preview.add_argument('finding')
    preview.add_argument('--audience', choices=['内审', '外部'], required=True)
    commands.add_parser('receipt').add_argument('number')
    args = parser.parse_args(argv)
    try:
        transcript, findings, masks, packets = load_case(args.case)
        result = packets.receipt(args.number) if args.action == 'receipt' else packets.preview(
            transcript, findings[args.finding], masks, args.audience
        )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
