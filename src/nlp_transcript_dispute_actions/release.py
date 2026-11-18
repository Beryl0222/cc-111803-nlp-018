from copy import deepcopy


def overlaps(left, right):
    return left.turn == right.turn and left.start < right.end and right.start < left.end


class ReviewPackets:
    """按认定结果和遮盖规则生成可分别发放的复核材料。"""

    def __init__(self, review):
        self.review = review
        self._issued = {}

    def preview(self, transcript, finding, masks, audience):
        turn, text = self.review.accept(transcript, finding)
        if audience not in {'内审', '外部'}:
            raise ValueError('未知的材料接收方')
        packet = {
            'call': transcript.call, 'version': transcript.version,
            'finding': finding.id, 'verdict': finding.verdict, 'speaker': turn.speaker,
            'audio_ms': [turn.start_ms, turn.end_ms], 'review_status': '可发放',
        }
        if audience == '外部':
            for mask in masks:
                transcript.locate(mask)
                if any(overlaps(mask, required) for required in finding.essential):
                    return {**packet, 'review_status': '须重新选择摘录'}
            text = ''.join(
                '■' if any(mask.turn == turn.id and mask.start <= position < mask.end for mask in masks)
                else character
                for position, character in enumerate(text, start=finding.passage.start)
            )
        return {**packet, 'excerpt': text}

    def issue(self, receipt, transcript, finding, masks):
        if receipt in self._issued:
            raise ValueError('发放回执已存在')
        packet = self.preview(transcript, finding, masks, '外部')
        if packet['review_status'] != '可发放':
            raise ValueError('摘录须重新复核')
        self._issued[receipt] = packet

    def receipt(self, receipt):
        return deepcopy(self._issued[receipt])
