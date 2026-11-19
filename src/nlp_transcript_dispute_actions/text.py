"""中文转写的规范化、线索识别和跨版本语义指纹。

这里刻意把“措辞”和“认定含义”分开：标点、断句和口语填充词只影响定位，
不会进入语义指纹；否定、条件、改口、转述、对象指代和编号才影响认定。
"""

import re
import unicodedata


# 句读、空白和符号只用于切分，不参与定位和判读。
def _is_separator(character):
    return character.isspace() or unicodedata.category(character).startswith('P')


# 人工校订常删改这些口语成分；它们不应要求专员重复审核。
_FILLERS = (
    '也就是说', '就是说', '然后呢', '这个呢', '那个呢', '然后', '那个', '这个',
    '您好', '你好', '嗯', '啊', '哦', '呃', '额', '呀', '呢', '吧', '嘛',
)

_NEGATION = re.compile(r'(无法|不能|不会|不予|没有|尚未|并未|绝不|从没|不|没|未|无|非|别|勿|莫)')
_PROMISE = re.compile(
    r'(承诺|保证|答应|可以退|能退|会退|给您退|给你退|帮您退|帮你退|退给您|退给你'
    r'|退款|退回|原路退回|一定|肯定|到账|打款)'
)
# “查退款进度”“看退款到账没有”是在说明查询动作，不是在答应退款。
_QUERY_CONTEXT = re.compile(
    r'(查|查询|确认|核实|跟进|了解|问|看|盯)[^，。；！？]{0,6}(退款|退回|到账|打款|进度)'
)
_QUERY = re.compile(r'(查询|查一下|检查|查|核实|确认一下|确认|了解|看一下|看看|跟进|反馈|问一下|帮您问|进度|处理中|调查)')
_CONDITION = re.compile(
    r'(前提|如果|要是|只要|除非|但是|不过|需要|需先|先要|才能|才可|之后|以后'
    r'|审核|结算|符合|满足|到账后|到账之后|成功|完成)'
)
_RETRACTION = re.compile(r'(改口|更正|说错|收回|不算|以.{0,8}为准|刚才说|重新说明|纠正)')
_RELAY = re.compile(
    r'(他说|她说|客服说|另一位|另一个客服|之前有人|有人承诺|别人说|转述|你们的人'
    r'|那位客服|别的客服|上面说|同事说|你们之前说|你们说|你说过|说好|之前说|答应过)'
)
_INTERRUPT = re.compile(r'(等一下|等等|打断|你先|停一下|我先说|别急|稍等|不好意思)')
_OTHER_SUBJECT = re.compile(r'(另一笔|另一个|另一单|另外一笔|另外一个|别的订单|其他订单|不是这单|不同订单)')
_SAME_SUBJECT = re.compile(r'(这单|这个订单|该订单|本单|同一单|同一订单|这笔|这笔退款|这一单|这次退款)')
_NEGATED_PROMISE = re.compile(
    r'(无法|没法|没办法|不能|不会|不予|没有|尚未|并未|绝不|从没|别|不要)'
    r'[^，。；！？]{0,8}?'
    r'(承诺|保证|答应|可以退|能退|会退|退款|退回|到账|打款)'
)
# “不好意思”“不过”“不仅”等不是对承诺的否定。
_NEGATED_PROMISE_SAFE = re.compile(r'(不好意思|不过|不仅|不但|不久|不妨|不少|不然)')
_NEGATIVE_RESULT = re.compile(r'(退不了|退不成|不给退|不能退|不可以退|不会退|无法退)')
_NUMBER = re.compile(r'[A-Za-z]*\d[A-Za-z0-9-]*')
_ORDER_REFERENCE = re.compile(r'(?:订单|单号|退款单|编号|账户|账号)[^，。；！？]{0,6}?([A-Za-z]*\d[A-Za-z0-9-]*)')


def normalize_text(value):
    if not isinstance(value, str):
        raise TypeError('文本必须是字符串')
    return re.sub(r'\s+', ' ', value).strip()


def _normalized_characters(value):
    """返回 (规范化字符流, 每个字符对应的原文位置)。"""
    characters = []
    positions = []
    for index, character in enumerate(value):
        for normalized in unicodedata.normalize('NFKC', character).casefold():
            if _is_separator(normalized):
                continue
            characters.append(normalized)
            positions.append(index)
    return characters, positions


def _remove_filler_run(characters, positions):
    stream = ''.join(characters)
    for filler in sorted(set(_FILLERS), key=len, reverse=True):
        normalized = ''.join(
            character for character in unicodedata.normalize('NFKC', filler).casefold()
            if not _is_separator(character)
        )
        if not normalized:
            continue
        start = 0
        while True:
            index = stream.find(normalized, start)
            if index < 0:
                break
            stream = stream[:index] + stream[index + len(normalized):]
            positions = positions[:index] + positions[index + len(normalized):]
            start = index
    return stream, positions


def normalized_stream(value):
    """用于跨版本定位的字符流，带原文位置映射。

    返回的 ``positions[i]`` 是字符流第 i 个字符在原文中的下标。标点、空白
    和口语填充词被移除，因此换行、重新断句或补删语气词不会破坏对应。
    """
    if not isinstance(value, str):
        raise TypeError('文本必须是字符串')
    characters, positions = _normalized_characters(value)
    stream, positions = _remove_filler_run(characters, positions)
    return stream, positions


def normalize_for_match(value):
    """只有标点、空白、全半角和口语填充词被忽略的规范化形式。"""
    return normalized_stream(value)[0]


def tokens(value):
    return re.findall(r'[A-Za-z0-9]+|[\u3400-\u9fff]', value.casefold())


def clause_spans(text):
    """按中文句读切出原文区间，返回保留字符位置的 (start, end)。"""
    spans = []
    start = 0
    for match in re.finditer(r'[，。！？；：、\n\r]+', text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [
        (left, right) for left, right in spans
        if normalize_for_match(text[left:right])
    ]


class Cues:
    """一段原文中与承诺认定有关的线索。"""

    __slots__ = (
        'promise', 'query', 'conditional', 'negated_promise', 'retraction',
        'relay', 'interruption', 'other_subject', 'same_subject',
        '_negations', '_conditions', '_numbers', '_order_references',
    )

    def __init__(self, text):
        self.promise = bool(_PROMISE.search(text))
        self.query = bool(_QUERY.search(text))
        self.conditional = bool(_CONDITION.search(text))
        self.negated_promise = (
            bool(_NEGATED_PROMISE.search(_NEGATED_PROMISE_SAFE.sub('', text)))
            or bool(_NEGATIVE_RESULT.search(text))
            or self._negated_modality(text)
        )
        self.retraction = bool(_RETRACTION.search(text))
        self.relay = bool(_RELAY.search(text))
        self.interruption = bool(_INTERRUPT.search(text))
        self.other_subject = bool(_OTHER_SUBJECT.search(text))
        self.same_subject = bool(_SAME_SUBJECT.search(text))
        self._negations = tuple(sorted(set(_NEGATION.findall(text))))
        self._conditions = tuple(sorted(set(_CONDITION.findall(text))))
        self._numbers = tuple(sorted(set(_NUMBER.findall(text))))
        self._order_references = tuple(sorted(set(_ORDER_REFERENCE.findall(text))))
        if self.query and not self.negated_promise:
            stripped = _QUERY_CONTEXT.sub('', text)
            self.promise = bool(_PROMISE.search(stripped))

    @property
    def negations(self):
        return self._negations

    @property
    def conditions(self):
        return self._conditions

    @property
    def numbers(self):
        return self._numbers

    @property
    def order_references(self):
        return self._order_references

    @staticmethod
    def _negated_modality(text):
        """“不能退”“没法承诺”“不给退”这类否定情态，即使中间隔着数量词。"""
        return bool(re.search(
            r'(无法|没法|没办法|不能|不可以|不给|不会)[^，。；！？]{0,4}'
            r'(退|承诺|保证|到账|打款)',
            text,
        ))

    @property
    def positive_promise(self):
        return self.promise and not self.negated_promise

    @property
    def statement(self):
        return self.promise or self.query

    @property
    def qualifier(self):
        return (
            self.conditional or self.negated_promise or self.retraction
            or bool(self.negations)
        )


def cues(text):
    return Cues(text)


def classify_text(text, speaker):
    """把一段坐席或客户原话归入专员可选择的认定类型。"""
    value = Cues(text)
    if speaker == '客户':
        return '客户转述' if value.relay else '客户发言'
    if value.positive_promise and value.conditional:
        return '附条件答复'
    if value.positive_promise and value.retraction:
        return '承诺'
    if value.negated_promise and value.query:
        return '仅代为查询'
    if value.negated_promise:
        return '否定答复'
    if value.query and not value.positive_promise:
        return '仅代为查询'
    if value.positive_promise:
        return '承诺'
    if value.retraction:
        return '改口'
    return '无法认定'


def verdict_for_category(category):
    return {
        '承诺': '承诺',
        '附条件答复': '附条件答复',
        '仅代为查询': '仅查询',
        '否定答复': '无法认定',
        '改口': '无法认定',
        '客户转述': '无法认定',
        '客户发言': '无法认定',
        '无法认定': '无法认定',
    }.get(category, '无法认定')


def semantic_signature(text, ignore_numbers=()):
    """保留会影响认定的特征；措辞、标点和断句变化不会改变它。"""
    value = Cues(text)
    ignored = {str(number).casefold() for number in ignore_numbers}
    subject = 'other' if value.other_subject else ('same' if value.same_subject else 'unknown')
    numbers = tuple(number for number in value.numbers if number.casefold() not in ignored)
    references = tuple(
        reference for reference in value.order_references
        if reference.casefold() not in ignored
    )
    return {
        'promise': value.positive_promise,
        'query': value.query,
        'conditional': value.conditional,
        'negated_promise': value.negated_promise,
        'retraction': value.retraction,
        'relay': value.relay,
        'subject': subject,
        'negations': value.negations,
        'numbers': numbers,
        'order_references': references,
    }


def semantic_differences(old_text, new_text, ignore_numbers=()):
    old = semantic_signature(old_text, ignore_numbers)
    new = semantic_signature(new_text, ignore_numbers)
    differences = []
    # 同义否定词的改换（“还没”改成“尚未”）不影响认定；有否定变成没有
    # 否定、或否定落到了承诺上，才需要重新审核。
    if bool(old['negations']) != bool(new['negations']):
        differences.append('否定词变化')
    if old['subject'] != new['subject']:
        differences.append('退款对象指代变化')
    if old['numbers'] != new['numbers'] or old['order_references'] != new['order_references']:
        differences.append('订单编号或金额变化')
    # 条件的写法可以改（“前提是…”写成“需要先…”），但只要条件还在、效力未变，
    # 就不必让专员重复审核；条件的效力变化通常同时改变否定词。
    if (old['promise'], old['query'], old['conditional'], old['negated_promise']) != (
            new['promise'], new['query'], new['conditional'], new['negated_promise']):
        differences.append('承诺或查询性质变化')
    if old['retraction'] != new['retraction']:
        differences.append('改口表述变化')
    if old['relay'] != new['relay']:
        differences.append('转述关系变化')
    return differences
