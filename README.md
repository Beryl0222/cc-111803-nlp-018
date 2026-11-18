# 服务通话中的承诺识别与争议处置

内审专员为通话争议选择摘录、登记认定和影响理解的词句。对外发放时按原文位置遮盖账户信息。如果遮盖会改动认定所依据的内容，预览会提示重新选择摘录。发布回执保留已发文字和它的音频范围。

## 环境

需要 Python 3.11 或更新版本。运行程序仅使用 Python 标准库。测试依赖固定为 pytest 8.3.4，可执行 `python -m pip install -r requirements-dev.txt` 安装。

## 运行

使用 `PYTHONPATH=src python -m nlp_transcript_dispute_actions.review_cli preview refund --audience 内审` 查看内部材料，使用 `PYTHONPATH=src python -m nlp_transcript_dispute_actions.review_cli receipt packet-8` 查看模拟已发的外部材料。`fixtures/review_case.json` 中的 `start` 和 `end` 是发言原文中的字符位置，含左端、不含右端。音频时间使用毫秒。案例、角色和账户代号均为本地合成资料。

从项目根目录执行 `PYTHONPATH=src python -m nlp_transcript_dispute_actions.cli ingest fixtures/records.json` 导入样例，随后执行 `PYTHONPATH=src python -m nlp_transcript_dispute_actions.cli list` 查看记录。工作数据写入 `var/records.json`。业务查询使用 fixtures/business.json 中的初始资料，约定见 fixtures/business-contract.md。执行 PYTHONPATH=src python -m nlp_transcript_dispute_actions.cli business 可查看当前业务目录结果。

样例为本地合成文本。记录包含来源、修订号和原文，当前工作区对空白做统一处理，并提供字符及英文词项切分。

## 测试

执行 `python -m pytest -q` 检查判读摘录、遮盖、发放回执和资料归档。
