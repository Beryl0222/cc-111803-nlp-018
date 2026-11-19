# 服务通话中的承诺识别与争议处置

内审专员为通话争议选择摘录、登记认定和影响理解的词句。对外发放时按原文位置遮盖账户信息。如果遮盖会改动认定所依据的内容，预览会提示重新选择摘录。发布回执保留已发文字和它的音频范围。

## 环境

需要 Python 3.11 或更新版本。运行程序仅使用 Python 标准库。测试依赖固定为 pytest 8.3.4，可执行 `python -m pip install -r requirements-dev.txt` 安装。

## 运行

`fixtures/review_case.json` 中的 `start` 和 `end` 是发言原文中的字符位置，含左端、不含右端。音频时间使用毫秒。案例、角色和账户代号均为本地合成资料。`fixtures/call21_v1.json` 与 `fixtures/call21_revision.json` 演示客户插话打断、条件隔轮、账户代号被拆开复读和人工校订后的对照。

### 找候选摘录

`PYTHONPATH=src python -m nlp_transcript_dispute_actions.review_cli --case fixtures/call21_v1.json candidates`

按争议事项关键词列出带说话人、音频位置和原文的候选，并区分 `承诺`、`附条件答复`、`仅代为查询`、`客户转述`、`否定答复`。条件或否定如果落在相邻发言、隔着一轮客户插话，候选会附上 `context`、`suggested_support` 和 `requires_context`，提示一并摘录。可用 `--keyword` 增加事项关键词。

### 核对与生成材料

- `preview <认定> --audience 内审`：完整摘录和判读理由。
- `preview <认定> --audience 外部`：遮盖账户代号后的摘录；可加 `--all` 按争议事项逐项查看。
- `explain <认定>`：说明认定类别和理由。
- `receipt <回执号>`：查看已交付材料原文、音频范围和遮盖字段。

外部材料只带遮盖后的文字和 `account_id` 字段类型；争议事项名里的代号也会被替换。遮盖若会抹掉否定、条件、改口或对象指代，或材料仍能拼回代号，该项返回 `须重新选择摘录`（`waiting_for_review`），不附带任何原文；`can_switch_excerpt` 和内部命令 `redaction_options` 可核实是否存在意思完整的替代片段。其它争议事项不受影响。

### 人工校订后的对照

`PYTHONPATH=src python -m nlp_transcript_dispute_actions.review_cli revision --case fixtures/call21_revision.json`

逐项把旧摘录放到新版转写中定位。标点、断句和语气词变化仍算 `同一段发言`；发言被拆到多轮也能对应。说话者、退款对象或限制条件改变的记为 `已改变认定`，找不到的记为 `无法对应`，多处相同的记为 `多个对应`。不能对应的项目会留下 `disagreements`（具体分歧和原文），其它项目照常可办理。

`receipt <回执号> --align --revision-file <新版对照文件>` 会在保留已交付文字的同时，更新 `still_usable`、`supporting_conclusion` 和 `revision_status`，说明该回执是否仍可作为本次争议的依据。`external <回执号>` 输出外部可见清单，被挡住的项目只显示 `等待复核`。

## 测试

执行 `python -m pytest -q` 检查判读摘录、遮盖、发放回执、版本对照和资料归档。

