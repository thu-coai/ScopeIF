target_extraction_prompt = """你是一个文本处理专家，擅长抽取指令中的重要信息。我将给你提供一个用户指令，以及一个用户指令中包含的给定要求，我的目标是判断某些人工智能助手的回复是否满足了该要求。你的任务是完整提取该要求中包含的所有计数目标。你在完成该任务时必须遵循以下原则：
1. （这一点非常重要）计数目标有两种形式，所有计数目标都必须优先写成第一种形式，如果实在无法使用第一种形式表述的，才允许写成第二种形式。
    a. 如果要求直接限制某些内容的数量，则计数目标是该内容的数量（不包括具体数量限制内容）。例如对于要求"短文中出现的每个字符，都必须至少满足以下四条中的三条"，对应计数目标为"短文中出现的每个字符，以下四条中满足的条数"；对于要求"分析内容需分为四个自然段"，对应计数目标为"分析内容分为的自然段数"；对于要求"回复中禁止出现关键词：快乐、心情"，对应计数目标为"回复中出现关键词快乐的次数"，以及"回复中出现关键词心情的次数"。
    b. 如果要求不能写成上述计数形式，则计数目标应写成判别形式，即可用是/否回答的问句，对应该要求是否满足。例如对于要求"最后一行写成ANSWER: <答案>"，没有明确限制数量，则计数目标是："是否最后一行写成ANSWER: <答案>"；对于要求"回复的结尾必须是'今天是美好的一天。'"，没有明确限制数量，则计数目标是："是否回复的结尾是'今天是美好的一天。'"。
2. （这一点非常重要）计数目标的拆分必须**全面无遗漏，该要求所有需要验证的部分都必须有对应的计数目标**。计数目标和给定要求内容必须一一对应，该给定要求的每一个子项都有对应的计数目标，同时计数目标中不存在任何和该要求无关的内容。换言之，该计数目标的每一项都在要求范围内，是该给定要求满足的充分必要条件。例如，对于要求【回答需划分为三个独立部分：逻辑框架阐释、路径举措说明、落地保障建议，三个部分中第一部分和第三部分的字数相等，第二部分的字数需为第一部分字数的1.2倍到1.5倍之间。】，仅以三部分的字数作为计数目标是不正确的，因为其没有验证回复的三个部分的内容是否恰好为"逻辑框架阐释、路径举措说明、落地保障建议"。此外，计数目标的拆分必须尽量原子化，对于复杂要求尽量拆分成多个计数目标。
3. 你应该注意区分每一个计数目标是全称还是特称，"全称"指该子项作用于数量不定的多个目标，而"特称"指该子项作用于仅一个特定目标。在最终输出时你应该给出每一个子项的类型。例如：
    a. 对于要求"回复每一段均在200-250字。"，作用于回复中的每一个段落，因此其属于全称；
    b. 对于要求“回复第一段在200-250字。”，作用于回复中的第一个段落，因此其属于特称；
    c. 对于要求"所有展现你我二人核心情感互动的句子字数控制在15到22字区间内。"，作用于回复中每一个符合该条件的句子，因此其属于全称。
对于特称的计数目标，再次强调其必须仅对应一个特定目标的统计结果，不能像"四个自然段的字数"这样对应多个统计结果。上述计数目标应该拆分为"第一个自然段的字数"，"第二个自然段的字数"，"第三个自然段的字数"，"第四个自然段的字数"。
4. 对于给出多条规则，要求满足若干条的要求，计数目标应当为满足要求的条数，而非若干条要求的具体内容。因为这些要求不一定需要全部满足，不应该将其直接作为计数目标。例如对于要求"短文中出现的每个字符，都必须至少满足以下四条中的三条：XX"，对应计数目标为"短文中出现的每个字符，以下四条中满足的条数"；对于要求"满足以下五项规范中的恰好三项：XX"，对应计数目标为"以下五项规范中满足的项数"。
5. 对于要求数量并非用具体数字而是用另一个参考对象的数量表示的要求，取该要求的主语部分作为计数目标，参考值不作为计数目标。例如：
    a. 对于要求"回复中感叹句的数量多于陈述句的数量。"，仅"回复中感叹句的数量"作为计数目标，"陈述句的数量"不作为计数目标；
    b. 对于要求"全文中换行符的总个数乘以2恰好等于中文分号的总个数加上中文问号的总个数之和"，仅"全文中换行符的总个数乘以2"作为计数目标，"中文分号的总个数加上中文问号的总个数之和"不作为计数目标；
    c. 对于要求"所有以感叹号结尾的句子中，“合规”这一关键词的出现总次数，与所有以问号结尾的句子中“疏漏”这一关键词的出现总次数加1之后的数值完全相等"，仅"所有以感叹号结尾的句子中，“合规”这一关键词的出现总次数"作为计数目标，"所有以问号结尾的句子中“疏漏”这一关键词的出现总次数加1之后的数值"不作为计数目标。
6. 计数目标的语言应该和用户指令相同。即如果用户指令为英文，计数目标也应该使用英文。


输出格式如下：
```
[给定要求-开始]
...（此处直接给出需要生成验证代码的要求，必须和我提供的要求完全一致，不能有任何修改）
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": ...(详细描述该要求子项的约束目标，尽量使用要求原文),
        "目标类型": ...("全称" 或者 "特称")
    }},
    ...
]
```
[计数目标-结束]
```

# 示例一：
```
[给定要求-开始]
The 2nd sentence of the analysis of existing theme features must be no less than 30 words, the 3rd sentence explaining the first new theme must be no less than 26 words, and the last sentence of the final summary must be no less than 19 words.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "The word count of the 2nd sentence of the analysis of existing theme features",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "The word count of the 3rd sentence explaining the first new theme",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "The word count of the last sentence of the final summary",
        "目标类型": "特称"
    }}
]
```
[计数目标-结束]
```

# 示例二：
```
[给定要求-开始]
All sentences showing the core emotional interaction between you and me should have 15 to 22 words.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "The word count of each sentence showing the core emotional interaction between you and me",
        "目标类型": "全称"
    }}
]
```
[计数目标-结束]
```

# 示例三：
```
[给定要求-开始]
Exclamation marks or question-mark characters must not appear in the 2nd to 8th characters of the third sentence of the answer.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "The number of exclamation marks in the 2nd to 8th characters of the third sentence of the answer",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "The number of question-mark characters in the 2nd to 8th characters of the third sentence of the answer",
        "目标类型": "特称"
    }}
]
```
[计数目标-结束]
```

# 示例四：
```
[给定要求-开始]
The number of occurrences of "craftsmanship" in the first declarative sentence you output must be less than the number of occurrences of "skills" in the 1st sentence of the third section, and the number of occurrences of "theme" in the 1st sentence listing the new theme must be greater than the number of occurrences of "activity" in the 2nd sentence of the second section.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "The number of occurrences of \\\"craftsmanship\\\" in the first declarative sentence you output",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "The number of occurrences of \\\"theme\\\" in the 1st sentence listing the new theme",
        "目标类型": "特称"
    }}
]
```
[计数目标-结束]
```

# 示例五：
```
[给定要求-开始]
The total number of Chinese characters in the reply must be more than 3 times the combined total number of English letters, Arabic numerals, and arrow characters.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "The total number of Chinese characters in the reply",
        "目标类型": "特称"
    }}
]
```
[计数目标-结束]
```

# 示例六：
```
[给定要求-开始]
The last character of every listed legal provision item must be the Arabic numeral "0", and the second character of every listed rights-protection step item must be a Chinese character.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "The count of the Arabic numeral \\\"0\\\" in the last-character position of each listed legal provision item",
        "目标类型": "全称"
    }},
    {{
        "计数目标": "The count of Chinese characters in the second-character position of each listed rights-protection step item",
        "目标类型": "全称"
    }}
]
```
[计数目标-结束]
```

# 示例七：
```
[给定要求-开始]
Output the analysis results as a Markdown table. The table must contain three columns, with headers in order: "popular phrase content" "context analysis" "social significance"; in every cell in the second column "context analysis", write exactly 3 complete sentences.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "Whether the analysis results are output as a Markdown table",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "The number of columns in the table",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "Whether the table headers are, in order, \\\"popular phrase content\\\" \\\"context analysis\\\" \\\"social significance\\\"",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "The number of complete sentences in each cell of the second column \\\"context analysis\\\"",
        "目标类型": "全称"
    }}
]
```
[计数目标-结束]
```

# 示例八：
```
[给定要求-开始]
Present the analysis conclusions item by item as numbered entries labeled "1." "2." "3." "4." "5."; between every pair of adjacent entries, the last character of the preceding entry must be exactly the same as the first character of the following entry.
[给定要求-结束]

[计数目标-开始]
```json
[
    {{
        "计数目标": "Whether the analysis conclusions are presented item by item as numbered entries labeled \\\"1.\\\" \\\"2.\\\" \\\"3.\\\" \\\"4.\\\" \\\"5.\\\"",
        "目标类型": "特称"
    }},
    {{
        "计数目标": "Whether, between every pair of adjacent entries, the last character of the preceding entry is exactly the same as the first character of the following entry",
        "目标类型": "特称"
    }}
]
```
[计数目标-结束]
```


下面是我给你提供的用户指令以及用户指令中包含的要求：
[用户指令-开始]
{prompt}
[用户指令-结束]

[给定要求-开始]
{checklist}
[给定要求-结束]
"""
