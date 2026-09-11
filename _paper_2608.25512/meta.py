import re, html

s = open(r'E:\Pioneer\自动电路设计研究\_paper_2608.25512\abs.html', encoding='utf-8', errors='ignore').read()


def strip(x):
    return html.unescape(re.sub(r'<[^>]+>', ' ', x)).strip()


pats = [
    ('TITLE', r'<h1 class="title mathjax">(.*?)</h1>'),
    ('AUTHORS', r'<div class="authors">(.*?)</div>'),
    ('DATELINE', r'<div class="dateline">(.*?)</div>'),
    ('ABSTRACT', r'<blockquote class="abstract mathjax">(.*?)</blockquote>'),
    ('COMMENTS', r'<td class="tablecell comments[^"]*">(.*?)</td>'),
    ('SUBJECTS', r'<td class="tablecell subjects">(.*?)</td>'),
    ('JREF', r'<td class="tablecell jref">(.*?)</td>'),
    ('DOI', r'<td class="tablecell doi">(.*?)</td>'),
    ('SUBMISSION-HISTORY', r'<div class="submission-history">(.*?)</div>'),
]
for name, p in pats:
    m = re.search(p, s, re.S)
    print('### ' + name + ':')
    print(strip(m.group(1))[:3000] if m else 'n/a')
    print()
