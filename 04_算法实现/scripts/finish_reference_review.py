"""Attach reviewed excerpts to local draft readers and verify their bundles."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/literature_review_20261105'
MARKER = '\n\n## 本轮重点段落对照与阅读批注\n'


def main():
    annotations = json.loads((OUT / 'reviewed_annotations.json').read_text(encoding='utf-8'))
    inventory = json.loads((OUT / 'inventory.json').read_text(encoding='utf-8'))
    index = ['# 参考论文全文索引与阅读状态', '',
             '71份PDF、1722页；P004/P030为字节完全相同的重复文件。',
             '索引完成不等于精读完成；下表“重点章节分析”仅指本轮已分析的相关章节。', '',
             '|编号|文件|页数|本轮状态|', '|---|---|---:|---|']
    focus = {a['paper'] for a in annotations}
    assets = {
        'P057': [('T001', 'assets/table4_2.png', 71, 'S00201', '表4.2 均方根误差表（真实误差）'),
                 ('T002', 'assets/table4_3.png', 71, 'S00201', '表4.3 均方根误差表（单位cm）')],
        'P061': [('T001', 'assets/table5_19.png', 87, 'S00091', '表5-19 车外远距离矩形移动实验结果对比')]
    }
    for record in inventory:
        pid = record['id']
        index.append(f'|[{pid}]({pid}/paper.md)|{record["filename"]}|{record.get("pages", "?")}|'
                     f'{"重点章节分析" if pid in focus else "全文索引，未完成精读"}|')
        directory = OUT / pid
        mapping = json.loads((directory / 'source_map.json').read_text(encoding='utf-8'))
        reader = (directory / 'paper.md').read_text(encoding='utf-8').split(MARKER)[0]
        sections = []
        mapping['blocks'] = [b for b in mapping['blocks'] if not b['id'].startswith('R')]
        mapping['figures'] = []
        for number, annotation in enumerate([a for a in annotations if a['paper'] == pid], 1):
            assert any(b['id'] == annotation['source'] and b['page'] == annotation['page'] for b in mapping['blocks']), annotation
            rid = f'R{number:03d}'
            sections.extend([f'<a id="{rid}"></a>',
                             f'**Source:** p.{annotation["page"]} {annotation["source"]}；重点摘句 {rid}',
                             '', '**Original:** ' + annotation['original'], '',
                             '**中文:** ' + annotation['translation'], '',
                             '**阅读批注（非原文）:** ' + annotation['note'], ''])
            mapping['blocks'].append(dict(id=rid, page=annotation['page'], type='note',
                                          order=len(mapping['blocks'])+1,
                                          original_text=annotation['original'], translation=annotation['translation'],
                                          confidence='medium', refs=[], source_block=annotation['source'],
                                          note='Reviewed excerpt; whitespace normalized; not full-block translation.'))
        for tid, file, page, source, caption in assets.get(pid, []):
            assert (directory / file).exists()
            sections.extend([f'<a id="{tid}"></a>', f'### {caption}',
                             f'**Placed near:** p.{page} {source}', f'**Source:** p.{page} {source}',
                             f'![{caption}]({file})', f'**Original caption:** {caption}',
                             '**中文图注:** 原图注为中文，同上。',
                             '**Reading note:** 对照原页的局部表格裁剪；不替代全文阅读。', ''])
            mapping['figures'].append(dict(id=tid, type='table', page=page, caption_id=source,
                                          image_path=file, placement_hint='critical_reading_notes',
                                          placed_after=source, alt_text=caption, crop_verified=True))
        if sections:
            reader += MARKER + '\n'.join(sections)
            notes = (directory / 'translation_notes.md').read_text(encoding='utf-8')
            update = '\n本轮仅完成文末重点摘句核译/中文原文批注；正文待译段落仍未完成。关键表格见文末对应卡片，其余图表未裁剪。\n'
            if update not in notes:
                (directory / 'translation_notes.md').write_text(notes + update, encoding='utf-8')
        (directory / 'paper.md').write_text(reader, encoding='utf-8')
        (directory / 'source_map.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
        assert '**Original:**' in reader and '**中文:**' in reader
        assert len({b['id'] for b in mapping['blocks']}) == len(mapping['blocks'])
        for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)', reader):
            assert (directory / link).exists(), link
    (OUT / 'README.md').write_text('\n'.join(index), encoding='utf-8')
    print(f'VALIDATED: {len(inventory)} source maps, {len(focus)} focused reading notes, 3 table crops.')


if __name__ == '__main__':
    main()
