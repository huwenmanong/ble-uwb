"""Local PDF inventory and page-anchored draft readers; run in PyCharm.

Does not train models or upload source text. English translations remain
explicitly pending until a human/agent supplies reviewed translations.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
BUNDLED = Path(r'C:\Users\zy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages')
if BUNDLED.exists():
    sys.path.append(str(BUNDLED))
from pypdf import PdfReader

OUT = ROOT / 'SCI-UWB/data/literature_review_20261105'


def extract_pages(path):
    try:
        reader = PdfReader(path)
        texts = [page.extract_text() or '' for page in reader.pages]
        metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items()}
        engine = 'pypdf'
    except Exception:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(str(path))
        texts = []
        for page in document:
            textpage = page.get_textpage()
            texts.append(textpage.get_text_range())
            textpage.close()
            page.close()
        document.close()
        metadata = {'extractor': 'pdfium fallback; no password supplied'}
        engine = 'pdfium'
    return [text.encode('utf-8', errors='replace').decode('utf-8') for text in texts], metadata, engine


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = {}
    if (OUT / 'inventory.json').exists():
        prior = {r['id']: r for r in json.loads((OUT / 'inventory.json').read_text(encoding='utf-8'))}
    records = []
    for i, path in enumerate(sorted((ROOT / '参考论文').glob('*.pdf')), 1):
        pid = f'P{i:03d}'
        dest = OUT / pid
        dest.mkdir(exist_ok=True)
        (dest / 'assets').mkdir(exist_ok=True)
        blocks, pages, page_texts, missing = [], [], [], []
        try:
            if pid in prior and 'error' not in prior[pid] and (dest / 'source_map.json').exists():
                records.append(prior[pid])
                print(f'{pid}: existing index retained', flush=True)
                continue
            texts, metadata, engine = extract_pages(path)
            for p, text in enumerate(texts, 1):
                if len(text.strip()) < 60:
                    missing.append(p)
                page_texts.append(f'\n===== PDF PAGE {p} =====\n{text}')
                ids = []
                paragraphs = re.split(r'\n\s*\n', text.strip())
                for paragraph in paragraphs:
                    if not paragraph.strip():
                        continue
                    bid = f'S{len(blocks)+1:05d}'
                    ids.append(bid)
                    blocks.append(dict(id=bid, page=p, type='paragraph', order=len(blocks)+1,
                                       original_text=paragraph, translation=None, bbox=None,
                                       confidence='medium', refs=[], insert_after=None))
                pages.append(dict(page=p, block_ids=ids))
            all_text = '\n'.join(page_texts)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprint = hashlib.sha256(re.sub(r'\s+', '', all_text).encode('utf-8')).hexdigest()
            record = dict(id=pid, filename=path.name, path=str(path), pages=len(pages),
                          chars=len(all_text), low_text_pages=missing, sha256=digest,
                          text_fingerprint=fingerprint, metadata=metadata, extractor=engine)
            source_map = dict(paper=dict(title=path.stem, source_type='pdf', source_path=str(path),
                                        language='mixed', status='draft_untranslated'),
                              blocks=blocks, pages=pages, figures=[], glossary=[
                                  dict(term='BLE', translation='低功耗蓝牙', note='Bluetooth Low Energy'),
                                  dict(term='UWB', translation='超宽带', note='Ultra-wideband'),
                                  dict(term='RSSI', translation='接收信号强度指示', note='不是测距真值'),
                                  dict(term='PEPS', translation='无钥匙进入与启动', note='Passive Entry Passive Start')])
            (dest / 'source_map.json').write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding='utf-8')
            (dest / 'fulltext.txt').write_text(all_text, encoding='utf-8')
            md = [f'# {path.stem}', '', '**状态：全文索引/双语阅读稿草稿，尚非完成的全文译本。**',
                  f'原始文件：{path}', f'PDF 页数：{len(pages)}；低文本页：{missing}', '',
                  '|术语|统一译法|', '|---|---|', '|BLE|低功耗蓝牙|', '|UWB|超宽带|',
                  '|RSSI|接收信号强度指示|', '|PEPS|无钥匙进入与启动|', '',
                  '## 页索引', '', ' '.join(f'[p.{p}](#page-{p})' for p in range(1, len(pages)+1))]
            current_page = 0
            for block in blocks:
                if block['page'] != current_page:
                    current_page = block['page']
                    md.extend(['', f'<a id="page-{current_page}"></a>', f'## PDF p.{current_page}'])
                md.extend(['', f'<a id="{block["id"]}"></a>', f'**Source:** p.{current_page} {block["id"]}',
                           '', '**Original:**', '', block['original_text'], '',
                           '**中文:** [待逐段核译；原文为中文的段落无需重复翻译。提取文本不等于已完成学术精读。]'])
            (dest / 'paper.md').write_text('\n'.join(md), encoding='utf-8')
            (dest / 'translation_notes.md').write_text(
                '# 草稿范围与缺口\n\n已索引所有可提取页面，并保留PDF页码和块ID。\n'
                '所有英文段落尚待逐段核译；中文段落保留原文。尚未逐图裁剪、翻译图注，assets当前为空。\n'
                '多栏顺序、公式、表格与低文本页可能存在提取误差，不能仅据自动提取断言方法结论。\n'
                '后续精读结论单独记录，未精读的条目不得标为精读完成。\n'
                f'低文本页：{missing}\n', encoding='utf-8')
            records.append(record)
            print(f'{pid}: {len(pages)} pages | {path.name}', flush=True)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:160]
            records.append(dict(id=pid, filename=path.name, error=error))
            print(f'{pid}: ERROR {type(exc).__name__}', flush=True)
    (OUT / 'inventory.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['id', 'filename', 'pages', 'chars', 'low_text_pages', 'error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f'COMPLETE: {len(records)} PDFs; output: {OUT}', flush=True)


if __name__ == '__main__':
    main()
