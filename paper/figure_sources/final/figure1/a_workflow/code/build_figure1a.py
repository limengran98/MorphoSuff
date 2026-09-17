"""Verify the versioned PPT/PDF pair and rebuild disposable panel exports."""
from pathlib import Path
import hashlib
import json
import shutil
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]

def build():
    manifest=json.loads((ROOT/'source/export_manifest.json').read_text(encoding='utf-8-sig'))
    pptx=ROOT/'source'/manifest['source_pptx']
    accepted_pdf=ROOT/manifest['exported_pdf']
    for path, key in ((pptx,'source_pptx_sha256'),(accepted_pdf,'exported_pdf_sha256')):
        if not path.is_file():
            raise SystemExit(f'Missing versioned workflow input: {path.relative_to(ROOT)}')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=manifest[key]:
            raise SystemExit(f'Authoritative source/export changed: {path.name}. Re-export this PPT with source/export_to_pdf.ps1; never run an older drawing generator.')
    pdf=ROOT/'figure/Figure1-a_workflow.pdf'
    pdf.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(accepted_pdf,pdf)
    doc=fitz.open(pdf)
    assert len(doc)==1
    page=doc[0]
    page.get_pixmap(dpi=600,alpha=False).save(pdf.with_suffix('.png'))
    pdf.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=False),encoding='utf-8')
    doc.close()
    print(f'Verified authoritative PPT/PDF and rendered: {pdf.name}')
    return pdf

if __name__=='__main__':
    build()
