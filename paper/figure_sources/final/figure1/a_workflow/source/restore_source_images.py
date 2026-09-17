"""Lossless image-stream restoration after native PowerPoint PDF conversion.

PowerPoint may downsample embedded images during PDF export. Restore the exact
PNG bytes from the authoritative PPTX at the already-exported PDF transforms.
No pixel editing, contrast changes, drawing, or layout regeneration is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import posixpath
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import fitz

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def restore(pptx: Path, native_pdf: Path, output: Path):
    doc=fitz.open(native_pdf)
    assert len(doc)==1, 'Expected one panel slide'
    page=doc[0]
    pictures=[]
    with ZipFile(pptx) as z:
        rels={n.attrib['Id']:n.attrib['Target'] for n in ET.fromstring(z.read('ppt/slides/_rels/slide1.xml.rels'))}
        for pic in ET.fromstring(z.read('ppt/slides/slide1.xml')).findall('.//p:pic',NS):
            assert pic.find('.//a:srcRect',NS) is None, 'Cropped source needs explicit crop-aware conversion'
            xf=pic.find('.//a:xfrm',NS)
            assert not xf.attrib.get('rot') and not xf.attrib.get('flipH') and not xf.attrib.get('flipV')
            off=xf.find('a:off',NS); ext=xf.find('a:ext',NS)
            x,y=[float(off.attrib[k])/12700 for k in ('x','y')]
            w,h=[float(ext.attrib[k])/12700 for k in ('cx','cy')]
            rid=pic.find('.//a:blip',NS).attrib[f'{{{NS["r"]}}}embed']
            target=rels[rid]
            part=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('ppt/slides',target))
            pictures.append((fitz.Rect(x,y,x+w,y+h),part,z.read(part)))
    records=[]
    for im in page.get_images():
        xref=im[0]
        rects=page.get_image_rects(xref)
        if not rects:
            continue
        candidates=[p for p in pictures if max(abs(a-b) for a,b in zip(rects[0],p[0]))<0.025]
        assert len(candidates)==1, f'Ambiguous image at {rects[0]}: {len(candidates)}'
        _,part,blob=candidates[0]
        assert all(any(max(abs(a-b) for a,b in zip(rect,p[0]))<0.025 and p[2]==blob for p in pictures) for rect in rects)
        page.replace_image(xref,stream=blob)
        records.append({'pptx_part':part,'source_image_sha256':hashlib.sha256(blob).hexdigest(),'rects_pt':[list(r) for r in rects],'downsampled_size_px':list(im[2:4])})
    assert sum(len(r['rects_pt']) for r in records)==len(pictures)
    output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(output,garbage=4,deflate=True)
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('pptx',type=Path);p.add_argument('native_pdf',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--record',type=Path)
    args=p.parse_args()
    records=restore(args.pptx,args.native_pdf,args.output)
    result={'source_pptx_sha256':sha(args.pptx),'exported_pdf_sha256':sha(args.output),'image_restoration':records}
    if args.record:
        args.record.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'source_images_restored':len(records),'pdf':str(args.output)}))
