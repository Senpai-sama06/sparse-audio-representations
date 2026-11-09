"""Export a single offline HTML with exact audio, figures and linked data.

WAVs are losslessly gzip-compressed and deduplicated, then embedded as base64.
Modern Chrome/Edge decompress audio on demand; no server or companion files are
needed. Plot pixels and WAV sample values are preserved exactly.
"""
import base64
import gzip
import hashlib
import html
import mimetypes
from pathlib import Path
import re
import time


def export_html(source, destination, *, title=None, notice=''):
    source, destination = Path(source), Path(destination)
    document=source.read_text(encoding='utf-8'); assets={}; root=source.parent
    def embed(match):
        attr, quote, name=match.groups()
        if name.startswith(('data:','http:','https:','#')): return match.group(0)
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            return '' if attr=='src' else 'href="#missing-resource"'
        data=target.read_bytes(); mime=mimetypes.guess_type(name)[0] or 'application/octet-stream'
        if name.lower().endswith('.wav') and attr=='src':
            key='asset_'+hashlib.sha256(data).hexdigest()
            if key not in assets: assets[key]=gzip.compress(data,mtime=0)
            return f'data-audio="{key}"'
        return f'{attr}="data:{mime};base64,{base64.b64encode(data).decode()}"'
    document=re.sub(r'\b(src|href)=("|\x27)(.*?)\2',embed,document)
    if title:
        document=re.sub(r'<title>.*?</title>',f'<title>{html.escape(title)}</title>',document,flags=re.S)
        document=re.sub(r'<h1>.*?</h1>',f'<h1>{html.escape(title)}</h1>',document,count=1,flags=re.S)
    if notice:
        document=re.sub(r'(</h1>)',r'\1<p style="background:#fff3cd;padding:1rem">'+html.escape(notice)+'</p>',document,count=1)
    document=re.sub(r'<audio\b([^>]*data-audio="([^"]+)"[^>]*)>',
                    lambda m:f'<button type="button" onclick="loadEmbeddedAudio(this,\x27{m[2]}\x27)">Load embedded audio</button><audio {m[1]}>',document)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.tmp.html')
    with temp.open('w',encoding='utf-8') as stream:
        stream.write(document)
        for key,data in assets.items():
            stream.write(f'<script type="application/octet-stream" id="{key}">{base64.b64encode(data).decode()}</script>')
        stream.write('''<script>
const embeddedUrls=new Map();
async function loadEmbeddedAudio(button,id){
 try{button.disabled=true;button.textContent='Loading…';
 if(!embeddedUrls.has(id)){
 const bytes=Uint8Array.from(atob(document.getElementById(id).textContent.trim()),c=>c.charCodeAt(0));
 const decoded=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
 embeddedUrls.set(id,URL.createObjectURL(await new Response(decoded).blob()));}
 const player=button.nextElementSibling;player.src=embeddedUrls.get(id);player.load();
 button.textContent='Audio loaded';player.play().catch(()=>{});
 }catch(e){button.disabled=false;button.textContent='Load failed: '+e.message;}
}
</script>''')
    # Windows HTTP readers and antivirus may briefly hold the current HTML
    # without delete sharing. Preserve the old complete report while retrying.
    for attempt in range(80):
        try:
            temp.replace(destination)
            break
        except PermissionError:
            if attempt == 79:raise
            time.sleep(.05)
    return dict(path=str(destination),bytes=destination.stat().st_size,unique_audio=len(assets))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');p.add_argument('--title');p.add_argument('--notice',default='')
    print(export_html(**vars(p.parse_args())))
