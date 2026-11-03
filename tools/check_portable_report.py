"""Validate offline report links and exact embedded WAV integrity without a server."""
import base64
import gzip
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class Inspector(HTMLParser):
    def __init__(self):super().__init__();self.audio=[];self.assets=set();self.external=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='audio' and 'data-audio' in attrs:self.audio.append(attrs['data-audio'])
        if tag=='script' and attrs.get('type')=='application/octet-stream':self.assets.add(attrs['id'])
        for k in ('href','src'):
            if k in attrs and not attrs[k].startswith(('data:','#')):self.external.append(attrs[k])


def check(path):
    text=Path(path).read_text(encoding='utf-8');parser=Inspector();parser.feed(text)
    assert not parser.external,parser.external
    assert set(parser.audio)<=parser.assets,'Missing audio assets'
    count=0
    for match in re.finditer(r'<script type="application/octet-stream" id="([^"]+)">([^<]+)</script>',text):
        raw=gzip.decompress(base64.b64decode(match[2]))
        assert raw[:4] in (b'RIFF',b'RF64') and raw[8:12]==b'WAVE','Not WAV audio'
        import hashlib
        assert match[1]=='asset_'+hashlib.sha256(raw).hexdigest(),'Embedded WAV digest mismatch'
        count+=1
    print(dict(file=str(path),audio_players=len(parser.audio),unique_audio=count,external_dependencies=parser.external))


if __name__=='__main__':check(sys.argv[1])
