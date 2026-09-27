# -*- coding: utf-8 -*-
"""EPUB → 纯文本转换，仅使用标准库。"""

import io
import posixpath
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree


_BLOCK_TAGS = {
    "p", "div", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "section", "article", "blockquote",
}


class _HtmlTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def _strip_html(html: str) -> str:
    extractor = _HtmlTextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass
    lines = []
    for line in "".join(extractor.parts).splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _opf_spine_paths(zf: zipfile.ZipFile) -> list[str]:
    container_bytes = zf.read("META-INF/container.xml")
    container = ElementTree.fromstring(container_bytes)
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = container.find(".//c:rootfile", ns)
    if rootfile is None:
        raise ValueError("EPUB 缺少 container.xml 结构")
    opf_path = rootfile.get("full-path", "")
    if not opf_path:
        raise ValueError("EPUB 无法定位 opf 文件")

    opf = ElementTree.fromstring(zf.read(opf_path))
    opf_dir = posixpath.dirname(opf_path)
    manifest = {}
    for item in opf.findall(".//{http://www.idpf.org/2007/opf}manifest/{http://www.idpf.org/2007/opf}item"):
        item_id = item.get("id")
        href = item.get("href")
        if item_id and href:
            manifest[item_id] = posixpath.normpath(posixpath.join(opf_dir, href))
    spine = [
        manifest.get(ref.get("idref"))
        for ref in opf.findall(".//{http://www.idpf.org/2007/opf}spine/{http://www.idpf.org/2007/opf}itemref")
    ]
    return [path for path in spine if path]


def _html_entries(zf: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in zf.namelist()
        if name.lower().endswith((".xhtml", ".html", ".htm"))
    )


def epub_to_text(data: bytes) -> str:
    """把 EPUB 文件的字节内容转成纯文本；解析失败时抛出 ValueError。"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("文件不是有效的 EPUB/ZIP 格式") from exc

    try:
        paths = _opf_spine_paths(zf)
    except Exception:
        paths = []
    if not paths:
        paths = _html_entries(zf)

    chapters = []
    for path in paths:
        try:
            raw = zf.read(path)
        except KeyError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gb18030", errors="ignore")
            except Exception:
                continue
        cleaned = _strip_html(text).strip()
        if cleaned:
            chapters.append(cleaned)

    result = "\n\n".join(chapters).strip()
    if not result:
        raise ValueError("EPUB 中没有提取到正文内容")
    return result


__all__ = ["epub_to_text"]
