"""report.md 를 그림이 내장된 단일 HTML 파일로 변환한다 (외부 라이브러리 없음).

보고서에 쓰인 문법만 처리한다: 제목, 문단, 글머리/번호 목록(들여쓰기 중첩), 표, 코드 블록, 이미지, 링크,
인라인 코드, 굵게, 가로줄, <a id> 앵커. 이미지는 base64 로 넣어 파일 하나로 공유할 수 있게 한다.

사용: python docs/report/md_to_html.py [report.md] [report.html] [--pdf]
  --pdf: 같은 이름의 A4 PDF 도 만든다 (Chrome/Chromium 필요)
"""
import base64
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
/* A4 문서 레이아웃: 화면에서는 A4 폭의 종이처럼, 인쇄/PDF 에서는 A4 쪽 단위로 나뉜다 */
@page { size: A4; margin: 16mm 15mm 16mm 15mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #59636e; } }
:root { --fg:#1f2328; --muted:#59636e; --line:#c9d1d9; --soft:#f4f6f8; --accent:#0b5cad; }
* { box-sizing: border-box; }
html { background: #d9dde3; }
body { margin: 0; color: var(--fg);
       font-family: "Noto Sans KR", "Noto Sans CJK KR", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
       font-size: 10pt; line-height: 1.55; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
main { width: 210mm; min-height: 297mm; margin: 12mm auto; padding: 16mm 15mm; background: #fff; box-shadow: 0 1px 6px rgba(0,0,0,.25); }
h1 { font-size: 18pt; line-height: 1.3; border-bottom: 2px solid var(--fg); padding-bottom: 3mm; margin: 0 0 4mm; }
h2 { font-size: 14pt; border-bottom: 1px solid var(--line); padding-bottom: 1.5mm; margin: 9mm 0 3mm; }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; }
h4 { font-size: 10.5pt; margin: 4mm 0 1.5mm; }
h2, h3, h4, p:has(> strong:first-child) { break-after: avoid; }
p, ul, ol, pre { margin: 1.6mm 0; }
ul, ol { padding-left: 5.5mm; }
li { margin: .4mm 0; }
li > ul, li > ol { margin: .4mm 0; }
a { color: var(--accent); text-decoration: none; }
code { font-family: "D2Coding", "Consolas", "Noto Sans Mono CJK KR", monospace; font-size: .88em;
       background: var(--soft); padding: 0 .3em; border-radius: 3px; }
pre { background: var(--soft); border: 1px solid var(--line); border-radius: 3px; padding: 2.5mm 3mm;
      white-space: pre-wrap; word-break: break-all; line-height: 1.35; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; }
.tbl { margin: 2mm 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; line-height: 1.35; }
th, td { border: 1px solid var(--line); padding: 1mm 1.6mm; vertical-align: top; word-break: keep-all; overflow-wrap: break-word; }
th { background: var(--soft); font-weight: 600; text-align: left; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
td code, th code { font-size: .92em; }
figure { margin: 2.5mm 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 215mm; height: auto; border: 1px solid #e3e7eb; }
figure.formula { text-align: left; margin: 1.5mm 0 1.5mm 3mm; }
figure.formula img { border: none; height: 9mm; max-width: 100%; object-fit: contain; object-position: left; }
figcaption { color: var(--muted); font-size: 8pt; margin-top: .8mm; }
hr { display: none; }
nav.toc { background: var(--soft); border: 1px solid var(--line); border-radius: 3px; padding: 1mm 5mm; }
nav.toc ul { margin: .6mm 0; }
@media print {
  html { background: none; }
  main { width: auto; min-height: 0; margin: 0; padding: 0; box-shadow: none; }
  h2 { break-before: page; margin-top: 0; }
  h2.nobreak { break-before: auto; margin-top: 6mm; }
}
"""


def inline(s, base):
    """인라인 마크다운: 코드, 링크, 굵게. 코드는 자리표시자로 빼 두었다가 마지막에 되돌린다(내부는 변환하지 않음)."""
    codes = []

    def keep(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    t = re.sub(r"`([^`]+)`", keep, s)
    t = html.escape(t, quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], t)


def image(alt, src, base):
    path = os.path.join(base, src)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    cls = ' class="formula"' if os.path.basename(src).startswith("f_") else ""
    cap = "" if cls else f"<figcaption>{html.escape(alt)}</figcaption>"
    return f'<figure{cls}><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">{cap}</figure>'


def list_block(lines, base):
    """들여쓰기로 중첩된 글머리/번호 목록을 HTML 로."""
    items = []  # (indent, ordered, text)
    for ln in lines:
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            items.append([len(m.group(1)), m.group(2)[0].isdigit(), m.group(3)])
        elif items:
            items[-1][2] += " " + ln.strip()
    out, stack = [], []  # stack: (indent, tag)
    for ind, ordered, text in items:
        tag = "ol" if ordered else "ul"
        while stack and ind < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or ind > stack[-1][0]:
            out.append(f"<{tag}>")
            stack.append((ind, tag))
        else:
            out.append("</li>")
        out.append(f"<li>{inline(text, base)}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def table_block(lines, base):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    h = "".join(f"<th>{inline(c, base)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c, base)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def convert(md, base):
    lines = md.splitlines()
    out, i, title = [], 0, "보고서"
    in_toc = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            lang = ln[3:].strip()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1:j]))
            out.append(f'<pre><code class="lang-{lang or "text"}">{code}</code></pre>')
            i = j + 1
            continue
        if not ln.strip():
            i += 1
            continue
        if re.match(r"^<a id=\"[^\"]+\"></a>$", ln.strip()):
            out.append(ln.strip())
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl, text = len(m.group(1)), m.group(2)
            if lvl == 1:
                title = text
            if in_toc:
                out.append("</nav>")
                in_toc = False
            cls = ' class="nobreak"' if text.strip() == "목차" else ""
            out.append(f"<h{lvl}{cls}>{inline(text, base)}</h{lvl}>")
            if text.strip() == "목차":
                out.append('<nav class="toc">')
                in_toc = True
            i += 1
            continue
        if ln.strip() == "---":
            if in_toc:
                out.append("</nav>")
                in_toc = False
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ln.strip())
        if m:
            out.append(image(m.group(1), m.group(2), base))
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(table_block(lines[i:j], base))
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            j = i
            while j < len(lines) and lines[j].strip() and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]) or lines[j].startswith("   ")):
                j += 1
            out.append(list_block(lines[i:j], base))
            i = j
            continue
        j = i
        para = []
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|```|\||!\[|\s*[-*]\s|\s*\d+\.\s|---)", lines[j]):
            para.append(lines[j].strip())
            j += 1
        if not para:  # 처리되지 않은 줄은 문단으로
            para, j = [ln.strip()], i + 1
        out.append(f"<p>{inline(' '.join(para), base)}</p>")
        i = j
    if in_toc:
        out.append("</nav>")
    body = "\n".join(out)
    return (f'<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n")


def to_pdf(html_path, pdf_path):
    """Chrome(헤드리스)로 A4 PDF 를 만든다. 쪽 크기·여백·쪽 번호는 CSS @page 를 따른다."""
    import shutil
    import subprocess
    chrome = next((c for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        print("Chrome/Chromium 이 없어 PDF 를 만들지 않았습니다 (HTML 을 브라우저에서 A4 로 인쇄하면 같은 결과)")
        return
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath(pdf_path)}", "file://" + os.path.abspath(html_path)],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print(f"{pdf_path}  ({os.path.getsize(pdf_path) / 1e6:.1f} MB)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = args[0] if args else os.path.join(HERE, "report.md")
    dst = args[1] if len(args) > 1 else os.path.splitext(src)[0] + ".html"
    with open(src, encoding="utf-8") as f:
        md = f.read()
    doc = convert(md, os.path.dirname(os.path.abspath(src)))
    with open(dst, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"{dst}  ({os.path.getsize(dst) / 1e6:.1f} MB, 이미지 {doc.count('<img ')}개 내장)")
    if "--pdf" in sys.argv:
        to_pdf(dst, os.path.splitext(dst)[0] + ".pdf")


if __name__ == "__main__":
    main()
