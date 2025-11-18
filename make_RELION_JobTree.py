#!/usr/bin/env python3
"""
make_relion_job_tree_png.py
Build a complete, top→bottom RELION job tree from the selected job’s upstream
parents (from pipeline STAR edges like Follow_Relion_Gracefully) and draw wide,
readable “cards” into one PNG (optionally SVG). Each setting from note.txt is
one line, grouped like the RELION GUI.

What’s new in this version
- Parses the relion command in note.txt (with shlex) and maps --flags → GUI tabs.
- Covers Input / Reference / CTF / Optimisation / Sampling / Masking / Helix /
  Compute / Running (+ Other flags for anything unmapped).
- TrueType font + ASCII fallbacks fix the Unicode/PIL error.

Outputs go to a subfolder:
  <current_folder>_jobtrees/
    job_tree_<FAMILY>_<jobNNN>.png
    job_tree_<FAMILY>_<jobNNN>.svg
    job_tree_<FAMILY>_<jobNNN>_lineage.json


Script by Serhat Dönmez & Lukas W. Bauer, 2025, with ChatGPT-4 as copilot.
"""



# ============================ USER INPUT (EDIT) ===============================
from pathlib import Path
import os

PROJECT_DIR = ""  # "" = launch from current folder

# ----------- NEW INTERACTIVE JOB SELECTOR (Option B) -----------
print("\n RELION JobTree Generator")
print("by Serhat Dönmez & Lukas W. Bauer 2025\n")

project_path = Path(PROJECT_DIR or os.getcwd())

# Detect families (ONLY those that actually contain jobNNN directories)
families = []
for d in project_path.iterdir():
    if not d.is_dir():
        continue
    jobs_here = [x for x in d.iterdir() if x.is_dir() and x.name.startswith("job")]
    if jobs_here:      # Only keep real RELION families
        families.append(d.name)

families = sorted(families)

print("Available job families:")
for i, fam in enumerate(families, start=1):
    print(f"  {i}. {fam}")

choice = int(input("\nSelect family by number: "))
selected_family = families[choice - 1]


# Detect jobNNN in family
jobs = sorted([
    d.name for d in (project_path / selected_family).iterdir()
    if d.is_dir() and d.name.startswith("job")
])
print(f"\nJobs in {selected_family}:")
for i, j in enumerate(jobs, start=1):
    print(f"  {i}. {j}")
choice2 = int(input("\nSelect job by number: "))
selected_jobnum = jobs[choice2 - 1]

# Final SELECTED_JOB
SELECTED_JOB = f"{selected_family}/{selected_jobnum}"
print(f"\nSelected job: {SELECTED_JOB}\n")

# ----------------------------------------------------------------

RENDER_ALL_UPSTREAM = True
STRICT_PIPELINE_ONLY = True
ALLOW_NOTE_FALLBACK = False
WRITE_SVG_TOO = False

TARGET_CANVAS_W_PX = 2400
CARD_COLS = 4
MAX_LINES_PER_CARD = 40
LINE_WRAP_CHARS = 60
# =============================================================================

import os, re, json, math, textwrap, shlex
from pathlib import Path
from collections import defaultdict, deque

def info(msg): print(msg, flush=True)

JOB_KEY_RE   = re.compile(r'^[A-Za-z0-9]+/job\d+$')
JOB_NUM_RE   = re.compile(r'job(\d+)$')
NOTE_JOB_RE  = re.compile(r'\b([A-Za-z0-9]+/job\d+)\b')

def job_key(s: str) -> str: return s.strip().rstrip("/")
def job_num(j: str) -> int:
    m = JOB_NUM_RE.search(j)
    return int(m.group(1)) if m else 10**9
def sanitize_for_file(j: str) -> str: return j.replace("/", "_")
def read_text(p: Path) -> str:
    try: return p.read_text(encoding="utf-8", errors="ignore")
    except Exception: return ""

# ------------------------------ STAR parsing ---------------------------------
def parse_star_loops(txt: str):
    need = {
        "proc": {"_rlnPipeLineProcessName", "_rlnPipeLineProcessTypeLabel", "_rlnPipeLineProcessStatusLabel"},
        "e_from": {"_rlnPipeLineEdgeFromNode", "_rlnPipeLineEdgeProcess"},
        "e_to": {"_rlnPipeLineEdgeProcess", "_rlnPipeLineEdgeToNode"},
    }
    out = {"processes": [], "e_from": [], "e_to": []}
    lines = txt.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip().lower() == "loop_":
            i += 1
            cols = []
            while i < n and lines[i].lstrip().startswith("_"):
                cols.append(lines[i].split()[0]); i += 1
            rows = []
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith("_") and not lines[i].strip().lower().startswith("data_") and lines[i].strip().lower() != "loop_":
                parts = re.split(r"\s+", lines[i].strip())
                if len(parts) >= len(cols): rows.append(parts[:len(cols)])
                i += 1
            cset = set(cols)
            map_rows = lambda: [dict(zip(cols, r)) for r in rows]
            if need["proc"].issubset(cset): out["processes"].extend(map_rows())
            elif need["e_from"].issubset(cset): out["e_from"].extend(map_rows())
            elif need["e_to"].issubset(cset): out["e_to"].extend(map_rows())
            continue
        i += 1
    return out

def load_pipeline(project: Path):
    procs = set(); e_from = []; e_to = []
    def add(txt: str):
        t = parse_star_loops(txt)
        for p in t["processes"]:
            name = job_key(p.get("_rlnPipeLineProcessName","")).rstrip("/")
            if name: procs.add(name)
        e_from.extend(t["e_from"]); e_to.extend(t["e_to"])
    ps = project / "default_pipeline.star"
    if ps.exists(): add(read_text(ps))
    for fam in sorted([d for d in project.iterdir() if d.is_dir()]):
        for jd in fam.glob("job*/"):
            for fn in ("job_pipeline.star", "default_pipeline.star"):
                f = jd / fn
                if f.exists(): add(read_text(f))
    node_to_producers = defaultdict(set)
    proc_to_input_nodes = defaultdict(set)
    for r in e_to:
        proc = job_key(r.get("_rlnPipeLineEdgeProcess",""))
        node = r.get("_rlnPipeLineEdgeToNode","").strip()
        if proc and node: node_to_producers[node].add(proc)
    for r in e_from:
        node = r.get("_rlnPipeLineEdgeFromNode","").strip()
        proc = job_key(r.get("_rlnPipeLineEdgeProcess",""))
        if proc and node: proc_to_input_nodes[proc].add(node)
    return procs, node_to_producers, proc_to_input_nodes

def build_parents(procs, node_to_producers, proc_to_input_nodes):
    parents = {p:set() for p in procs}
    for proc in procs:
        for node in proc_to_input_nodes.get(proc, ()):
            for par in node_to_producers.get(node, ()):
                if par != proc:
                    parents[proc].add(par)
    return parents

# -------------------- command-line (note.txt) → flags dict -------------------
def extract_relion_command(jobdir: Path) -> str:
    p = jobdir / "note.txt"
    if not p.exists(): return ""
    lines = read_text(p).splitlines()
    cmd = []
    capture = False
    for ln in lines:
        lo = ln.strip()
        if "with the following command" in lo.lower():
            capture = True
            continue
        if capture:
            if not lo or lo.startswith("++++"): break
            if "relion" in lo:
                cmd.append(lo)
    if not cmd:
        for ln in lines:
            if " relion_" in ln or ln.strip().startswith("relion_"):
                cmd = [ln.strip()]
                break
    s = " ".join(cmd)
    s = s.replace("`which", "").replace("`", "")
    s = re.sub(r"\bwhich\s+relion_[^\s]+\s*", "", s)
    return s.strip()

def parse_flags_from_command(cmd: str) -> dict:
    if not cmd: return {}
    try: toks = shlex.split(cmd)
    except Exception: toks = cmd.split()
    if toks and ("relion_" in toks[0]): toks = toks[1:]
    flags = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t.lstrip("-")
            if i+1 >= len(toks) or toks[i+1].startswith("--"):
                flags[key] = True; i += 1
            else:
                val = toks[i+1]
                if val.startswith("--"):
                    flags[key] = True; i += 1
                else:
                    flags[key] = val; i += 2
        else:
            i += 1
    return flags

def fmt_bool(v): return "true" if v is True else str(v)
def fmt_ang(v):  return f"{v} deg"
def fmt_px(v):   return f"{v} px"
def fmt_A(v):    return f"{v} A"

FLAG_MAP = {
    "i":("Input","Input star",str),
    "i2":("Input","Input star 2",str),
    "ref":("Reference","Reference map",str),
    "ini_high":("Input","Initial low-pass (A)",str),
    "firstitercc":("Input","firstiter_cc",str),

    "K":("Optimisation","Number of classes (K)",str),
    "tau2_fudge":("Optimisation","tau2_fudge",str),
    "flatten_solvent":("Masking","Flatten solvent",fmt_bool),
    "zero_mask":("Masking","Zero outside mask",fmt_bool),
    "pad":("Optimisation","Padding",str),
    "sym":("Optimisation","Symmetry",str),
    "norm":("Optimisation","Scale intensities",fmt_bool),
    "oversampling":("Sampling","Oversampling",str),
    "pipeline_control":("Running","pipeline_control",str),

    "healpix_order":("Sampling","Angular sampling (order)",str),
    "offset_range":("Sampling","Offset search range (px)",fmt_px),
    "offset_step":("Sampling","Offset step (px)",fmt_px),

    "ctf":("CTF","Use CTF",fmt_bool),
    "ctf_phase_flipped":("CTF","CTF phase flipped",fmt_bool),
    "ctf_intact_first_peak":("CTF","CTF intact first peak",fmt_bool),

    "solvent_mask":("Masking","Solvent mask",str),
    "auto_local_healpix_order":("Sampling","auto_local_healpix_order",str),
    "mask":("Masking","Mask",str),
    "solvent_correct_fsc":("Masking","Solvent correct FSC",fmt_bool),

    "helical_rise":("Helix","Helical rise (A)",fmt_A),
    "helical_twist":("Helix","Helical twist (deg)",fmt_ang),
    "helical_nr_asu":("Helix","Helical ASU",str),
    "helical_z_percentage":("Helix","Helical Z %",str),

    "j":("Compute","MPI/threads (j)",str),
    "gpu":("Compute","GPU",str),
    "pool":("Compute","pool",str),
    "dont_combine_weights_via_disc":("Compute","dont_combine_weights_via_disc",fmt_bool),
    "dont_check_norm":("Compute","dont_check_norm",fmt_bool),
    "scale":("Compute","scale",str),
}

GUI_ORDER = ("Built from","Input","Output","Reference","CTF","Optimisation",
             "Sampling","Masking","Helix","Compute","Running","Other flags")

def parse_note_sections(jobdir: Path, header_lines=140):
    p = jobdir / "note.txt"
    gui_buckets = {k:[] for k in GUI_ORDER}
    if p.exists():
        lines = read_text(p).splitlines()[:header_lines]
        cur = None
        for raw in lines:
            ln = raw.strip()
            if not ln: continue
            low = ln.lower()
            if low.startswith("built from"): cur="Built from"; gui_buckets[cur].append(ln); continue
            if low.startswith("input"): cur="Input"
            elif low.startswith("outputs") or low.startswith("output"): cur="Output"
            elif low.startswith("reference"): cur="Reference"
            elif low.startswith("ctf") or " use ctf" in low: cur="CTF"
            elif low.startswith("optimis") or "optimisation" in low or "optimization" in low: cur="Optimisation"
            elif low.startswith("sampling"): cur="Sampling"
            elif low.startswith("mask"): cur="Masking"
            elif low.startswith("helic"): cur="Helix"
            elif low.startswith("gpu") or "threads" in low or "mpi" in low: cur="Compute"
            elif low.startswith("running") or low.startswith("which relion"): cur="Running"
            else:
                if cur is None: cur="Other flags"
            gui_buckets[cur].append(ln)

    cmd = extract_relion_command(jobdir)
    flags = parse_flags_from_command(cmd)
    used = set()

    if "i" in flags: gui_buckets["Input"].append(f"Particles = {flags['i']}"); used.add("i")
    if "i2" in flags: gui_buckets["Input"].append(f"Particles 2 = {flags['i2']}"); used.add("i2")
    if "ini_high" in flags: gui_buckets["Input"].append(f"Initial low-pass (A) = {flags['ini_high']}"); used.add("ini_high")
    if "ref" in flags: gui_buckets["Reference"].append(f"Reference map = {flags['ref']}"); used.add("ref")

    for k,v in flags.items():
        if k in used: continue
        if k in FLAG_MAP:
            tab,pretty,fmt=FLAG_MAP[k]
            try: gui_buckets[tab].append(f"{pretty} = {fmt(v)}")
            except: gui_buckets[tab].append(f"{pretty} = {v}")
            used.add(k)

    for k,v in flags.items():
        if k in used: continue
        v = "true" if v is True else str(v)
        gui_buckets["Other flags"].append(f"{k} = {v}")

    sections = [(k, gui_buckets[k]) for k in GUI_ORDER if gui_buckets[k]]
    return sections

def note_parents(jobdir: Path, header_lines=140):
    p = jobdir / "note.txt"
    if not p.exists(): return set()
    text = "\n".join(read_text(p).splitlines()[:header_lines])
    return set(job_key(m.group(1)) for m in NOTE_JOB_RE.finditer(text))

def guarded_keep(child: str, candidates: set, valid_jobs: set):
    kept = set()
    for cand in candidates:
        if not JOB_KEY_RE.match(cand): continue
        if job_num(cand) >= job_num(child): continue
        if cand not in valid_jobs: continue
        kept.add(cand)
    return kept

# --------------------------- upstream & layering ------------------------------
def upstream_all(selected: str, parents_map: dict):
    keep=set([selected]); q=deque([selected])
    while q:
        u=q.popleft()
        for p in parents_map.get(u,()):
            if p not in keep: keep.add(p); q.append(p)
    sub={k:set(v) for k,v in parents_map.items() if k in keep}
    for k in list(sub): sub[k] = {p for p in sub[k] if p in keep}
    return keep, sub

def layers_top_to_bottom(selected: str, parents: dict):
    memo={}
    def dist(u):
        if u in memo: return memo[u]
        if not parents.get(u): memo[u]=0; return 0
        memo[u]=1+max(dist(p) for p in parents[u]); return memo[u]
    for u in parents: dist(u)
    maxd=max(memo.values()) if memo else 0
    layers=[[] for _ in range(maxd+1)]
    for j,d in memo.items(): layers[d].append(j)
    for L in layers: L.sort(key=lambda j:(job_num(j),j))
    return layers

def longest_path_chain(selected: str, layers, parents):
    pos={j:i for i,j in enumerate([x for L in layers for x in L])}
    parent_of={}
    for L in layers:
        for j in L:
            best=None
            for p in parents.get(j,()):
                if best is None or pos[p]<pos[best]: best=p
            parent_of[j]=best
    chain=[]; cur=selected; seen=set()
    while cur and cur not in seen:
        chain.append(cur); seen.add(cur); cur=parent_of.get(cur)
    chain.reverse()
    sub={j:set() for j in chain}
    for ch in chain[1:]: sub[ch].add(parent_of[ch])
    return [chain], sub

# ------------------------------- rendering -----------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image=None

FAM_COLOR = {
    "Class3D":"#5fa6ff", "Refine3D":"#ffb45f", "Select":"#b690ff",
    "MaskCreate":"#7fdc97", "JoinStar":"#6dd5d5"
}

CANDIDATE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

def find_ttf_font():
    for p in CANDIDATE_FONTS:
        if Path(p).exists(): return p
    hint=os.environ.get("RELION_TREE_FONT")
    if hint and Path(hint).exists(): return hint
    return None

def to_ascii(s: str) -> str:
    trans={"—":"-","–":"-","…":"...","×":"x","°":" deg","Å":"A","µ":"u","μ":"u","±":"+/-",
           "α":"alpha","β":"beta","γ":"gamma"}
    for k,v in trans.items(): s=s.replace(k,v)
    try: return s.encode("ascii","ignore").decode("ascii")
    except: return s

def wrap_text(s: str, width_chars: int):
    return textwrap.wrap(s, width=width_chars, break_long_words=False, break_on_hyphens=False)

def measure_card_lines(sections, parents_line, cols, wrap_chars, max_lines):
    header_lines=2
    body=[]
    for sec,arr in sections:
        body.append(f"{sec}:")
        for a in arr:
            if len(a)>200: a="..."+a[-198:]
            body.extend(wrap_text("  "+a, wrap_chars))
    if not body: body=["(no note.txt found or empty header)"]
    body=body[:max_lines]
    col_height=math.ceil(len(body)/max(1,cols))
    return header_lines + col_height

def draw_card(draw, font, x, y, w, lh, job, parents_line, sections, cols, wrap_chars, max_lines, scale, ascii_only=False):
    r=12*scale
    draw.rounded_rectangle([x,y,x+w,y+lh], radius=r, fill="white", outline="#444444", width=max(1,scale))
    fam=job.split("/")[0]
    color=FAM_COLOR.get(fam,"#c8ccd0")
    draw.rectangle([x,y,x+w,y+8*scale], fill=color)
    pad=10*scale; lineh=16*scale; title_y=y+20*scale

    title=job
    parents_txt=f"Parents: {parents_line}" if parents_line else "Parents: -"
    if ascii_only:
        title=to_ascii(title); parents_txt=to_ascii(parents_txt)

    draw.text((x+pad,title_y), title, fill="black", font=font)
    draw.text((x+pad,title_y+lineh), parents_txt, fill="#333333", font=font)

    body=[]
    for sec,arr in sections:
        body.append(f"{sec}:")
        for a in arr:
            if len(a)>200: a="..."+a[-198:]
            body.extend(wrap_text("  "+a, wrap_chars))
    if not body: body=["(no note.txt found or empty header)"]
    body=body[:max_lines]
    if ascii_only: body=[to_ascii(b) for b in body]

    if cols<1: cols=1
    col_h=math.ceil(len(body)/cols)
    col_w=(w - 2*pad - (cols-1)*pad)//cols
    idx=0
    for c in range(cols):
        bx=x+pad + c*(col_w+pad)
        by=title_y + 2*lineh
        for r_i in range(col_h):
            if idx>=len(body): break
            draw.text((bx,by+r_i*lineh), body[idx], fill="#222222", font=font)
            idx+=1

def render_png(project: Path, selected: str, layers, parents, out_png: Path):
    if Image is None:
        info("[err] Pillow not installed; cannot render PNG.")
        return

    ttf=find_ttf_font()
    ascii_only=False
    def load_font(sz):
        nonlocal ascii_only
        if ttf:
            try: return ImageFont.truetype(ttf,sz)
            except Exception: pass
        ascii_only=True
        return ImageFont.load_default()

    info("[render] Reading note.txt headers & flags…")
    sections={}
    for L in layers:
        for j in L:
            fam,jname=j.split("/")
            sections[j]=parse_note_sections(project/fam/jname, header_lines=160)

    W=TARGET_CANVAS_W_PX
    pad=20; line_h=16; card_g=10
    heights=[]
    for L in layers:
        for j in L:
            pars=", ".join(sorted(parents.get(j,()))) or "-"
            lines=measure_card_lines(sections[j], pars, CARD_COLS, LINE_WRAP_CHARS, MAX_LINES_PER_CARD)
            card_lines=2+lines
            h=8+8+card_lines*line_h+10
            heights.append(h)
    total_h=sum(h+card_g for h in heights) + pad*2 + 6*(len(layers))

    MAX_PIXELS=85_000_000
    scale=2
    def pixels(s): return (W*s)*(total_h*s)
    while pixels(scale)>MAX_PIXELS and scale>1: scale-=1
    if pixels(scale)>MAX_PIXELS:
        while pixels(scale)>MAX_PIXELS and W>1200:
            W=int(W*0.85)

    info(f"[render] canvas ~{W}px wide; scale={scale} (est. {int(pixels(scale)/1e6)} MP)")

    img=Image.new("RGB",(W*scale,int(total_h*scale)),"white")
    draw=ImageDraw.Draw(img)
    font=load_font(max(12,12*scale))

    y=pad*scale
    for L in layers:
        for j in L:
            pars=", ".join(sorted(parents.get(j,()))) or "-"
            lines=measure_card_lines(sections[j], pars, CARD_COLS, LINE_WRAP_CHARS, MAX_LINES_PER_CARD)
            card_lines=2+lines
            card_h=(8+8+card_lines*line_h+10)*scale
            draw_card(draw, font, pad*scale, y, (W-2*pad)*scale, card_h,
                      j, pars, sections[j], CARD_COLS, LINE_WRAP_CHARS,
                      MAX_LINES_PER_CARD, scale, ascii_only=ascii_only)
            y+=card_h+card_g*scale
        y+=6*scale

    img.save(out_png)
    info(f"[render] Wrote {out_png}")

def render_svg(project: Path, selected: str, layers, parents, out_svg: Path):
    W=TARGET_CANVAS_W_PX
    pad=18; line_h=16; title_h=22; gap=10
    body_lines=MAX_LINES_PER_CARD
    card_h_guess = title_h + 8 + line_h*(2 + math.ceil(body_lines/max(1,CARD_COLS))) + 10
    y=pad
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="10" font-family="Menlo, Monaco, Consolas, monospace" font-size="12">']
    def add(s): lines.append(s)
    add(f'<text x="{pad}" y="{y+16}" font-size="18" font-weight="700">Upstream of {selected}</text>')
    y+=30
    sec_cache={}
    for L in layers:
        for j in L:
            fam,jname=j.split("/")
            sec_cache[j]=parse_note_sections(project/fam/jname, header_lines=160)
    for L in layers:
        for j in L:
            fam=j.split("/")[0]
            c={"Class3D":"#5fa6ff","Refine3D":"#ffb45f","Select":"#b690ff","MaskCreate":"#7fdc97","JoinStar":"#6dd5d5"}.get(fam,"#c8ccd0")
            add(f'<rect x="{pad}" y="{y}" width="{W-2*pad}" height="{card_h_guess}" rx="10" fill="white" stroke="#444"/>')
            add(f'<rect x="{pad}" y="{y}" width="{W-2*pad}" height="8" fill="{c}"/>')
            add(f'<text x="{pad+10}" y="{y+title_h}" font-weight="700">{j}</text>')
            pars=", ".join(sorted(parents.get(j,()))) or "-"
            add(f'<text x="{pad+10}" y="{y+title_h+line_h}" fill="#333">Parents: {pars}</text>')
            body=[]
            for sec,arr in sec_cache[j]:
                body.append(f"{sec}:")
                for a in arr:
                    if len(a)>200: a="..."+a[-198:]
                    body.extend(textwrap.wrap("  "+a, width=LINE_WRAP_CHARS, break_long_words=False, break_on_hyphens=False))
            body=body[:body_lines]
            col_h=math.ceil(len(body)/max(1,CARD_COLS))
            col_w=(W-2*pad-(max(1,CARD_COLS)-1)*pad)//max(1,CARD_COLS)
            idx=0
            for c_i in range(max(1,CARD_COLS)):
                bx=pad+10 + c_i*(col_w+pad)
                by=y+title_h+2*line_h
                for r in range(col_h):
                    if idx>=len(body): break
                    add(f'<text x="{bx}" y="{by+r*line_h}" fill="#222">{body[idx]}</text>')
                    idx+=1
            y += card_h_guess + gap
        y+=6
    lines.append("</svg>")
    Path(out_svg).write_text("\n".join(lines), encoding="utf-8")
    info(f"[render] Wrote {out_svg}")

# ----------------------------------- main ------------------------------------
def main():
    project = Path(PROJECT_DIR or os.getcwd())
    selected = job_key(SELECTED_JOB)
    fam_sel = sanitize_for_file(selected)

    # NEW — output directory creation
    cwd_name = project.name
    out_dir = project / f"{cwd_name}_jobtrees"
    out_dir.mkdir(exist_ok=True)
    info(f"[0/8] Output folder: {out_dir}")

    info(f"[1/8] Project: {project}")
    info(f"[2/8] Selected job: {selected}")

    procs, node_to_prod, proc_to_in = load_pipeline(project)
    parents = build_parents(procs, node_to_prod, proc_to_in)

    added_from_note = defaultdict(list)
    if (not STRICT_PIPELINE_ONLY) and ALLOW_NOTE_FALLBACK:
        added_ct=0
        for p in list(parents):
            if parents[p]: continue
            fam,jname=p.split("/")
            cand=note_parents(project/fam/jname, header_lines=160)
            keep=guarded_keep(p, cand, procs)
            if keep:
                parents[p].update(keep)
                added_from_note[p]=sorted(keep)
                added_ct+=1
        if added_ct:
            info(f"[fix] Added parents from note.txt for {added_ct} job(s).")

    keep, sub = upstream_all(selected, parents)
    if selected not in keep:
        info("[err] Selected job not in pipeline processes; cannot build upstream.")
        return
    for ch in list(sub): sub[ch]={p for p in sub[ch] if job_num(p)<job_num(ch)}

    layers = layers_top_to_bottom(selected, sub)
    order = [j for L in layers for j in L]
    info(f"[3/8] Lineage ready: {len(order)} card(s) in {'all upstream' if RENDER_ALL_UPSTREAM else 'single path'}")
    info("       Order (top→bottom): " + " → ".join(order[:12]) + (" ..." if len(order)>12 else ""))

    if not RENDER_ALL_UPSTREAM:
        layers, sub = longest_path_chain(selected, layers, sub)
        order = layers[0]
        info(f"[3b] Reduced to single path of length {len(order)}.")

    dbg = {
        "project": str(project),
        "selected": selected,
        "order_top_to_bottom": order,
        "parents": {k:sorted(sub.get(k,())) for k in order},
        "flag_sections_from_note": True,
    }

    # output paths
    json_path = out_dir / f"job_tree_{fam_sel}_lineage.json"
    json_path.write_text(json.dumps(dbg, indent=2), encoding="utf-8")
    info(f"[4/8] Wrote {json_path}")

    png_path = out_dir / f"job_tree_{fam_sel}.png"
    render_png(project, selected, layers, sub, png_path)

    if WRITE_SVG_TOO:
        svg_path = out_dir / f"job_tree_{fam_sel}.svg"
        render_svg(project, selected, layers, sub, svg_path)

    info("[8/8] Done. If any setting still doesn’t appear, add it in FLAG_MAP.")

if __name__ == "__main__":
    main()
