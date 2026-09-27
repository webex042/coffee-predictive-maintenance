"""
dashboard/theme.py  --  the "arcade cabinet" skin for the dashboard
===================================================================
Presentation ONLY. This file changes how the dashboard LOOKS, never what it
computes. app.py still loads the same model, scores the same data and shows
the same numbers -- this module just injects CSS (and a couple of tiny HTML
helpers) so the control room reads like a retro arcade cabinet.

Why a separate file? So all the styling lives in ONE place with named
"design tokens" (colours, fonts) you can tweak without hunting through
app.py. Nothing here imports the model or touches your data.

How Streamlit theming works (the short version):
  * Streamlit builds normal HTML under the hood. We can't rename its
    elements, but we CAN attach CSS to them with st.markdown("<style>...").
  * We target Streamlit's stable `data-testid` hooks (e.g. stMetric) plus
    plain tags (h1, button), and only touch colours/fonts/borders/spacing --
    never layout rules that could stop clicking or scrolling.
  * Every animation sits behind `prefers-reduced-motion`: anyone whose OS
    asks for reduced motion gets a calm, static screen. Nothing here needs
    motion to be understood.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# DESIGN TOKENS -- the whole palette in one glance. Change a hex here and it
# updates everywhere (the CSS below reads them as var(--name)).
# ---------------------------------------------------------------------------
# Deep-navy "cabinet" body, the electric-blue perspective grid, and the neon
# accents pulled straight from the arcade reference: cyan joystick, magenta
# CRT screen, purple glow, Pac-Man yellow.
TOKENS = {
    "bg-void":   "#0a0e27",  # deepest background (the room / cabinet body)
    "bg-panel":  "#141a44",  # raised panels / cards
    "bg-panel2": "#1b2358",  # hover / inner panels
    "grid":      "#1e2a6b",  # faint perspective-grid lines
    "cyan":      "#33e1ff",  # primary accent: interactive, focus, links, OK
    "magenta":   "#ff2e97",  # danger / alert energy (the CRT screen)
    "purple":    "#7b2ff7",  # ambient glow partner
    "yellow":    "#ffd21e",  # the ONE hero highlight (Pac-Man / the score)
    "ink":       "#eaf2ff",  # main text (blue-white)
    "ink-dim":   "#8ea0d0",  # secondary text
}

# Health bands come from config (green/amber/orange/red = real meaning). On a
# neon-dark screen those muted hues look dull, so each band gets a brighter
# arcade twin FOR DISPLAY ONLY -- the meaning (which band a machine is in) is
# unchanged, this is pure colour, not logic.
NEON_BAND = {
    "Healthy":  "#3dff88",  # neon green
    "Monitor":  "#ffd21e",  # arcade yellow
    "Warning":  "#ff9838",  # neon orange
    "Critical": "#ff2e5b",  # neon red
}


def neon_colour(band_name):
    """Neon display colour for a health-band name (falls back to cyan)."""
    return NEON_BAND.get(band_name, TOKENS["cyan"])


# Google Fonts: "Press Start 2P" is THE arcade pixel face (used only for
# headings / labels / scores -- it's wide, so never for body text). "Space
# Grotesk" is a clean, slightly techy sans used for everything you actually
# read (body, tables, inputs, captions).
_FONTS = (
    '@import url("https://fonts.googleapis.com/css2?'
    'family=Press+Start+2P&family=Space+Grotesk:wght@400;500;600;700&display=swap");'
)


def _root_vars():
    """Emit the tokens as CSS custom properties so the CSS below can use them."""
    body = "".join(f"--{name}:{hex_};" for name, hex_ in TOKENS.items())
    return ":root{" + body + "}"


# ---------------------------------------------------------------------------
# CSS block 1 -- the "cabinet": deep-navy base, electric-blue perspective grid
# that fades toward the top, and a soft CRT vignette + faint scanlines. Both
# effects are drawn on ::before/::after pseudo-elements with pointer-events
# off, so they can NEVER intercept a click. They also sit BEHIND the content
# (z-index 0) so text stays crisp and the glow shows through translucent cards.
# ---------------------------------------------------------------------------
_CSS_BG = """
.stApp{
  background:
    radial-gradient(1200px 520px at 50% -8%, rgba(123,47,247,.20), transparent 60%),
    radial-gradient(900px 480px at 88% 0%, rgba(51,225,255,.10), transparent 55%),
    var(--bg-void);
}
.stApp::before{                       /* the perspective grid */
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:46px 46px;
  opacity:.35;
  -webkit-mask-image:radial-gradient(ellipse 80% 60% at 50% 0%, #000 45%, transparent 100%);
  mask-image:radial-gradient(ellipse 80% 60% at 50% 0%, #000 45%, transparent 100%);
  animation:pm-grid 26s linear infinite;
}
.stApp::after{                        /* vignette + faint scanlines */
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(ellipse at 50% 40%, transparent 58%, rgba(4,6,20,.60) 100%),
    repeating-linear-gradient(rgba(255,255,255,.028) 0 1px, transparent 1px 3px);
}
/* Lift all real content above the two background layers. */
[data-testid="stMain"], [data-testid="stHeader"], [data-testid="stSidebar"]{
  position:relative; z-index:1;
}
[data-testid="stHeader"]{ background:transparent; }
@keyframes pm-grid{ from{background-position:0 0,0 0;} to{background-position:0 46px,46px 0;} }
@media (prefers-reduced-motion: reduce){ .stApp::before{ animation:none; } }
"""

# ---------------------------------------------------------------------------
# CSS block 2 -- typography. Space Grotesk for everything you read; the pixel
# font ("Press Start 2P") is reserved for the marquee title and short labels
# via the .pm-pixel helper, so we never make you read a paragraph in pixels.
# Subheadings get a small glowing cyan tick -- an arcade "panel title" cue.
# ---------------------------------------------------------------------------
_CSS_TYPE = """
html, body, .stApp, [data-testid="stAppViewContainer"],
input, textarea, button, select{
  font-family:'Space Grotesk','Segoe UI',system-ui,sans-serif;
}
.stApp{ color:var(--ink); }
.pm-pixel{ font-family:'Press Start 2P',monospace; line-height:1.55; }

/* Markdown subheadings: readable sans, bold, with a neon panel-title tick. */
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{
  color:var(--ink); font-weight:700; letter-spacing:.02em;
  display:flex; align-items:center; gap:.55rem;
}
[data-testid="stMarkdownContainer"] h2::before,
[data-testid="stMarkdownContainer"] h3::before{
  content:""; width:.42rem; height:1.05em; border-radius:2px;
  background:var(--cyan); box-shadow:0 0 10px var(--cyan);
}
a, a:visited{ color:var(--cyan); text-decoration:none; }
a:hover{ text-shadow:0 0 8px var(--cyan); }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{ color:var(--ink); }

/* Text-selection highlight: black instead of the browser's default blue. */
::selection{ background:#000; color:var(--ink); }
::-moz-selection{ background:#000; color:var(--ink); }
"""

# ---------------------------------------------------------------------------
# CSS block 3 -- the marquee (attract-screen header) and the disclaimer
# ticker. The marquee is the ONE place we spend real boldness: bordered like a
# cabinet with corner "bolts", a blinking SYSTEM ONLINE LED, and a glowing
# pixel title. Everything else on the page stays quiet by comparison.
# ---------------------------------------------------------------------------
_CSS_MARQUEE = """
.pm-marquee{
  position:relative; text-align:center; margin:.1rem 0 1.1rem;
  padding:1.4rem 1.2rem 1.5rem; border-radius:14px;
  background:linear-gradient(180deg, rgba(27,35,88,.85), rgba(10,14,39,.92));
  border:2px solid rgba(51,225,255,.55);
  box-shadow:0 0 0 4px rgba(10,14,39,.9), 0 0 34px rgba(51,225,255,.26),
             inset 0 0 42px rgba(123,47,247,.18);
}
.pm-marquee::before, .pm-marquee::after{     /* corner bolts */
  content:""; position:absolute; top:11px; width:9px; height:9px;
  border-radius:50%; background:var(--cyan); box-shadow:0 0 10px var(--cyan);
}
.pm-marquee::before{ left:13px; } .pm-marquee::after{ right:13px; }
.pm-marquee__title{
  font-family:'Press Start 2P',monospace; margin:.6rem 0 .55rem; color:#fff;
  font-size:clamp(1.02rem,3.1vw,2.05rem); line-height:1.5;
  text-shadow:0 0 10px var(--cyan), 0 0 26px rgba(123,47,247,.7);
}
.pm-marquee__sub{ color:var(--ink-dim); max-width:62ch; margin:.1rem auto 0;
  font-size:1rem; line-height:1.5; }
.pm-led{ display:inline-flex; align-items:center; gap:.5rem; color:var(--cyan);
  font-family:'Press Start 2P',monospace; font-size:.58rem; letter-spacing:.09em; }
.pm-led .pm-dot{ width:9px; height:9px; border-radius:50%; background:#3dff88;
  box-shadow:0 0 10px #3dff88; animation:pm-blink 1.6s steps(1) infinite; }
@keyframes pm-blink{ 50%{ opacity:.22; } }

.pm-ticker{
  display:flex; gap:.7rem; align-items:baseline; border-radius:10px;
  border:1px solid rgba(255,46,151,.45); border-left:4px solid var(--magenta);
  background:linear-gradient(90deg, rgba(255,46,151,.13), rgba(255,46,151,.02));
  color:var(--ink); padding:.7rem .95rem; margin:0 0 1.15rem; font-size:.92rem;
  line-height:1.45;
}
.pm-ticker b{ color:var(--magenta); font-family:'Press Start 2P',monospace;
  font-size:.6rem; letter-spacing:.03em; white-space:nowrap; }
@media (prefers-reduced-motion: reduce){ .pm-led .pm-dot{ animation:none; } }
"""

# ---------------------------------------------------------------------------
# CSS block 4 -- the SCOREBOARD (the four KPI tiles up top, drawn like an
# arcade score readout). The "average health" tile is the hero and glows
# Pac-Man yellow; an at-risk count > 0 turns its tile magenta (set in app.py).
# ---------------------------------------------------------------------------
_CSS_SCORE = """
.pm-board{ display:flex; gap:.8rem; flex-wrap:wrap; margin:.1rem 0 .3rem; }
.pm-tile{ flex:1 1 165px; border-radius:12px; padding:.9rem 1rem 1rem;
  background:linear-gradient(180deg, var(--bg-panel2), var(--bg-panel));
  border:1px solid rgba(51,225,255,.28);
  box-shadow:inset 0 0 22px rgba(51,225,255,.06), 0 6px 18px rgba(4,6,20,.5); }
.pm-tile__label{ font-family:'Press Start 2P',monospace; font-size:.5rem;
  letter-spacing:.05em; color:var(--ink-dim); margin-bottom:.6rem;
  display:flex; align-items:center; gap:.4rem; line-height:1.5; }
.pm-tile__label::before{ content:""; width:6px; height:6px; border-radius:1px;
  background:var(--cyan); box-shadow:0 0 8px var(--cyan); flex:none; }
.pm-tile__value{ font-family:'Press Start 2P',monospace; font-size:1.45rem;
  color:var(--cyan); text-shadow:0 0 12px rgba(51,225,255,.45); line-height:1.25; }
.pm-tile__value small{ font-family:'Space Grotesk',sans-serif; font-size:.8rem;
  color:var(--ink-dim); }
.pm-tile--hero{ border-color:rgba(255,210,30,.55);
  box-shadow:inset 0 0 26px rgba(255,210,30,.10), 0 6px 18px rgba(4,6,20,.5); }
.pm-tile--hero .pm-tile__value{ color:var(--yellow);
  text-shadow:0 0 14px rgba(255,210,30,.5); }
.pm-tile--hero .pm-tile__label::before{ background:var(--yellow);
  box-shadow:0 0 8px var(--yellow); }
.pm-tile--alert{ border-color:rgba(255,46,151,.5); }
.pm-tile--alert .pm-tile__value{ color:var(--magenta);
  text-shadow:0 0 13px rgba(255,46,151,.5); }
.pm-tile--alert .pm-tile__label::before{ background:var(--magenta);
  box-shadow:0 0 8px var(--magenta); }
"""

# ---------------------------------------------------------------------------
# CSS block 5 -- machine "unit modules" (the coloured cards on the line) and
# the stage headers. Each card is a dark panel with a neon left edge in its
# health colour (var(--accent), set per-card in app.py); a predicted failure
# adds a pulsing magenta DANGER led. Readable first, arcade second.
# ---------------------------------------------------------------------------
_CSS_UNIT = """
.pm-stage{ font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:.82rem; color:var(--cyan); text-align:center; letter-spacing:.02em;
  padding:.35rem .2rem .5rem; margin-bottom:.7rem;
  border-bottom:2px solid rgba(51,225,255,.3); text-shadow:0 0 8px rgba(51,225,255,.4); }
.pm-unit{ position:relative; border-radius:11px; margin-bottom:.6rem;
  padding:.7rem .8rem .75rem .95rem; overflow:hidden;
  background:linear-gradient(180deg, rgba(27,35,88,.6), rgba(10,14,39,.72));
  border:1px solid rgba(255,255,255,.06);
  border-left:4px solid var(--accent,#33e1ff);
  box-shadow:0 0 15px -5px var(--accent,#33e1ff), inset 0 0 18px rgba(4,6,20,.4); }
/* Each health band sets --accent here (in the stylesheet, so Streamlit's HTML
   sanitizer can't strip it the way it strips an inline custom property). */
.pm-unit--Healthy{ --accent:#3dff88; }
.pm-unit--Monitor{ --accent:#ffd21e; }
.pm-unit--Warning{ --accent:#ff9838; }
.pm-unit--Critical{ --accent:#ff2e5b; }
.pm-unit__top{ display:flex; justify-content:space-between; align-items:center; }
.pm-unit__id{ font-family:'Press Start 2P',monospace; font-size:.64rem;
  color:#fff; text-shadow:0 0 8px var(--accent,#33e1ff); }
.pm-unit__type{ font-size:.8rem; color:var(--ink-dim); margin-top:.2rem; }
.pm-unit__score{ font-family:'Press Start 2P',monospace; font-size:1.1rem;
  color:var(--accent,#33e1ff); margin-top:.5rem; text-shadow:0 0 10px var(--accent,#33e1ff); }
.pm-unit__score small{ font-family:'Space Grotesk',sans-serif; font-size:.66rem;
  color:var(--ink-dim); }
.pm-unit__band{ font-size:.78rem; color:var(--ink); opacity:.92; margin-top:.25rem; }
.pm-unit__led{ width:11px; height:11px; border-radius:50%; background:var(--magenta);
  box-shadow:0 0 10px var(--magenta), 0 0 18px var(--magenta); flex:none;
  animation:pm-pulse 1.1s ease-in-out infinite; }
@keyframes pm-pulse{ 0%,100%{ transform:scale(1); opacity:1; }
  50%{ transform:scale(.68); opacity:.5; } }
@media (prefers-reduced-motion: reduce){ .pm-unit__led{ animation:none; } }
"""

# ---------------------------------------------------------------------------
# CSS block 6 -- Streamlit's own chrome: the sidebar ("control panel") and the
# tab bar ("arcade menu"). We only recolour and re-font these; we never touch
# their layout/position, so every click and scroll keeps working exactly as
# Streamlit intends.
# ---------------------------------------------------------------------------
_CSS_NAV = """
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(20,26,68,.97), rgba(10,14,39,.99));
  border-right:1px solid rgba(51,225,255,.22);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{
  color:var(--cyan); font-size:1rem; text-shadow:0 0 8px rgba(51,225,255,.3);
}

[data-testid="stTabs"] [data-baseweb="tab-list"]{
  gap:.4rem; border-bottom:1px solid rgba(51,225,255,.22);
}
[data-testid="stTabs"] [data-baseweb="tab"]{
  background:rgba(27,35,88,.45); border-radius:9px 9px 0 0; padding:.55rem .85rem;
  border:1px solid transparent; border-bottom:none;
}
[data-testid="stTabs"] [data-baseweb="tab"] p{
  font-family:'Press Start 2P',monospace; font-size:.58rem; letter-spacing:.01em;
  color:var(--ink-dim); line-height:1.5;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover p{ color:var(--ink); }
[data-testid="stTabs"] [aria-selected="true"]{
  background:rgba(51,225,255,.10); border-color:rgba(51,225,255,.4);
}
[data-testid="stTabs"] [aria-selected="true"] p{
  color:var(--cyan); text-shadow:0 0 9px rgba(51,225,255,.6);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{
  background:var(--cyan); box-shadow:0 0 10px var(--cyan); height:3px;
}
"""

# ---------------------------------------------------------------------------
# CSS block 7 -- controls: buttons get the chunky "arcade key" look (a solid
# drop edge that depresses on click), the file-uploader becomes a dark
# drop-slot, and the machine picker gets a cyan-outlined field. Focus rings
# stay visible for keyboard users.
# ---------------------------------------------------------------------------
_CSS_CONTROLS = """
.stButton>button, .stDownloadButton>button, [data-testid="stFileUploader"] button{
  font-family:'Press Start 2P',monospace; font-size:.58rem; letter-spacing:.01em;
  color:var(--bg-void); background:linear-gradient(180deg,#5cebff,var(--cyan));
  border:none; border-radius:9px; padding:.6rem .95rem;
  box-shadow:0 4px 0 #1591b3, 0 0 16px rgba(51,225,255,.38);
  transition:transform .05s ease, box-shadow .12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover,
[data-testid="stFileUploader"] button:hover{
  box-shadow:0 4px 0 #1591b3, 0 0 22px rgba(51,225,255,.65);
}
.stButton>button:active, .stDownloadButton>button:active,
[data-testid="stFileUploader"] button:active{
  transform:translateY(3px); box-shadow:0 1px 0 #1591b3, 0 0 14px rgba(51,225,255,.5);
}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible,
[data-testid="stFileUploader"] button:focus-visible{ outline:2px solid #fff; outline-offset:2px; }

[data-testid="stFileUploaderDropzone"]{
  background:rgba(27,35,88,.5); border:1px dashed rgba(51,225,255,.5);
  border-radius:11px;
}
[data-testid="stSelectbox"] div[data-baseweb="select"]>div{
  background:rgba(27,35,88,.6); border:1px solid rgba(51,225,255,.35);
  border-radius:9px;
}
"""

# ---------------------------------------------------------------------------
# CSS block 8 -- data surfaces: st.metric tiles (used on the machine-detail
# page) become mini scoreboards, and Streamlit's alert / dataframe / expander
# boxes are re-skinned as dark panels so they sit on the navy screen instead
# of flashing white. The alert ICON keeps its colour, so info/ok/warn/error
# still read correctly.
# ---------------------------------------------------------------------------
_CSS_DATA = """
[data-testid="stMetric"]{
  background:linear-gradient(180deg, rgba(27,35,88,.6), rgba(10,14,39,.72));
  border:1px solid rgba(51,225,255,.25); border-radius:11px; padding:.7rem .9rem;
}
[data-testid="stMetricValue"]{ font-family:'Press Start 2P',monospace;
  font-size:1rem; color:var(--cyan); text-shadow:0 0 10px rgba(51,225,255,.4); }
[data-testid="stMetricLabel"] p{ color:var(--ink-dim); font-weight:600; }
[data-testid="stAlert"]{ background:rgba(27,35,88,.62);
  border:1px solid rgba(51,225,255,.28); border-radius:10px; }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{ color:var(--ink); }
[data-testid="stDataFrame"]{ border:1px solid rgba(51,225,255,.28);
  border-radius:10px; overflow:hidden; }
[data-testid="stExpander"]{ border:1px solid rgba(51,225,255,.25);
  border-radius:10px; background:rgba(27,35,88,.4); }
[data-testid="stExpander"] summary{ color:var(--cyan); }
[data-testid="stExpander"] summary:hover{ color:#fff; }
"""

# ---------------------------------------------------------------------------
# CSS block 9 -- odds and ends: page width/top padding, dimmed captions, a
# glowing hairline divider, and the responsive tweaks that keep the scoreboard
# and tabs tidy on a phone.
# ---------------------------------------------------------------------------
_CSS_MISC = """
[data-testid="stMainBlockContainer"]{ padding-top:2.4rem; max-width:1180px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] div{ color:var(--ink-dim); }
[data-testid="stSpinner"]{ color:var(--cyan); }
hr{ border:none; height:1px; margin:1.1rem 0;
  background:linear-gradient(90deg, transparent, rgba(51,225,255,.4), transparent); }
@media (max-width:640px){
  .pm-marquee__title{ font-size:1.02rem; }
  .pm-tile{ flex-basis:132px; }
  .pm-tile__value{ font-size:1.12rem; }
  [data-testid="stTabs"] [data-baseweb="tab"] p{ font-size:.5rem; }
}
"""


def _full_css():
    """Concatenate every CSS block into one <style> tag."""
    return ("<style>" + _FONTS + _root_vars()
            + _CSS_BG + _CSS_TYPE + _CSS_MARQUEE + _CSS_SCORE + _CSS_UNIT
            + _CSS_NAV + _CSS_CONTROLS + _CSS_DATA + _CSS_MISC + "</style>")


def inject_theme():
    """Paint the arcade skin. Call ONCE, right after st.set_page_config()."""
    st.markdown(_full_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML helpers -- small chunks of custom markup app.py drops in. Each returns
# nothing; it renders straight into the page. They only ever receive text/
# numbers app.py already computed, so they can't change any result.
# ---------------------------------------------------------------------------
def marquee(title, subtitle, status="SYSTEM ONLINE"):
    """The attract-screen header: blinking LED, glowing pixel title, subtitle."""
    st.markdown(
        f'<div class="pm-marquee">'
        f'<div class="pm-led"><span class="pm-dot"></span>{status}</div>'
        f'<h1 class="pm-marquee__title">{title}</h1>'
        f'<p class="pm-marquee__sub">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def ticker(text, label="SIMULATED DATA"):
    """The magenta warning strip that carries the SIMULATED-data disclaimer."""
    st.markdown(
        f'<div class="pm-ticker"><b>&#9642; {label}</b><span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def scoreboard(items):
    """
    Render the top KPI row as arcade score tiles.

    `items` is a list of dicts, each: {"label", "value", "unit"?, "kind"?}
    where kind is "hero" (glows yellow -- for the one headline number),
    "alert" (glows magenta -- e.g. machines at risk), or omitted (cyan).
    """
    tiles = []
    for it in items:
        cls = "pm-tile"
        if it.get("kind") == "hero":
            cls += " pm-tile--hero"
        elif it.get("kind") == "alert":
            cls += " pm-tile--alert"
        unit = f"<small> {it['unit']}</small>" if it.get("unit") else ""
        tiles.append(
            f'<div class="{cls}">'
            f'<div class="pm-tile__label">{it["label"]}</div>'
            f'<div class="pm-tile__value">{it["value"]}{unit}</div></div>'
        )
    st.markdown('<div class="pm-board">' + "".join(tiles) + "</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly styling -- one helper so every chart matches the arcade palette
# (transparent background, neon series, faint blue grid, Space Grotesk font).
# ---------------------------------------------------------------------------
CHART_COLORWAY = ["#33e1ff", "#ff2e97", "#ffd21e", "#7b2ff7", "#3dff88"]


def style_fig(fig, height=None):
    """Recolour a Plotly figure to the arcade look. Returns the same figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#eaf2ff", size=13),
        colorway=CHART_COLORWAY,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(family="Space Grotesk, sans-serif", color="#eaf2ff"),
    )
    axis = dict(gridcolor="rgba(51,225,255,.14)", zerolinecolor="rgba(51,225,255,.25)",
                linecolor="rgba(51,225,255,.30)", color="#8ea0d0")
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if height:
        fig.update_layout(height=height)
    return fig
