"""Assemble the main manuscript into one file (current section order). Unified preamble; bodies concatenated."""
import re
PRE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,float,booktabs,tabularx,array,siunitx,microtype}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\newcommand{\V}{\textcolor{blue!60!black}{\footnotesize[V]}}
\newcommand{\U}{\textcolor{red}{\footnotesize[U]}}
\newcommand{\GCS}{G_{\mathrm{CS}}}
\newcommand{\eC}{e_{C}}
\newcommand{\aAP}{a_{\mathrm{AP}}}
\newcommand{\phid}{\phi_{d_1}}
\newcommand{\phie}{\phi_{e}}
\newcommand{\eps}{\varepsilon}
\title{\vspace{-2em}\textbf{Paper A --- Full main manuscript (assembled draft v1)}\vspace{-0.5em}}
\date{}
\begin{document}\maketitle
"""
def body(fn):
    t=open(fn).read()
    m=re.search(r'\\begin\{document\}(.*)\\end\{document\}',t,re.S)
    t=m.group(1) if m else t
    t=re.sub(r'\\maketitle','',t)
    t=re.sub(r'\\title\{.*?\}\s*(\\vspace\{[^}]*\})?','',t,flags=re.S,count=1)
    t=re.sub(r'\\date\{\}','',t)
    t=re.sub(r'\\noindent\\emph\{[^{}]*(\{[^{}]*\}[^{}]*)*\}','',t,count=1)  # drop standalone note
    return t.strip()
order=[('Summary + Introduction','Paper_A_Intro_Summary_v1.tex'),
       ('Model (NOTE: relocate to Methods for Nature format)','Paper_A_Model_Section_v3.tex'),
       ('Results','Paper_A_Results_v1.tex'),
       ('Discussion','Paper_A_Discussion_v1.tex'),
       ('Methods','Paper_A_Methods_v1.tex'),
       ('References','Paper_A_References_v2.tex')]
out=[PRE]
for label,fn in order:
    out.append(f"\n\n%% ================= {label} =================\n")
    out.append(body(fn))
out.append("\n\\end{document}\n")
open('Paper_A_Main_v1.tex','w').write('\n'.join(out))
print("wrote Paper_A_Main_v1.tex")
