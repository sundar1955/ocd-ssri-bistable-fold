"""Merge finalized augmentation SIs (memantine S12, AP S13, combination S14) into the master,
shifting data S12->S15 and refs S13->S16. Produces MASTER_v2 (v1 untouched). OCT3 deferred."""
M = open('Paper_A_Supplement_MASTER_v1.tex').read()
# 1. add \aAP macro (AP section needs it) after \eC in preamble
assert M.count(r'\newcommand{\eC}{e_{C}}')==1
M = M.replace(r'\newcommand{\eC}{e_{C}}',
              r'\newcommand{\eC}{e_{C}}'+'\n'+r'\newcommand{\aAP}{a_{\mathrm{AP}}}', 1)
# 2. Contents table: replace the S12-data row with the augmentation block + shifted back-matter
old_toc = ("S11 & Serotonin-syndrome thresholds and the ceiling on serotonergic augmentation\\\\\n"
           "S12 & Digitized trajectory data\\\\")
new_toc = ("S11 & Serotonin-syndrome thresholds and the ceiling on serotonergic augmentation\\\\\n"
           "S12 & Memantine and glutamatergic augmentation\\\\\n"
           "S13 & Antipsychotic augmentation: PK bridge and the Bloch comparison\\\\\n"
           "S14 & Combination therapy: complementary routes to the fold\\\\\n"
           "S15 & Digitized trajectory data\\\\\n"
           "S16 & Supplementary references\\\\")
assert old_toc in M, "TOC anchor not found"
M = M.replace(old_toc, new_toc, 1)
# 3. split at the S12-data section; renumber tail data->S15, refs->S16
marker = r'\section*{S12\quad Digitized trajectory data}'
i = M.index(marker); head, tail = M[:i], M[i:]
tail = tail.replace(r'\section*{S12\quad Digitized trajectory data}',
                    r'\section*{S15\quad Digitized trajectory data}', 1)
tail = tail.replace(r'\section*{S13\quad Supplementary references}',
                    r'\section*{S16\quad Supplementary references}', 1)
# 4. extract bodies (from first \section* to \end{document}) of each standalone
def body(fn):
    t = open(fn).read(); a = t.index(r'\section*{'); b = t.index(r'\end{document}')
    return t[a:b].rstrip()
mem, ap, combo = (body('Paper_A_SI_Memantine_v1.tex'),
                  body('Paper_A_SI_AP_v1.tex'),
                  body('Paper_A_SI_Combination_v1.tex'))
# 5. reassemble (head ends with a \clearpage before the old S12 marker)
CP = '\n\n\\clearpage\n'
out = head + mem + CP + ap + CP + combo + CP + tail
open('Paper_A_Supplement_MASTER_v2.tex','w').write(out)
print("wrote Paper_A_Supplement_MASTER_v2.tex  (%d -> %d chars)" % (len(M), len(out)))
