# GitHub Pages site

This directory is the source for the project's GitHub Pages site at
<https://boxerab.github.io/glass2glass/>.

To publish: in the repo settings, set **Pages → Source = Deploy from a
branch**, pick the branch hosting `main`, and select **/docs** as the
folder. GitHub will serve `docs/index.html` at the site root within a
minute or two.

The site is a single static `index.html` (no Jekyll, no build step).
Modeled on the Aavaaz docs site (`~/src/Aavaaz/aavaaz/docs/site/`) but
re-themed for g2g (cyan accent on a near-black background, no audio
demo, content swapped to the multimedia/Rust framework story).
