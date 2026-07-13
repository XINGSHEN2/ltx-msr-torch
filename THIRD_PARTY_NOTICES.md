# Third-Party Notices

## ComfyUI-derived LTX VAE and vocoder implementation

Files under `src/ltx_msr_torch/vae/` (except `__init__.py`) and
`src/ltx_msr_torch/vocoder.py` are derived from ComfyUI.

- Source: <https://github.com/comfyanonymous/ComfyUI>
- Source commit: `dd17debce517f8818ae9910b437cb1ebaa673176`
- Original source paths are recorded in each derived file.
- Modified for `ltx-msr-torch` on 2026-07-13 to remove the ComfyUI runtime
  dependency while retaining checkpoint and numerical compatibility.
- License: GNU General Public License version 3.0 (`GPL-3.0-only`).

The complete GPL text is included in the repository `LICENSE` file. Model
weights are not included and remain governed by their respective licenses,
including the LTX-2 Community License and Gemma Terms of Use.
