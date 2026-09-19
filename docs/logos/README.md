# cqed-scf brand assets

The mark: a molecule between two Fabry-Perot mirrors, dressed by a cavity photon mode.

| File | Use |
| --- | --- |
| `cqed-scf-logo.svg` | primary lockup, light backgrounds |
| `cqed-scf-logo-dark.svg` | primary lockup, dark backgrounds |
| `cqed-scf-logo-notagline*.svg` | lockup without the tagline, for widths under 300 px |
| `cqed-scf-icon.svg` / `-dark.svg` | icon alone (120 x 120 grid) |
| `cqed-scf-icon-512.png` | GitHub organisation / social avatar |
| `favicon-32.png` | docs favicon |
| `cqed-scf-social-preview.png` | GitHub repo social preview (Settings -> Social preview) |

## Palette

| Token | Hex | Use |
| --- | --- | --- |
| Ink | `#171340` | mirrors, wordmark, rules |
| Violet | `#6D3AF2` | `SCF`, upper polariton branch in figures |
| Teal | `#17BEBB` | molecular density, lower polariton branch |
| Gold | `#F5A524` | the photon: cavity mode, Rabi splitting, the hyphen |

Gold is reserved for the photon everywhere it appears, including in plots.

## Type

Poppins Bold (`CQED`), Poppins Medium (`SCF`), Poppins Regular (tagline).
All text is outlined to paths, so nothing needs the font installed to render.

## Clear space and minimum size

Keep clear space equal to the mirror gap (26 units on the 120-unit grid) on all sides.
Minimum icon size 16 px; minimum lockup width 180 px (use the no-tagline variant below 300 px).

## README snippet

```markdown
<p align="center">
  <img src="docs/logos/cqed-scf-logo.svg#gh-light-mode-only" width="420" alt="CQED-SCF">
  <img src="docs/logos/cqed-scf-logo-dark.svg#gh-dark-mode-only" width="420" alt="CQED-SCF">
</p>
```
