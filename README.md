# The optimal three-outcome CGLMP value in the Gupta–Pienaar qplex model

**Yusuf Kaan Öğüt** · AI contributors: Claude Opus 5.5 (Anthropic) and OpenAI Codex (GPT-6.2 Astra) · November 2026

*Türkçe özet aşağıda.*

## Result

Gupta and Pienaar ([arXiv:2606.07485](https://arxiv.org/abs/2606.07485)) found qplex correlations with CGLMP value
2 + 2/√3 ≈ 3.1547 for d = 3. At the end of their Section 3 they leave open whether qplexes can go higher, and they
conjecture that they cannot. This repository contains a proof of that conjecture within their construction:

> For all outcome weights, priors and germs, the qplex correlations satisfy I₃ ≤ 2 + 2/√3.
> The bound is attained, and equality forces uniform weights on Alice's side for both settings.

Along the way:

- For uniform weights and a flat prior, a short Fourier argument with a rank-two projector certificate.
- Every unbiased qplex box lies in the macroscopically local set Q1, and max over Q1 of I₃ is exactly 2 + 2/√3.
- The maximising box lies outside the almost-quantum set Q1+AB.
- A rational biased qplex box with I₃ = 391/144 lies outside Q1, with an exact integer separating matrix.
- The general bound is computer-assisted: 858 outer boxes with rational dual certificates, plus a quadratic local
  certificate in a ball of radius 1/125 around the uniform point.

## Files

| Path | Contents |
|---|---|
| `CGLMP-qplex-paper.pdf` | The paper (10 pages) |
| `Qplex-CGLMP-herkes-icin-ozet.pdf` | Four-page general-audience summary in Turkish |
| `paper/` | TeX sources of the paper (`main.tex`) and of the summary (`herkes_icin.tex`) |
| `certificate/` | Fixed certificates (`biased.json`, `boxes.json`, `local.json`) and the verifier |
| `independent-checks/` | A second, separately written checker for Proposition 10 and Theorem 11 |
| `discovery/` | The original numerical search programs and their data (not needed for verification) |

## Verifying the proof

Python 3.9 or newer, standard library only. No solver, network access or model access is used.

```bash
python3 certificate/verify.py
```

Expected output (about 35 seconds on a laptop):

```
Attaining model verified: I3 = 2 + 2/sqrt(3).
Biased rational model verified: I3 = 391/144; Q1 witness pairing = -1189/900.
Outer region verified: 858 certified boxes and 16 inner boxes.
Local upper bound verified at |delta| <= 1/125. The two regions cover the full domain.
```

The independent checker rebuilds the dual slack from the formula in the paper. It checks box coverage by exact
volume and disjoint interiors instead of recursion, and uses its own ℚ(√3) arithmetic (`k3.py`):

```bash
python3 -I independent-checks/hakem_onerme10.py certificate
python3 -I independent-checks/hakem_teorem11.py certificate
```

To rebuild the PDFs: `cd paper && tectonic main.tex` (or XeLaTeX + BibTeX).

## Status and caveats

- Not peer reviewed. No human expert has checked the argument yet.
- Step 1 of Theorem 11 (the reduction to Alice's weights) rests on our reading of the Gupta–Pienaar model. This is
  the step that most deserves a sceptical look.
- Claude produced the original method and the first certificate code, so its later audit is not fully independent.

## Contributions

Yusuf Kaan Öğüt posed the optimality problem, chose to include arbitrary weights, directed the work and takes
responsibility for it. Claude Opus 5.5 produced the first proofs, the computational search, the certificates and the
first draft, and later audited the final version. OpenAI Codex (GPT-6.2 Astra) reviewed the arguments, wrote a
separate checker, derived the projector form of the unbiased certificate, constructed the rational biased example
and revised the manuscript. Gemini 4 Neon reviewed the manuscript and suggested improvements.

---

## Türkçe özet

Gupta ve Pienaar, qplex modellerinin üç sonuçlu CGLMP Bell testinde 2 + 2/√3 ≈ 3,1547 puanına ulaşabildiğini
gösterdi. Bunun aşılıp aşılamayacağını açık soru olarak bıraktılar ve aşılamayacağını tahmin ettiler. Bu depo, o
tahminin kanıtını içeriyor: Ağırlıklar, başlangıç durumu ve qplex ne seçilirse seçilsin puan 2 + 2/√3'ü geçemez.
Eşitlik ancak Alice'in sonuçları eşit olasılıklıyken mümkün.

Genel durumun kanıtı bilgisayar destekli ve tamamen tam aritmetikle (kesirler ve a + b√3 sayıları) kontrol ediliyor.
Kontrol için `python3 certificate/verify.py` komutunu çalıştırman yeterli. Sadece Python'un standart kütüphanesi
gerekiyor ve yaklaşık 35 saniye sürüyor.

Herkes için yazılmış dört sayfalık Türkçe anlatım: `Qplex-CGLMP-herkes-icin-ozet.pdf`.

Not: Sonuç henüz hakemli bir dergide yayımlanmadı ve hiçbir insan uzman tarafından incelenmedi.
