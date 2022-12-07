# EXPANSION model for Type Ia Supernova Light Curve Fitting version 2022

<!---Esses são exemplos. Veja https://shields.io para outras pessoas ou para personalizar este conjunto de escudos. Você pode querer incluir dependências, status do projeto e informações de licença aqui--->

![GitHub repo size](https://img.shields.io/github/repo-size/CassiaNascimento/EXP_model_for_SN_Ia_LC_fitting?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/CassiaNascimento/EXP_model_for_SN_Ia_LC_fitting?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/CassiaNascimento/EXP_model_for_SN_Ia_LC_fitting?style=for-the-badge)

## About
The first version of this model was presented in 2021 in the Masters Dissertation of João Paulo Correia de França, MSc, supervised by Dr. Ribamar Rondon de Rezende dos Reis, PhD, at the Federal University of Rio de Janeiro, entitled "[Investigando o ajuste de curva de luz de supernovas Ia](https://www.if.ufrj.br/)".

Here we present both the training and the testing of the EXPANSION model, this model proposes a modification in the Type Ia Supernova empirical flux description proposed in SALT2 (Spectral Adaptive Light Curve Template 2, see [Betoule 2014](https://www.aanda.org/articles/aa/abs/2014/08/aa23413-14/aa23413-14.html) for last SALT2 training).

In this repository we present the pre-processing steps applied to the SNFactory data used to create our templates. We also provide our templates, the validation fits and a light curve fitting of the Pantheon Sample (see Scolnic 2018 for more information about this sample). To conclude, we perform a cosmological analysis in order to compare our constraining power with SALT2's over the same subsample, yet without considering distance bias corrections.

## Python

**Requirements**
Partial list of requirements: [requirements](https://github.com/CassiaNascimento/EXP_model_for_SN_Ia_LC_fitting/blob/main/requirements.txt)

## Citation

If you use the current code in a publication, please cite the following paper:

[link](https://academic.oup.com/mnras)

## Special mention regarding SNEMO

The spectra files used in this work were publicly available by the [SNEMO](https://iopscience.iop.org/article/10.3847/1538-4357/aaec7e) team and thus, we give all the credits on the contributions. 