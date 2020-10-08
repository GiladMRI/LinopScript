# LinopScript

Running BART's parallel-imaging compressed-sensing on more easily defined image-to-signal operators. Enables using BART's solvers (ADMM, ISTA, FISTA) on a variety of MR (and non-MR) recon problems. 
(For BART, see here: https://mrirecon.github.io/bart/)

# Documentation
[Manual and detailed examples here.](https://docs.google.com/presentation/d/1YmyeK1_T8uhIUAd3G5F9goTb5ghviJb-jwwZZM-SJUE/edit?usp=sharing)

[Presentation](https://docs.google.com/presentation/d/1Tp0DRTxJwQY7UIGnKhTqk1ejsNbuMuFuYASiSinyiTE/edit?usp=sharing) given at the Martinos center with detailed walk-throughs (but less updated than the manual).

# Examples
The bart-0.4.04b/matlab/ folder contains many examples (LinopScript_... .m), including: 

**EPTI:**
Subspace + B0 phase evolution
+ example of partitioning the calculation to reduce GPU memory requirements

**PEPTIDE:**
Non-cartesian + Subspace + B0 phase evolution

**SCEPTI:**
Non-cartesian + Time-segmentation + Subspace + B0 phase evolution

**T2-Shuffling:**
Subspace + spatiotemporal trick : https://giladddd.github.io/LinopScript/rrsg.html

**Time-segmentation:**
Non-cart + Time-segmentation (not accounting for T2*, Sutton/Fessler classic paper)
 
**SpiMRF:**
NonCart + Subspace + Spatiotemporal trick through Toeplitz embedding

# linopScript command to test operator
Usage: linopScript \[-N\] \[-A\] \[-j d\] \<OpScriptTxt\> \<StartDims\> \<input\> \[\<file0\> \[\<file\> \[\<file2\> \[...\]\]\]\] \<output\>\

Apply linop from script:\
linopScript \<OpScriptTxt\> \<StartDims\> \<input\> \[\<file0\> \[\<file\> \[\<file2\> \[...\]\]\]\] \<output\>\
-----------------------------------------\
Apply operator script from OpScriptTxt on the input, and save in output\
Uses other files if mentioned\
Linops:\
FFT/IFFT/FFTC/IFFTC <FFT_FLAGS>\
FMAC <Which_file_no> <SQUASH_FLAGS> : multiplies and then sums\
Transpose <dim1> <dim2> : transposes the dims\
Print <messageId> : print messageId on frwrd/adjoint/normal calls\
ident - do nothing\
Samp <Which_file_no> : Sampling is multiplication by binary map - so forward=adjoint=normal\
Part Dim1 Dim2 K\
Hankel Dim1 Dim2 K\
resize FileIdx : File contains new size\
dblsz/halfsz/dblszc/halfszc Flags : Flags for dims to double/half\
NUFFT TrajFileIdx WeightsFileIdx BasisFileIdx NUFlags ToepBool pcycleBool periodicBool lowmemBool : TrajFile should be [3 readout spokes]. Bool 0/1. NUFT defaults are false.\

-N		Apply normal\
-A		Apply adjoint\
-j fftmod_flags      	flags for fftmod_flags of (k-space?) input (if forward/normal) and (sensitivity?) file0\
-h		help\
