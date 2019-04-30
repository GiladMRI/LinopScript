# LinopScript

Running BART's parallel-imaging compressed-sensing on more easily defined image-to-signal operators.

Look at the .txt files to see how to define an operator.
For example, nuftScriptN.txt, is a simple sensitivity maps+nuft operator, with separately defined normal operator using Topelitz embedding:\
fmac 1 0\
nufft 0 -1 -1 7 0 0 0 0\
NORMAL 000\
f 0\
dblsz 3\
fft 3\
fmac 2 0\
a 3     This is the adjoint of dblsz, i.e. cropping\
a 2     This is the adjoint of fft, i.e. ifft\
a 0

# Example
Here:
https://giladddd.github.io/LinopScript/rrsg.html

# linopScript command to test operator
Usage: linopScript [-N] [-A] [-j d] <OpScriptTxt> <StartDims> <input> [<file0> [<file> [<file2> [...]]]] <output>\

Apply linop from script:\
linopScript <OpScriptTxt> <StartDims> <input> [<file0> [<file> [<file2> [...]]]] <output>\
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
