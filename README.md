# LinopScript

Running BART's parallel-imaging compressed-sensing on more easily defined image-to-signal operators.

Looks at the .txt files to see how to define an operator.
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
