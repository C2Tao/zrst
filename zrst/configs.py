sptransition = """\
<TRANSP> 3
 0.000000e+00 1.000000e+00 0.000000e+00
 0.000000e+00 5.000000e-01 5.000000e-01
 0.000000e+00 0.000000e+00 0.000000e+00
<ENDHMM>
"""

siltransition = """\
<TRANSP> 5
0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
0.000000e+00 6.000000e-01 4.000000e-01 0.000000e+00 0.000000e+00
0.000000e+00 0.000000e+00 6.000000e-01 4.000000e-01 0.000000e+00
0.000000e+00 0.000000e+00 0.000000e+00 7.000000e-01 3.000000e-01
0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
<ENDHMM>
"""

addsil = """\
MU 2 {sil.state[2-4].mix}
AT 2 4 0.2 {sil.transP}
AT 4 2 0.2 {sil.transP}
AT 1 3 0.3 {sp.transP}
TI silst {sil.state[3],sp.state[2]}
"""

addmix = """\
MU 3 {sil.state[2-4].mix}
"""
addmix_all = """\
MU 3 {sil.state[2-4].mix}
MU +1 {*.state[2-14].mix}
"""

macrosf = """\
~o
<STREAMINFO> 1 39
<VECSIZE> 39
<NULLD>
<MFCC_Z_E_D_A>
"""

macrosf_user = """\
~o
<STREAMINFO> 1 39
<VECSIZE> 39
<NULLD>
<USER>
"""

hcopy = """\
#Coding parameters
SOURCEFORMAT=WAV
TARGETKIND=MFCC_Z_E_D_A
TARGETRATE=100000.0
#10ms
SAVECOMPRESSED=F
SAVEWITHCRC=F
WINDOWSIZE=320000.0
#32ms
USEHAMMING=T
PREEMCOEF=0.97
NUMCHANS=26
CEPLIFTER=22
NUMCEPS=12
ENORMALIZE=T
NATURALREADORDER=TRUE
NATURALWRITEORDER=TRUE
"""

config = """\
#Coding parameters
#SOURCEFORMAT=WAV
TARGETKIND=MFCC_Z_E_D_A
TARGETRATE=100000.0
#10ms
SAVECOMPRESSED=F
SAVEWITHCRC=F
WINDOWSIZE=320000.0
#32ms
USEHAMMING=T
PREEMCOEF=0.97
NUMCHANS=26
CEPLIFTER=22
NUMCEPS=12
ENORMALIZE=T
NATURALREADORDER=TRUE
NATURALWRITEORDER=TRUE
"""

config_user = """\
#Coding parameters
TARGETKIND=USER
TARGETRATE=100000.0
#10ms
SAVECOMPRESSED=F
SAVEWITHCRC=F
WINDOWSIZE=320000.0
#32ms
USEHAMMING=T
PREEMCOEF=0.97
NUMCHANS=26
CEPLIFTER=22
NUMCEPS=12
ENORMALIZE=T
NATURALREADORDER=TRUE
NATURALWRITEORDER=TRUE
"""
