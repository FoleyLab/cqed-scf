import psi4 

oh = psi4.geometry ("""
0 2
O 0 0 0
H 0 0 0.9697
""")
# the OH bond-length was obtained from (https://cccbdb.nist.gov/exp2x.asp?casno=3352576&charge=0)
psi4.set_options({'basis' : '6-311+G*', 'reference': 'uhf'})
oh_en, oh_wfn = psi4.energy('scf' , molecule = oh, return_wfn = True)

print(f" the OH energy in UHF is {oh_en:.12f}")
