# Elements_of_quantum_chemistry
Electronic structure methods coded as a part of course curriculum taught by Dr. Rodney Bartlett at UF chemistry

#Content

These are the implementations of HF and post HF-methods using array of one and two electron integrals. This is a python implmentation. This code can be used generally if arrays of one and two electron integrals are provided. 

#Usage
1. Install ACESII 
2. Import the aces2py module (the python interface of ACESII)
3. Use igetrecpy module to get different informations such as nbasis, one electron integral, two electron integrals
4. Input file in ACESII is called the ZMAT file where you specify the coordinates and the details of the calculations
5. Once the calculation setup is done the following codes can be used to do RHF, CIS, MP2, TDHF. 
