import aces2py as a2
import numpy as np
import sys

print(a2.__doc__)
a2.init()
a2.run("ints")
Nelec=10
nbasis=np.array([1])

#number of basis functions
a2.igetrecpy("NBASTOT ",nbasis, 1)
nbas=nbasis[0]
print("total number of basis function are ",nbas)

#array for overlap matrix
olp_=np.zeros(nbas*nbas)
a2.get1ehpy("overlap", olp_, nbas)
olp=np.reshape(list(olp_),(nbas,nbas))
#print("The overlap matrix is " +"\n" ,  olp)

#array for one electron integrals
oneh_=np.zeros(nbas*nbas)
a2.get1ehpy("oneh", oneh_, nbas)
oneh=np.reshape(list(oneh_),(nbas,nbas))
#print("The one electron intrgrals are " +"\n" ,  oneh)


#array for two electron integrals
twoh_=np.zeros(nbas*nbas*nbas*nbas)
a2.get2ehpy("2elints", twoh_, nbas)
twoh=np.reshape(list(twoh_),(nbas**2,nbas**2))
#print("The two electron intrgrals are " +"\n" ,  twoh)
twoh__=twoh_.reshape(nbasis[0],nbasis[0],nbasis[0],nbasis[0])
#print(twoh__)

#diagonalization of overlap matrix
e, v= np.linalg.eig(olp)
idx_=e.argsort()
e=e[idx_]
v=v[:,idx_]


#forming transformation matrix from diagonalized overlpa matrix
diag_olp=np.around(np.matmul(np.matmul(v.transpose(),olp),v),5)

#transformation matrix
for i in range(nbasis[0]):
    diag_olp[i][i]=1/(np.sqrt(diag_olp[i][i]))

trans_mat= v @ diag_olp @ v.transpose()
#print(trans_mat)


#forming two electron matrix G as given in Szabo and Ostlund

twoh__=twoh_.reshape(nbas,nbas,nbas,nbas)
B=np.reshape(np.zeros(nbas*nbas*nbas*nbas),(nbas,nbas,nbas,nbas))
C=np.reshape(np.zeros(nbas*nbas*nbas*nbas),(nbas,nbas,nbas,nbas))

for i in range(nbas):
    for j in range(nbas):
        for k in range(nbas):
            for l in range(nbas):
                B[i][j][k][l]=twoh__[i][j][l][k]
                C[i][j][k][l]=twoh__[i][k][l][j]


B=B.reshape(nbas*nbas,nbas*nbas)
C=C.reshape(nbas*nbas,nbas*nbas)
BC=np.subtract(B,C/2)

#density matrix guess
p = np.zeros((nbas,nbas), dtype=float)

#function to find distance between two matrix

def SD_succesive_matrix_elements(p_tilde,p):
    x=np.linalg.norm(np.subtract(p_tilde,p))
    return x

#function for calculation of electronic energy

def Energy(P,F,H):
    P_=np.reshape(P,(1,nbas*nbas))
    F_=np.reshape(F,(1,nbas*nbas))
    H_=np.reshape(H,(1,nbas*nbas))

    u=np.sum((np.multiply(P_,np.add(F_,H_))),axis=1, dtype=float)/2
    return(u)
    #print(u)

#initial guess for density matrix
p_previous = np.zeros((nbas,nbas), dtype=float)
np.fill_diagonal(p_previous, 1)
p_list=[]


####################################################
#
#  RHF calculations
#
####################################################


threshold=100
counter=1
while threshold > 0.0000001:
    P=p.flatten()
    G=np.reshape(np.sum(np.multiply(P,BC), axis=1,dtype=float),(nbas,nbas))
    F=np.add(oneh,G)   
    F_prime=np.matmul(trans_mat.transpose(),(np.matmul(F,trans_mat)))
    e_f,v_f=np.linalg.eig(F_prime)
    halwa=Energy(p,F,oneh)
    #sorting eigenvectors
    idx=e_f.argsort()
    e_f=e_f[idx]
    v_f=v_f[:,idx]
    C_=np.matmul(trans_mat,v_f)
    p = np.zeros((nbas,nbas), dtype=float)

    for i in range(nbas):
        for j in range(nbas):
            for a in range(int(Nelec/2)):
                p[i][j]+=2*C_[i][a]*C_[j][a]
    
    p_list.append(p)
    threshold= SD_succesive_matrix_elements(p_previous,p)
    p_previous=p.copy()
    counter += 1

print("E(RHF_electronic)=" , float(halwa), "Hartrees")

