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
    print("Energy of iteration" , counter,  "is" , float(halwa))
    #sorting
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

#.........................Building CIS Matrix.............#

#.....................Integral transformation from AO basis to MO basis.......................#


dim = nbas # dimension of arrays ... e.g number of basis functions  
MO1 = np.zeros((dim,dim,dim,dim)) # For our first dumb O[N^8] method  
INT1 = twoh_.reshape(dim**2,dim**2)

INT = twoh_.reshape(dim,dim,dim,dim)
INT1t=(INT1.transpose())
#print(np.sum(INT1t,axis=1))
#print(np.sum(twoh_))
#print(INT1,"non transpose")
#for i in range(0,dim):  
#    for j in range(0,dim):  
#        for k in range(0,dim):  
#            for l in range(0,dim):  
#                for m in range(0,dim):  
#                    for n in range(0,dim):  
#                        for o in range(0,dim):  
#                            for p in range(0,dim):  
#                                MO1[i,j,k,l] += C_[m,i]*C_[n,j]*C_[o,k]*C_[p,l]*INT[m,n,o,p]
#                               

temp = np.zeros((dim,dim,dim,dim))  
temp2 = np.zeros((dim,dim,dim,dim))  
temp3= np.zeros((dim,dim,dim,dim))  
for i in range(0,dim):  
    for m in range(0,dim):  
        temp[i,:,:,:] += C[m,i]*INT[m,:,:,:]  
    for j in range(0,dim):  
        for n in range(0,dim):  
            temp2[i,j,:,:] += C[n,j]*temp[i,n,:,:]  
        for k in range(0,dim):  
            for o in range(0,dim):  
                temp3[i,j,k,:] += C[o,k]*temp2[i,j,o,:]  
            for l in range(0,dim):  
                for p in range(0,dim):  
                    MO1[i,j,k,l] += C[p,l]*temp3[i,j,k,p]  


####################################################
#
#  CONVERT SPATIAL TO SPIN ORBITAL MO
#
####################################################

# This makes the spin basis double bar integral
#two electron integrals from spatial MO basis to spin basis


#dim=dim*2
spinints=np.zeros((dim*2,dim*2,dim*2,dim*2))
for p in range(1,dim*2+1):
  for q in range(1,dim*2+1):
    for r in range(1,dim*2+1):
      for s in range(1,dim*2+1):
        value1 = MO1[(p+1)//2-1,(r+1)//2-1,(q+1)//2-1,(s+1)//2-1] * (p%2 == r%2) * (q%2 == s%2)
        value2 = MO1[(p+1)//2-1,(s+1)//2-1,(q+1)//2-1,(r+1)//2-1] * (p%2 == s%2) * (q%2 == r%2)
        spinints[p-1,q-1,r-1,s-1] = value1 - value2


#arranging eigenvalues from RHF procedure in an array
#####################################################
#
#  Spin basis fock matrix eigenvalues 
#
#####################################################


fs = np.zeros((dim*2))
for i in range(0,dim*2):
    fs[i] = e_f[i//2]
fs = np.diag(fs)
#print(fs)
#dim=nbas

NOV = Nelec*(2*dim-Nelec)

A = np.zeros((NOV,NOV))
B = np.zeros((NOV,NOV))
I = -1
for i in range(0,Nelec):
  for a in range(Nelec,dim*2):
    I = I + 1
    J = -1
    for j in range(0,Nelec):
      for b in range(Nelec,dim*2):
        J = J+1
        A[I,J] = (fs[a,a] - fs[i,i]) * ( i == j ) * (a == b) + spinints[a,j,i,b]
        B[I,J] =  spinints[a,b,i,j]

#print(A)
# Solve CIS matrix equation
ECIS,CCIS = np.linalg.eig(A)
print('The excitation energis from CIS are', ECIS)
print("E(CIS) = ", np.amax(ECIS), "Hartrees")

