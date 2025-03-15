#!/usr/bin/env python
# coding: utf-8

# In[449]:





# In[496]:


#lp file should be in this format, not that 0's and 1's must be written clearly as coefficients in order to make the code run properly. Some example lp file can be seen in the zip.
#Lp file is called two times 'datas = parse_lp_file('example.lp')', please do the changes in both of them
import numpy as np
import re
## parse_lp_file is a function to read the lp file properly.objective, constraints, and bounds are seperated in this function
def parse_lp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    d = 0
    objective = []
    bounds = []
    constraints = []
    variables = []
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("\\"):  # Ignore empty lines and comments
            continue
        if line.lower().startswith("subject to"):
            section = 'constraints'
            continue
        if line.lower().startswith("minimize"):
            section = 'objective'
            d = 1
            continue
        if line.lower().startswith("maximize") :
            section = 'objective'
            continue
        if line.lower().startswith("bounds"):
            section = 'bounds'
            continue
        elif line.lower().startswith("binary") or line.lower().startswith("general") or line.lower().startswith("semi"):
            section = None
        if section == 'constraints':
            constraints.append(line)
        if section == 'objective':
            objective.append(line)
        if section == 'bounds':
            bounds.append(line)
   
    return constraints,' '.join(objective),bounds,variables, d
    
datas = parse_lp_file('example.lp')

constraints = datas[0]
objective = datas[1]
bounds = datas[2]
d = datas[4]

##extracts the coefficients of the variables at constraints
def extract_coefficients(constraints):
    coefficients = []
    for constraint in constraints:
        # Remove the constraint label (e.g., "c1:")
        constraint = re.sub(r'^\w+:\s*', '', constraint)

        # Find all variable-coefficient pairs using a regular expression
        coeffs = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z]\w*)', constraint)

        # Clean and store coefficients
        clean_coeffs = [(float(c.replace(' ', '')) if c.strip() else 1.0 if '+' in c else -1.0, var) for c, var in
                        coeffs]
        coefficients.append(clean_coeffs)

    return coefficients

coefficients = extract_coefficients(constraints)
print(coefficients)

## extracts the objective coefficients
def extract_obj_coefficients(objective):
    # Find all variable-coefficient pairs using a regular expression
    coeffs = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z]\w*)', objective)

    # Clean and store coefficients
    clean_coeffs = [(float(c.replace(' ', '')) if c.strip() else 1.0 if '+' in c else -1.0, var) for c, var in coeffs]

    return clean_coeffs
objcoefficients = extract_obj_coefficients(objective)


print("1",extract_obj_coefficients(objective))

##gets the c matrix
def c_matrix(objcoefficients):
    num_vars = len(objcoefficients)
    A_matrix = np.zeros((1, num_vars), dtype=float)

    for i, (coeff, _) in enumerate(objcoefficients):
        A_matrix[0, i] = coeff
        
    A_matrix = np.delete(A_matrix, 0, axis=1)

    return A_matrix


c_matrix = c_matrix(objcoefficients)


## gets the rhs 
def extract_rhs(constraints):
    rhs_values = []
    for constraint in constraints:
        # Remove the constraint label (e.g., "c1:")
        constraint = re.sub(r'^\w+:\s*', '', constraint)

        # Find the right-hand side (RHS) value using a regular expression
        rhs = re.findall(r'([<>=]+)\s*(\S+)', constraint)[-1][1]  # Get the last match in case there are multiple
        rhs_values.append(float(rhs))
    b_matrix = np.zeros((len(rhs_values),1))   
    for i in range(len(rhs_values)):
        b_matrix[i][0] = rhs_values[i]
    return b_matrix
print(extract_rhs(constraints))


def a_matrix(coefficients):
    A_matrix = np.zeros((len(coefficients),len(coefficients[0])))
    for i in range(len(coefficients)): 
        for j in range(len(coefficients[i])):
            A_matrix[i][j] = coefficients[i][j][0]
    return A_matrix
  
## gets the signs
def extract_signs(constraints):
    signs = []
    for constraint in constraints:
        # Find inequality signs (<=, >=, <, >)
        match = re.search(r'(<=|>=|<|>|=)', constraint)
        if match:
            signs.append(match.group(0))

    return signs
    

    
datas = parse_lp_file('example.lp')

constraints = datas[0]
objective = datas[1]
bounds = datas[2]
d = datas[4]

print(constraints)
print(objective)
print(bounds)
variables = set()

# Iterate through each constraint
for constraint in coefficients:
    for coefficient, variable in constraint:
        variables.add(variable)
variables = set(list(variables)[::-1])
print(variables)
bounded_variables = set()
for bound in bounds:
    match = re.findall(r'([a-zA-Z]\w*)', bound)
    bounded_variables.update(match)

# Find boundless variables

boundless_variables = variables - bounded_variables

# Output the set of variables, reversed, and identify boundless ones
reversed_variables = set(list(variables)[::-1])
print("Reversed set of variables:", reversed_variables)
print(boundless_variables)



##modifies the c matrix if there is a boundless variable
def unc_to_cont(variables, boundless_variables, c):
    k = 0
    for i in variables:
        for j in boundless_variables:
            if i == j:
                c = np.hstack((c[0,0:k+1], -1*c[0,k], c[0,k+1:]))
        
        k+= 1       
    return c

## modifies A matrix if ther eis a boundless variable
def unc_to_cont_Amatrix(variables, boundless_variables, A):
    k = 0
    for i in variables:
        for j in boundless_variables:
            if i == j:
                A = np.hstack((A[:,0:k+1], -1*A[:,k:k+1], A[:,k+1:]))
        
        k+= 1    
        
    return A


# In[497]:


## full_A is a function 
def full_A(Amatrix, signs):
    
    default = Amatrix.shape[1]
    for i in range(len(signs)):
        if(signs[i] == '<='):
            slack_matrix = np.zeros((Amatrix.shape[0], 1))
            slack_matrix[i] = 1
            Amatrix = np.hstack((Amatrix, slack_matrix))
        if(signs[i] == '>='):
            slack_matrix = np.zeros((Amatrix.shape[0], 1))
            Amatrix[i][:] = Amatrix[:][i]*-1
            slack_matrix[i] = 1
            Amatrix = np.hstack((Amatrix, slack_matrix))
        if(signs[i] == '='):
            slack_matrix = np.zeros((Amatrix.shape[0], 1))
            Amatrix[i][:] = Amatrix[i][:]*-1
            slack_matrix[i] = 1
            Amatrix = np.hstack((Amatrix, slack_matrix))
            new_row = Amatrix[i][:]
            Amatrix = np.vstack((Amatrix, new_row))
            Amatrix[i][:default] = Amatrix[i][:default]*-1
            Amatrix[-1][-1] = 0
            slack_matrix2 = np.zeros((Amatrix.shape[0], 1))
            slack_matrix2[-1] = 1
            Amatrix = np.hstack((Amatrix, slack_matrix2))
            
    for j in range(Amatrix.shape[0]):
        for i in range(Amatrix.shape[0]):
            if Amatrix[i][default+j] == 1:
                Amatrix[[i, j]] = Amatrix[[j, i]]

    return Amatrix


def full_b(b):
    for i in range(len(signs)):
        if(signs[i] == '>='):
            b[i,0] = -1*(b[i,0])
        if(signs[i] == '='):
            b = np.vstack((np.transpose(np.array([b[:i+1,0]])), -1*(b[i,0]), np.transpose(np.array([b[i+1:,0]]))))
            constraints.append('1')
    b = np.vstack((np.array([[0]]), b))
    return b


# In[498]:


V = a_matrix(coefficients)
signs = extract_signs(constraints)
zero_matrix = np.zeros((1,len(full_A(a_matrix(coefficients),extract_signs(constraints)))))
c = np.hstack((c_matrix, zero_matrix))
A = full_A(V, signs)
b = extract_rhs(constraints)
b = full_b(b)


# In[499]:


V = a_matrix(coefficients)
#creating tableau
signs = extract_signs(constraints)
c = unc_to_cont(variables, boundless_variables, c)
A = unc_to_cont_Amatrix(variables, boundless_variables, A)
tableau = np.vstack((-1*c, A))
tableau = np.hstack((tableau, b))
if d == 1:
    tableau[0,:] = -1*tableau[0,:]


# In[500]:


B1 = tableau[1:, tableau.shape[1]-len(constraints)-1:-1]
cb = np.array([tableau[0, tableau.shape[1]-len(constraints)-1:-1]])
bb = np.transpose(np.array([tableau[1:, -1]]))
Ab = tableau[1:, :tableau.shape[1]-len(constraints)-1]
cnb = -1*np.array([tableau[0, :tableau.shape[1]-len(constraints)-1]])
print(B1) ## B^-1
print(cb) ## coefficients of basic variables
print(bb) ## RHS
print(Ab) ## coefficients of original variables in constraints
print(cnb) ## coefficients of non-basic variables
bbo = bb

u = 1
iterations(B1, cb, bb, Ab, cnb, tableau, bbo, A, u)



def inner_iter(B1, cb, bb, bbo, Ab, cnb, nc, tableau, B1rhs, A, u):
    mini = nc[0,0]
    g = 0
    for i in range(nc.shape[1]):
        if nc[0,i] < mini:
            mini = nc[0,i]
            g = i
    print('1.4 updating An')
    B1an = np.matmul(B1, np.array(tableau[1:, g:g+1]))
    print(B1, '*', np.array(tableau[1:, g:g+1]), ' = ', B1an)

## min ratio test
    min1 = float('inf')
    muindex = 0
    o = 0
    for i in range(len(B1an)):
        try:
            min2 = B1rhs[i, 0] / B1an[i, 0]
            print(B1rhs[i,0])
        except ZeroDivisionError:
        # Handle the case where division by zero occurs
            min2 = min1
            print(min2)
        
        if min2 >= 0 and min1 > min2 and min2!= float('inf'):
            muindex = i
            min1 = min2
            o+=1
    print('answer of the min ratio test:')
    print(min1)
    print('pivot index:')
    print(muindex)
    if o > 0:
        mu = np.vstack(((np.transpose(np.array([B1an[:muindex, 0]/(-B1an[muindex,0])])), np.array([1/B1an[muindex,0]]), np.transpose(np.array([B1an[muindex+1:, 0]/(-B1an[muindex,0])])))))
        print('Mu:')
        print(mu)

        E = np.eye(B1.shape[0], B1.shape[1])
        for i in range(B1.shape[1]):
            E[i,muindex] = mu[i,0]
        print('E:')
        print(E)
        print(' ')
        print('1.5')
        print(E, '*', B1)
        B1 = np.matmul(E, B1)
        print('current B^-1')
        print(B1)
        cby = cb[0, muindex]
        cb[0, muindex] = cnb[0,g]
        cnb[0,g] = cby

        cbB1 = np.matmul(cb,B1)
        print(Ab)
        nc = np.matmul(cbB1, Ab) - cnb
        print(nc)
        bb = np.matmul(B1, bb)
        for i in range(Ab.shape[0]):
            Ab[i,g] = A[i, len(variables)+muindex]
        
        B1rhs = np.matmul(B1,bbo)

        z = np.matmul(cb, B1rhs)

        uu = u + 1
        tableau2 = np.hstack((Ab, B1, B1rhs))
        tableau3 = np.hstack((cnb, cbB1, z))
        tableau4 = np.vstack((tableau3, tableau2))
        print(tableau4)
        iterations(B1, cb, bb, Ab, cnb, tableau, bbo, A, uu)
    else:
        print('Unbounded')
        
    
    
def iterations(B1, cb, bb, Ab, cnb, tableau, bbo, A, u):
    print('Iteration', u)
    print('current B^-1')
    print(B1)
    print('coefficients of basic variables') 
    print(cb)
    print('RHS')
    print(bb)
    print('Ab')
    print(Ab)
    print('coefficients of non-basic variables')
    print(cnb)
    
    # Matrix multiplication of basic variables' coefficients and B^-1
    cbB1 = np.matmul(cb,B1)
    print('1.1 Matrix multiplication of basic variables coefficients and B^-1')
    print(cb, '*', B1, ' = ', cbB1)
    print(' ')
    ## min non basic calculation
    nc = np.matmul(cbB1, Ab) - cnb
    print('1.2 min non basic calculation')
    print(cbB1, '*', Ab, '-', cnb, ' = ', nc)
    print(' ')
    ## current rhs calculation
    B1rhs = np.matmul(B1,bbo)
    print('1.3 current rhs calculation')
    print(B1, '*', bbo, ' = ', B1rhs)
    print(' ')
    l = 0
    for i in range(nc.shape[1]):
        print(nc[0,i])
        if nc[0, i] < 0:
            l+=1
    p = 0
    for i in range(B1rhs.shape[0]):
        if B1rhs[i, 0] < 0:
            p+=1
    
    if l > 0:
        inner_iter(B1, cb, bb, bbo, Ab, cnb, nc, tableau, B1rhs, A, u)
   
    elif (l == 0 and p > 0):
        #infeasible if there is no negative in non-basic variables and if there is a negative value in rhs
        print('Infeasible')
        print('Optimal B^-1 is:')
        print(B1)
    
    else:
        #feasible and optimal if l is no longer a positive integer 
        print('Feasible')
        z = np.matmul(cb, B1rhs)
        tableau2 = np.hstack((Ab, B1, B1rhs))
        tableau3 = np.hstack((cnb, cbB1, z))
        tableau4 = np.vstack((tableau3, tableau2))
        print('Final Tableau:')
        print(tableau4)
        print('Optimal B^-1 is:')
        print(B1)
        print('z is:')
        print(z)   
            
            
            
            
            


# In[36]:





# In[ ]:





# In[ ]:




