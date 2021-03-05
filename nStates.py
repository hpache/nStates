''' Henry Pacheco Cachon
    Created: 01/15/2021 Last Modified: 01/18/2021 
    Creating a class that sets up our NState system and
    sets up the needed matrix equation to solve for the steady state '''

import numpy as np 
from sympy import *
from sympy.physics.quantum import *
import pandas as pd 
from scipy.sparse import linalg as l


class States:

    # Initialize the number of states in our system
    # As well as the hamiltonian of our system
    # nd a dictionary containing our static parameters (parameters kept constant).
    def __init__(self, numberStates, hamiltonian,staticDictionary):
        self.N = numberStates
        self.H = Matrix(hamiltonian)
        self.static = staticDictionary
        self.rhos = [Symbol("rho_{}{}".format(i,j)) for i in range(self.N) for j in range(self.N) if i == j] + [Symbol("rho_{}{}".format(i,j)) for i in range(self.N) for j in range(self.N) if i != j] 



    ''' The following are get functions. These functions just 
        give us an object that we want. I.e. if we want the hamiltonian of our system
        we would just run States.getHamiltonian() '''

    # Returns the hamiltonian of our system
    def getHamiltonian(self):
        self.H = Matrix(self.H)
        return(self.H)


    # Returns Lindblad matrix of our system
    def getLindblad(self):
        self.matrix = Matrix(self.matrix)
        return(self.matrix)


    # Returns density matrix
    def getDensityMatrix(self):
        self.rho = Matrix(self.rho)
        return(self.rho)


    # returns 1/2*i[p,H] + L matrix
    def getSuperMatrix(self):
        self.A = Matrix(self.A)
        return(self.A)

    def getEquations(self):
        return(self.eqns)

    def getVector(self):
        return(self.vector)


    ''' The following functions just initialize the needed matrices
        This is done symbolically for some of them.'''

    # Function that evaluates the Lindblad operator for any given state.
    # In this case, the states are given by left and right 
    # The elements of the Lindblad matrix are given by 
    # <left|L|right>, this function creates one of those elements.
    def superOperatorElements(self, allowedTransitions, left, right):

        # Initializing placeholder list. Using a list for the sum() function
        # Which just adds all the elements within a list!
        self.lindbladMaster = []

        # Initializing input variables
        self.allowedTransitions = allowedTransitions
        self.left = left
        self.right = right

        # This section goes through a possible transition, computes the lindblad equation
        # and it then appends that result to the lindbladMaster list.
        for transitions in self.allowedTransitions:
            self.i = transitions[0]
            self.f = transitions[1]

            # Initialize Symbols
            self.gamma = Symbol('Gamma_{}{}'.format(self.i,self.f))
            self.rho_ii = Symbol('rho_{}{}'.format(self.i,self.i))
            self.rho_iright = Symbol('rho_{}{}'.format(self.i,self.right))
            self.rho_lefti = Symbol('rho_{}{}'.format(self.left,self.i))

            # Initializing Bras and Kets for the Lindblad Equation
            self.iKet = OrthogonalKet(self.i)
            self.iBra = OrthogonalBra(self.i)
            
            self.fKet = OrthogonalKet(self.f)
            self.fBra = OrthogonalBra(self.f)

            self.Right = OrthogonalKet(self.right)
            self.Left = OrthogonalBra(self.left)

            # Lindblad equation, look in notes to see what this is!
            self.L = self.gamma * ( (self.Left*self.fKet).doit() * self.rho_ii * (self.fBra*self.Right).doit() - 1/2*( (self.Left*self.iKet).doit() * self.rho_iright + self.rho_lefti * (self.iBra*self.Right).doit() ) )
            self.lindbladMaster.append(self.L)

        self.total = sum(self.lindbladMaster)

        return(self.total)


    # This function constructs the lindblad Matrix by using the superOperatorElements function. 
    # The only input that is needed for this function is allowedTransitions.
    # AllowedTransitions is a list of tuples that contain all the allowed transitions in our system!
    def lindbladMatrix(self, allowedTransitions,laser = False):

        if laser == False:
            self.transitions = allowedTransitions
            self.matrix = [[self.superOperatorElements(self.transitions,i,j) for j in range(self.N)] for i in range(self.N)]
        elif laser == True:
            self.transitions = allowedTransitions
            self.matrix = [[self.superOperatorElements(self.transitions,i,j) for j in range(self.N)] for i in range(self.N)]
            self.laser = [[0,Symbol('gamma_p')*Symbol('rho_01'),-(Symbol('gamma_p')+Symbol('gamma_c'))*Symbol('rho_02'),-(Symbol('gamma_p')+Symbol('gamma_c'))*Symbol('rho_03')],[-Symbol('gamma_p')*Symbol('rho_10'),0,-Symbol('gamma_c')*Symbol('rho_12'),-Symbol('gamma_c')*Symbol('rho_13')],[-(Symbol('gamma_p')+Symbol('gamma_c'))*Symbol('rho_20'),-Symbol('gamma_c')*Symbol('rho_21'),0,0],[-(Symbol('gamma_p')+Symbol('gamma_c'))*Symbol('rho_30'),-Symbol('gamma_c')*Symbol('rho_31'),0,0]]
            self.matrix = Matrix(self.matrix) + Matrix(self.laser)

    
    # This function constructs the density matrix and it has no inputs!
    def densityMatrix(self):

        self.rho = np.array([[Symbol('rho_{}{}'.format(i,j)) for j in range(self.N)] for i in range(self.N)])


    # This function computes 1/2*i*[p,H] + L
    # The function has no inputs since the p, H, and L matrices are stored automatically and globally. 
    def superMatrix(self):

        self.rho = Matrix(self.rho)
        self.matrix = Matrix(self.matrix)

        self.A = 1j*(self.rho*self.H - self.H*self.rho) + self.matrix



    ''' The following functions contain most of the computational work 
        First we unpack our super matrix and get a list of equations 
        Then we substitute some of the variables with static parameters 
        After we just substitute remaining variables with changing parameters 
        Then we just solve our matrix equation Ax = b'''

    # This function makes our list of equations by only using the static parameters.
    def makeEquations(self):

        # Extracting elements from super matrix and putting
        # them into a list        
        self.eqns = [self.A[i,j] for i in range(self.N) for j in range(self.N) if i == j] + [self.A[i,j] for i in range(self.N) for j in range(self.N) if i != j]

        # Substituting parameters that are fixed
        for key in self.static.keys():
            self.eqns = [i.subs(key,self.static[key]) for i in self.eqns ]
        
    
    # Function changes the non static parameters
    # The inputs are a dictionary and an optional key.
    def changeParam(self,variables):

        self.change = variables 

        # Just in case there were some things that were not evaluated.
        # This really only happens if you use the unevaluatedExpr() wrapper 

        # Makes our matrix and vector
        self.final , self.vector = linear_eq_to_matrix(self.eqns,self.rhos)

        self.final1 = self.remove(self.final,0,0)
        #self.final1 = SparseMatrix(self.final1)
        self.vector = np.array(self.vector).astype(np.complex128)
        self.vector = np.delete(self.vector,0,axis=0)

        self.f = lambdify(self.change,self.final1,'numpy')

        return(self.f)

        
    def remove(self,x,i,j):
        y = x[:-1,:-1]
        y[:i,j:] = x[:i,j+1:]
        y[i:,:j] = x[i+1:,:j]
        y[i:,j:] = x[i+1:,j+1:]

        return(y)

    # This function solves our final matrix problem!
    def solve(self,matrix):
        self.inputMatrix = matrix
        # Finally it solves our modified matrix and vector
        self.solutions = np.linalg.solve(self.inputMatrix, self.vector)

        return(self.solutions)
    

    # This function makes a pandas data frame
    def makeDataFrame(self, solutionsArray):

        # First, convert the list of rhos from symbols to strings
        self.columns = [str(i) for i in self.rhos[1:]]
        self.array = solutionsArray 

        # Make a data table (aka data frame) with labels
        self.df = pd.DataFrame(self.array, columns=self.columns)

        return(self.df)

    
