def deriv(y0, t, k, N, beta, gamma, alpha, theta_r, omega):
    """
    This function computes the derivatives for the SIR model with vaccinations.
        y0: (tuple) a vector describing initial conditions for each population
        t: (numpy array) the time for which to compute the derivatives
        k: (int) the number of populations in the model
        N: (tuple) a k length vector of the population sizes
        beta: (tuple) a k length vector of S -> I coefficients
        gamma: (tuple) a k length vector of I -> R coeffs
        alpha: (tuple) a k length vector of I -> D coeffs
        theta_r: (numpy array) a k length vector describing the maximum vaccinations per day for each population (rate)
        omega: (numpy array) a k x k matrix descrbibing interconnectedness (symetric, zeros on the diagonal)
    """
    # Initialize the derivatives
    dSdt, dIdt, dRdt, dDdt, dVdt = tuple(np.zeros(k) for _ in range(5))
    
    # Split the y vector into Susceptible, Infected, and Vaccines allocated
    S, I, V = y0[:k], y0[k:2*k], y0[4*k:]

    # use vaccines according to the administrative constraint (theta_r)
    for i in range(k):        
        if V[i] > theta_r[i]:
            dVdt[i] = (-1)*theta_r[i]
        else:
            dVdt[i] = (-1)*V[i]
                    
        # compute the derivatives
        dSdt[i] = (-S[i]/N[i])*(beta[i]*I[i] + np.matmul(omega[i, ], I)) + dVdt[i]
        dIdt[i]  = (S[i]/N[i])*(beta[i]*I[i] + np.matmul(omega[i, ], I)) - I[i]*(1 - alpha[i])*gamma[i] - I[i]*alpha[i] 
        dRdt[i] = I[i]*(1 - alpha[i])*gamma[i] - dVdt[i]
        dDdt[i] = I[i]*alpha[i]
                
    solution = np.concatenate((dSdt, dIdt, dRdt, dDdt, dVdt))

    return solution

def calculate_deaths(y0, t, k, N, beta, gamma, alpha, theta_r, omega, T):
    """
    This function computes the deaths at a given time T for the SIR model with vaccinations.
        y0: (tuple) a vector describing initial conditions for each population
        t: (numpy array) the time for which to compute the derivatives
        k: (int) the number of populations in the model
        N: (tuple) a k length vector of the population sizes
        beta: (tuple) a k length vector of S -> I coefficients
        gamma: (tuple) a k length vector of I -> R coeffs
        alpha: (tuple) a k length vector of I -> D coeffs
        theta_r: (numpy array) a k length vector describing the maximum vaccinations per day for each population (rate)
        omega: (numpy array) a k x k matrix descrbibing interconnectedness (symetric, zeros on the diagonal)
        T: (int) the time at which to compute the deaths
    """
    ret = odeint(deriv, y0, t, args=(k, N, beta, gamma, alpha, theta_r, omega))
    D = ret.T[3*k:4*k]
    D_list = np.zeros(k)
    for i in range(k):
        D_list[i] = D[i, T]
        
    return D_list

def calculate_susceptible(y0, t, k, N, beta, gamma, alpha, theta_r, omega, T):
    """
    This function computes the susceptible at a given time T for the SIR model with vaccinations.
        y0: (tuple) a vector describing initial conditions for each population
        t: (numpy array) the time for which to compute the derivatives
        k: (int) the number of populations in the model
        N: (tuple) a k length vector of the population sizes
        beta: (tuple) a k length vector of S -> I coefficients
        gamma: (tuple) a k length vector of I -> R coeffs
        alpha: (tuple) a k length vector of I -> D coeffs
        theta_r: (numpy array) a k length vector describing the maximum vaccinations per day for each population (rate)
        omega: (numpy array) a k x k matrix descrbibing interconnectedness (symetric, zeros on the diagonal)
        T: (int) the time at which to compute the deaths
    """
    ret = odeint(deriv, y0, t, args=(k, N, beta, gamma, alpha, theta_r, omega))
    S = ret.T[0:k]
    S_list = np.zeros(k)
    for i in range(k):
        S_list[i] = S[i, T]
        
    return S_list

def find_marginal_vax_effect(y0, t, k, N, beta, gamma, alpha, theta_r, omega, T):
    """
    This function computes the marginal effect of vaccines on deaths at a given time T.
        y0: (tuple) a vector describing initial conditions for each population
        t: (numpy array) the time for which to compute the derivatives
        k: (int) the number of populations in the model
        N: (tuple) a k length vector of the population sizes
        beta: (tuple) a k length vector of S -> I coefficients
        gamma: (tuple) a k length vector of I -> R coeffs
        alpha: (tuple) a k length vector of I -> D coeffs
        theta_r: (numpy array) a k length vector describing the maximum vaccinations per day for each population (rate)
        omega: (numpy array) a k x k matrix descrbibing interconnectedness (symetric, zeros on the diagonal)
        T: (int) the time at which to compute the deaths
    """
    # calculate deaths for the initial scenario, for each society the starting death is the same (the sum of all)
    D_list_init = calculate_deaths(y0, t, k, N, beta, gamma, alpha, theta_r, omega, T)
    D_init = sum(D_list_init)
    for i in range(k):
        D_list_init[i] = D_init
    y01 = list(y0)
    
    D_list_new = np.zeros(k)

    #add 10 to every society, calculate and assign total death change
    for i in range(k):
        y01 = list(y0)
        y01[4*k + i] = y01[4*k + i] + 10
        y01 = tuple(y01)
        D_list_new[i] = sum(calculate_deaths(y01, t, k, N, beta, gamma, alpha, theta_r, omega, T))
    
    D_list_init = matrix(D_list_init)
    D_list_new = matrix(D_list_new)
    
    m_eff = (D_list_init - D_list_new)
    m_eff = m_eff.tolist()
    m_eff = m_eff[0]
        
    return m_eff

def transfer_vax(m_eff, y0, k):
    """
    This function computes the marginal effect of vaccines on deaths at a given time T.
        m_eff: (list) a list of marginal effects
        y0: (tuple) a vector describing initial conditions for each population
        k: (int) the number of populations in the model
    """
    # find the min and max indices
    min_ind = m_eff.index(min(m_eff))
    max_ind = m_eff.index(max(m_eff))
    m_eff_inside = m_eff.copy()
    
    # get the vaccine vector
    y1 = list(y0).copy()
    V = y1[4*k: 5*k] 

    # initialize the check vector   
    check = list(np.zeros(len(V)))

    # transfer 10 vax from min ind to max ind
    while sum(check) != -len(V) + 1:
        min_ind = m_eff_inside.index(np.nanmin(m_eff_inside))
        max_ind = m_eff_inside.index(np.nanmax(m_eff_inside))
        if V[min_ind] >= 10:
            V[min_ind] = V[min_ind] - 10
            V[max_ind] = V[max_ind] + 10
            break
        else:
            m_eff_inside[min_ind] = np.nan
            check[min_ind] = -1
        
    # check if the transfer is possible
    if sum(check) != -len(V) + 1:
        y1[4*k: 5*k] = V
        y1 = tuple(y1)
        return y1    
    else:
        y1.append(-1)
        return y1
    
    