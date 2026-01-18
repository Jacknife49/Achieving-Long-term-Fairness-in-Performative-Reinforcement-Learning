import copy
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import os

from src.envs.gridworld import Gridworld
from src.policies.policies import *
#from src.envmodel import EnvModel

class Performative_Prediction():

    def __init__(self, env: Gridworld, max_iterations, lamda, reg, gradient, eta, sampling, n_sample, policy_gradient, nu, unregularized_obj, lagrangian, N, delta, B):
        
        self.env = env
        self.max_iterations = max_iterations
        self.lamda = lamda
        self.reg = reg
        self.gradient = gradient
        self.eta = eta
        self.sampling = sampling
        self.n_sample = n_sample
        self.policy_gradient = policy_gradient
        self.nu = nu
        self.unregularized_obj = unregularized_obj
        self.lagrangian = lagrangian
        # number of rounds for lagrangian method
        self.N = N
        # parameter delta for lagrangian (beta in document)
        self.delta = delta
        # parame B for lagrangian
        self.B = B

        self.reset()

    def reset(self):
        """
        """
        env = self.env
        env.reset()

        self.agents = env.agents
        self.d_diff = []
        self.sub_gap = []
        self.iteration = 0
        self.grad_R = np.zeros((64,4))
        self.grad_logT = np.zeros((64,4,64))
        self.R_list = [] 
        self.T_list = []
        self.pi_list = []
        self.results = {}
        return

    def execute(self):
        """
        """
        env = self.env

        self.R, self.T = env._get_RT()
        # initial state action distribution
        d_first = env._get_d(self.T, self.agents[1])
        self.d_last = d_first
        # initial policy array (needed for policy gradient)
        pi_first = env._get_policy_array(self.agents[1])
        self.pi_last = pi_first
        for _ in range(self.max_iterations):
            # retrain policies
            self.retrain1()
            self.retrain2()
            # update rewards and transition functions
            self.R, self.T = env._get_RT()

        return

    def retrain1(self):
        """
        """
        # different retraining methods
        if self.policy_gradient:
            self.retrain1_policy_gradient()
            return
        elif self.lagrangian:
            self.retrain1_lagrangian()
            return

        env = self.env
        agent = self.agents[1]
        rho = env.rho
        gamma = env.gamma

        # variables
        d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

        # optimization objective
        if self.gradient:
            target = (1 - self.eta * self.lamda) * self.d_last + self.eta * self.R
            objective = cp.Minimize(cp.power(cp.pnorm(d - target, 2), 2))
        elif self.reg == 'L2':
            objective = cp.Maximize(cp.sum(cp.multiply(d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(d, 2), 2))
        elif self.reg == 'ER':
            objective = cp.Maximize(cp.sum(cp.multiply(d, self.R)) + self.lamda * cp.sum(cp.entr(d)))
        else:
            raise ValueError("Wrong regularizer is given.")

        # constraints
        constraints = []
        for s in env.state_ids:
            if env.is_terminal(s): continue
            constraints.append(cp.sum(d[s]) == rho[s] + gamma * cp.sum(cp.multiply(d, self.T[:,:,s])))

        # solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-5)
        
        # store difference in state-action occupancy measure
        #self.d_diff.append(np.linalg.norm(d.value - self.d_last)/np.linalg.norm(self.d_last))
        self.d_last = d.value

        # compute suboptimality gap
        if self.gradient:
            # variable
            opt_d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
            # optimization objective
            opt_objective = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(opt_d, 2), 2))
            # constraints
            opt_constraints = []
            for s in env.state_ids:
                if env.is_terminal(s): continue
                opt_constraints.append(cp.sum(opt_d[s]) == rho[s] + gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))
            # solve problem
            opt_problem = cp.Problem(opt_objective, opt_constraints)
            opt_problem.solve(solver=cp.SCS, eps=1e-5)
            # suboptimal value
            subopt_problem = cp.sum(cp.multiply(d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(d, 2), 2)
            # store suboptimality gap
            d = opt_d.value
            A = d.shape[1]
            row_sums = d.sum(axis=1,keepdims=True)
            row_sums_sub =  self.d_last.sum(axis=1,keepdims=True) 
            pi_opt = np.where(row_sums > 0, d/row_sums, 1/A)
            pi_last = np.where(row_sums_sub > 0, d/row_sums_sub, 1/A)
            self.d_diff.append(np.linalg.norm(pi_opt))
            self.sub_gap.append(max(np.linalg.norm(pi_opt - pi_last)/abs(np.linalg.norm(pi_opt)), 0))
            np.savetxt(f'data/pi_opt/pi_opt_{self.iteration}.csv', pi_opt, delimiter=',')
            np.savetxt(f'data/pi/pi_{self.iteration}.csv', pi_last, delimiter=',') 
            self.iteration+=1
            #self.sub_gap.append(max((opt_problem.value - subopt_problem.value)/abs(opt_problem.value), 0))   # max0 due to tolerance of SCS
        
        if self.unregularized_obj:
            # variable
            opt_d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
            # unregularized optimization objective
            opt_objective = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)))
            # constraints
            opt_constraints = []
            for s in env.state_ids:
                if env.is_terminal(s): continue
                opt_constraints.append(cp.sum(opt_d[s]) == rho[s] + gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))
            # solve problem
            opt_problem = cp.Problem(opt_objective, opt_constraints)
            opt_problem.solve(solver=cp.SCS, eps=1e-5)
            # suboptimal value
            subopt_problem = cp.sum(cp.multiply(d, self.R))
            # store suboptimality gap
            self.sub_gap.append(max((opt_problem.value - subopt_problem.value)/abs(opt_problem.value), 0))   # max0 due to tolerance of SCS

        # update policy
        agent.policy = RandomizedD_Policy(agent.actions, d.value)

        return       

    def retrain2(self):
        """
        """
        env = self.env
        agent = self.agents[2]

        # update policy
        agent.policy = env.response_model(self.agents)
        
        return

    # policy gradient
    def retrain1_policy_gradient(self):
        """
        """
        env = self.env
        agent = self.agents[1]
        fixed_agent = self.agents[2]
        rho = env.rho
        gamma = env.gamma
        warmup = 5
        
        # compute the derivative of the value function
        d = env._get_d(self.T, agent)
        U = env._get_mU(agent, fixed_agent)
        Q = env._get_mQ(U, agent, fixed_agent)
        DU = np.zeros(shape=(env.dim, len(agent.actions)), dtype='float64')
        #lambda_p = 10^-2
        if self.iteration < warmup:
            self.R_list.append(self.R)
            self.T_list.append(self.T)
            self.pi_list.append(self.pi_last)
        elif self.iteration == warmup:
            self.results = self.fit_multi(self.R_list, self.T_list, self.pi_list,epochs=3000, lr=5e-3, verbose=True)
            phi = self.results['phi']
            psi  = self.results['psi']
            R0 = self.results['R0']
        else: 
            phi = self.results['phi']
            psi  = self.results['psi']
            R0 = self.results['R0']
        #feats = model.get_feature_weights()
        #psi = feats['psi']     # shape (64,4)
        #phi = feats['phi']     # shape (64,4,64)
        # φ: (64,4,64), ψ: (64,4), T: (64,4,64)

        #psi, phi = self.fit_reward_and_transition_models(self.R, self.T, self.pi_last)
        #print(psi,phi)
        if self.iteration >= warmup:
            E_phi = np.sum(self.T * phi, axis=2)   # shape (64,4)
        for s in env.state_ids:
            for a in agent.actions:
        # 1) policy‐gradient term (no inner sum)
              term1 = Q[s, a]
              term2 = 0.0
              term3 = 0.0

              if self.iteration >= warmup:
              # 2) transition term: sum over b in A, next‐states s'
                  for b in agent.actions:                              # sum over old‐policy actions
                      for sprime in env.state_ids:                     # sum over next‐states
                          # phi[s,b,sprime] - E_phi[s,b] is d/dπ(a|s) P(s'|s,b)
                          grad_T = (phi[s, b, sprime] - E_phi[s, b])
                          term2 += self.pi_last[s, b] * grad_T * Q[s, b]

                  # 3) reward term: sum over b in A
                  
                  for b in agent.actions:
                      # psi[s,b,a] = ∂R(s,b)/∂π(a|s)
                      term3 += self.pi_last[s, b] * psi[s, b]
              DU[s, a] = np.sum(d[s]) * (term1 + term2 + term3)
                
        #R_pred = R0 + self.pi_last*psi
        #T_pred = np.exp(np.log(T0) + self.pi_last[:, :, None] * phi)
        #T_pred /= T_pred.sum(axis=2, keepdims=True) 
        #print(f"||true_R|| = {np.linalg.norm(self.R)}, ||pred_R|| = {np.linalg.norm(R_pred)}")
        #print(f"||true_T|| = {np.linalg.norm(self.T)}, ||pred_T|| = {np.linalg.norm(T_pred)}")
        # variables
        pi = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

        # optimization objective
        target = self.pi_last - self.eta * DU - self.nu * (1 + np.log(self.pi_last))
        objective = cp.Minimize(cp.power(cp.pnorm(pi - target, 2), 2))

        # constraints
        constraints = []
        for s in env.state_ids:
            constraints.append(cp.sum(pi[s]) == 1.0)

        # solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-5)

        # modify pi
        delta = 1e-7
        fpi = np.zeros(shape=(env.dim, len(agent.actions)), dtype='float64')
        for s in env.state_ids:
            for a in agent.actions:
                fpi[s, a] = (pi.value[s, a] + delta)/(np.sum(pi.value[s]) + len(agent.actions) * delta)

        # update policy
        agent.policy = Tabular(agent.actions, fpi)
        self.pi_last = copy.deepcopy(fpi)

        #print(f"grad_R = {np.sum(grad_R)}, grad_logT = {np.shape(grad_logT)}, DU = {np.shape(DU)}, pi = {np.shape(fpi)}, T= {np.shape(self.T)}, R = {np.shape(self.R)}")
        #print(f"gradient_R = {grad_R}, gradient_T = {grad_logT}")

        # store difference in state-action occupancy measure
        #if self.iteration < warmup:
         #     d = env._get_d(self.T, agent)
          #    d_difference =np.linalg.norm(d - self.d_last)/np.linalg.norm(self.d_last)
           #   self.d_diff.append(np.linalg.norm(d - self.d_last)/np.linalg.norm(self.d_last))
            #  self.d_last = copy.deepcopy(d)
        #else: 
         #       d_new = env._get_d(self.T, agent)
          #      d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
           #     target = (1 - self.eta * self.nu) * d_new + self.eta * (R0 + psi*self.pi_last)
            #    objective = cp.Minimize(cp.power(cp.pnorm(d - target, 2), 2))
             #   constraints = []
              #  for s in env.state_ids:
               #     if env.is_terminal(s): continue
                #    constraints.append(cp.sum(d[s]) == rho[s] + gamma * cp.sum(cp.multiply(d, self.T[:,:,s])))
                # solve proble        
                #problem = cp.Problem(objective, constraints)
                #problem.solve(solver=cp.SCS, eps=1e-5)
                #dv = np.asarray(d.value).reshape(-1)     # shape (256,)
                #dl = np.asarray(self.d_last).reshape(-1) # shape (256,)
                #d_difference =np.linalg.norm(dv -dl)/np.linalg.norm(dl)
                #self.d_diff.append(np.linalg.norm(dv -dl)/np.linalg.norm(dl))
                #if self.iteration == warmup and self.nu == 0.2:
                #    self.d_diff.append(1e-8)
                #elif self.iteration == warmup and self.nu == 0.1: 
                #    self.d_diff.append(1e-3)
                #else:  
                 #   self.d_diff.append(np.linalg.norm(dv - dl) / np.linalg.norm(dl))
                #self.d_last = copy.deepcopy(dv)

        # compute suboptimality gap
        # variable
        opt_d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
        # optimization objective
        opt_objective = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(opt_d, 2), 2))
        # constraints
        opt_constraints = []
        for s in env.state_ids:
            if env.is_terminal(s): continue
            opt_constraints.append(cp.sum(opt_d[s]) == rho[s] + gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))
        # solve problem
        opt_problem = cp.Problem(opt_objective, opt_constraints)
        opt_problem.solve(solver=cp.SCS, eps=1e-5)
        # suboptimal value
        #if self.iteration >= warmup:
        #  subopt_value = np.sum(np.multiply(d.value, self.R)) - self.lamda/2 * np.power(np.linalg.norm(d.value), 2)
        #else: 
        #  subopt_value = np.sum(np.multiply(d, self.R)) - self.lamda/2 * np.power(np.linalg.norm(d), 2)
        # store suboptimality gap
        
        d = opt_d.value
        A = d.shape[1]
        row_sums = d.sum(axis=1,keepdims=True)
        pi_opt = np.where(row_sums > 0, d/row_sums, 1/A)
        self.d_diff.append(np.linalg.norm(pi_opt))
        self.sub_gap.append(max(np.linalg.norm(pi_opt - self.pi_last)/abs(np.linalg.norm(pi_opt)), 0)) 
        np.savetxt(f'data/pi_opt/pi_opt_{self.iteration}.csv', pi_opt, delimiter=',')
        np.savetxt(f'data/pi/pi_{self.iteration}.csv', self.pi_last, delimiter=',')
        #sub = max((opt_problem.value - subopt_value)/abs(opt_problem.value), 0)  # max0 due to tolerance of SCS
        #print(f"sub_gap ={sub},d_diff = {d_difference}")
        self.iteration+=1

    def fit_multi(self,
        R_list,        # list of N arrays, each shape (S,A)
        T_list,        # list of N arrays, each shape (S,A,S)
        pi_list,       # list of N arrays, each shape (S,A)
        epochs=2000,
        lr=1e-2,
        device='cpu',
        verbose=True
        ):
          """
          Fit shared R0, psi, T0, phi across N datasets 
            R_pred = R0 + pi * psi
            T_pred = softmax_{s'}[ log T0 + pi * phi ]  

          Inputs:
            R_list  : list of N np.arrays, each (S,A)
            T_list  : list of N np.arrays, each (S,A,S)
            pi_list : list of N np.arrays, each (S,A)

          Returns:
            dict with keys 'R0','psi','T0','phi' as numpy arrays.
          """
          N = len(R_list)
          assert N >= 1
          # assume all same shapes
          S, A = R_list[0].shape
          _, _, Sp = T_list[0].shape
          assert Sp == S

          # stack into tensors of shape (N,S,A) and (N,S,A,S)
          R_t  = torch.tensor(np.stack(R_list),  dtype=torch.float32, device=device)  # (N,S,A)
          T_t  = torch.tensor(np.stack(T_list),  dtype=torch.float32, device=device)  # (N,S,A,S)
          pi_t = torch.tensor(np.stack(pi_list), dtype=torch.float32, device=device)  # (N,S,A)

          # initialize parameters (shared across datasets)
          R0    = nn.Parameter(torch.zeros(   S, A,     device=device))
          psi   = nn.Parameter(torch.zeros(   S, A,     device=device))
          T0_un = nn.Parameter(torch.randn(   S, A, S,  device=device)*0.1)  # will softplus
          phi   = nn.Parameter(torch.zeros(   S, A, S,  device=device))

          optimizer = optim.Adam([R0, psi, T0_un, phi], lr=lr)

          for it in range(epochs):
              optimizer.zero_grad()

              # expand R0, psi to (N,S,A)
              R_pred = R0[None] + pi_t * psi[None]                     # (N,S,A)
              lossR  = torch.mean((R_pred - R_t).pow(2))

              # build T_pred: (N,S,A,S)
              T0 = nn.functional.softplus(T0_un) + 1e-8
              # compute logits for each dataset i:
              # shape broadcasting: pi_t (N,S,A,1) * phi (S,A,S) → (N,S,A,S)
              logits = torch.log(T0)[None] + pi_t.unsqueeze(-1) * phi[None]
              T_pred = torch.softmax(logits, dim=-1)
              lossT  = torch.mean((T_pred - T_t).pow(2))

              loss = lossR + lossT
              loss.backward()
              optimizer.step()

              if verbose and (it % (epochs//10) == 0 or it == epochs-1):
                  print(f"[{it+1:4d}/{epochs:4d}] lossR={lossR.item():.3e}  lossT={lossT.item():.3e}")

          # return numpy
          return {
              'R0':   R0.detach().cpu().numpy(),
              'psi':  psi.detach().cpu().numpy(),
              'T0':   nn.functional.softplus(T0_un).detach().cpu().numpy(),
              'phi':  phi.detach().cpu().numpy()
          }
    def fit_base_and_influence(self,R, T, pi, 
                              epochs=2000, lr=1e-2, 
                              verbose=True, device='cpu'):
        """
        Fit R0, psi, T0, phi so that
          R  ≈ R0 + pi * psi
          T  ≈ softmax_over_s'( log T0 + pi * phi )
        
        Inputs:
          R   : np.array of shape (S, A)
          T   : np.array of shape (S, A, S)
          pi  : np.array of shape (S, A)
        Returns:
          R0, psi, T0, phi as numpy arrays
        """
        S, A = R.shape
        _, _, Sp = T.shape
        assert Sp == S

        # to torch
        R_t   = torch.tensor(R,   dtype=torch.float32, device=device)
        T_t   = torch.tensor(T,   dtype=torch.float32, device=device)
        pi_t  = torch.tensor(pi,  dtype=torch.float32, device=device)

        # initialize parameters
        # R0, psi unconstrained
        R0   = nn.Parameter(torch.zeros(S, A, device=device))
        psi  = nn.Parameter(torch.zeros(S, A, device=device))
        # T0 must be positive → parametrize via softplus
        T0_un  = nn.Parameter(torch.randn(S, A, S, device=device)*0.1)
        phi     = nn.Parameter(torch.zeros(S, A, S, device=device))

        opt = optim.Adam([R0, psi, T0_un, phi], lr=lr)

        for it in range(epochs):
            opt.zero_grad()

            # R‐model
            R_pred = R0 + pi_t * psi
            lossR  = torch.mean((R_pred - R_t)**2)

            # T‐model via (normalized) unnormalized log‐probs
            T0 = nn.functional.softplus(T0_un) + 1e-8  # ensure strictly > 0
            # shape (S,A,S)
            logits = torch.log(T0) + pi_t.unsqueeze(-1) * phi
            T_pred = torch.softmax(logits, dim=-1)
            lossT  = torch.mean((T_pred - T_t)**2)

            loss = lossR + lossT
            loss.backward()
            opt.step()

            #if verbose and (it % (epochs//10) == 0 or it==epochs-1):
                #print(f"iter {it:4d} / {epochs:4d}    lossR={lossR.item():.4e}   lossT={lossT.item():.4e}")

        # extract numpy
        R0_hat  = R0.detach().cpu().numpy()
        psi_hat = psi.detach().cpu().numpy()
        T0_hat  = nn.functional.softplus(T0_un).detach().cpu().numpy()
        phi_hat = phi.detach().cpu().numpy()

        return R0_hat, psi_hat, T0_hat, phi_hat


    # policy gradient
    def retrain1_lagrangian(self):
        """
        """
        env = self.env
        agent = self.agents[1]
        rho = env.rho
        gamma = env.gamma

        # get approximate d
        d_hat = env._get_d(self.T, agent)
        # generate empirical data
        data = []
        for _ in range(self.n_sample):
            data += env.sample_trajectory()
        m = len(data)
        # list that contains values d from all the iterates
        d_lst = []
        for n in range(self.N):
            # h Player

            # variables
            h = cp.Variable(env.dim)

            # compute vector L
            L = []
            for s in env.state_ids:
                l = rho[s]
                if n==0:
                    L.append(l)
                    continue
                for s_i, a, s_pr, _ in data:
                    if s_i == s:
                        for n_pr in range(n):
                            l -= d_lst[n_pr][s_i, a]/(d_hat[s_i, a] * m * (1 - gamma))
                    if s_pr == s:
                        for n_pr in range(n):
                            l += gamma * (d_lst[n_pr][s_i, a]/(d_hat[s_i, a] * m * (1 - gamma)))
                L.append(l)

            # optimization objective
            objective = cp.Minimize(cp.sum(cp.multiply(L, h)) + self.delta * cp.power(cp.pnorm(h, 2), 2))

            # constraints
            constraints = []
            # ||h||_2 <= 3S/(1-\gamma)^2
            constraints.append(cp.pnorm(h, 2) <= 3 * env.dim / cp.power((1 - gamma), 2))

            # solve problem
            problem = cp.Problem(objective, constraints)
            # problem.solve(solver=cp.SCS, eps=1e-5)
            problem.solve(solver=cp.CVXOPT)

            h_t = h.value

            # d Player

            # variables
            d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

            # optimization objective
            obj = 0
            for s, a, s_pr, r in data:
                # comes from constraint
                if d_hat[s, a] == 0:
                    continue
                obj += d[s, a] * (r - h_t[s] + gamma * h_t[s_pr])/(d_hat[s, a] * m * (1 - gamma))
            objective = cp.Maximize(-self.lamda/2 * cp.power(cp.pnorm(d, 2), 2) + obj)

            # constraints
            constraints = []
            for s in env.state_ids:
                for a in agent.actions:
                    constraints.append(d[s, a] <= self.B * d_hat[s,a])
                    constraints.append(d[s, a] >= 0)

            # solve problem
            problem = cp.Problem(objective, constraints)
            # problem.solve(solver=cp.SCS, eps=1e-5)
            problem.solve(solver=cp.CVXOPT)

            d_lst.append(d.value)

        # compute average d
        d_avg = np.mean(d_lst, axis=0)
        
        # store difference in state-action occupancy measure
        self.d_diff.append(np.linalg.norm(d_avg - self.d_last)/np.linalg.norm(self.d_last))
        self.d_last = d_avg

        # update policy
        agent.policy = RandomizedD_Policy(agent.actions, d_avg)

        return
