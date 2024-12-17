import pandas as pd
import numpy as np
import itertools
import nashpy
import bimatrix


def belief_construct(cond_player, targ_player):
    for it_cond, t_cond in enumerate(cond_player.T):
        cond_player.belief[t_cond] = {}
        for it_targ, t_targ in enumerate(targ_player.T):
            # p(t_targ|t_cond)
            cond_player.belief[t_cond][t_targ] = (targ_player.prior[it_targ] * cond_player.prior[it_cond] /
                                                  cond_player.prior[it_cond])
    return cond_player.belief

def two_type_strgy_name_gen(A=None, action_name=None, num_a=1):
    """
    Construct all pure strategies for one player
    :param A: strategy construct holder, e.g. A = ['LL', 'LR', 'RL', 'RR']
    :param num_a: number of actions in action space
    """
    B = []
    if A is not None:
        if action_name is None:
            for a in A:
                for action_indx in range(num_a):
                    B.append(f'{a}{action_indx}')
        else:
            for a in A:
                for action in action_name:
                    B.append(a + action)
    else:
        if action_name is None:
            assert num_a >= 1
            B = [f'{a_indx}' for a_indx in range(num_a)]
        else:
            assert type(action_name) is list
            B = action_name
    return B


class BNE_instance:
    def __init__(self, prior_1, prior_2, T1, T2, U1, U2, num_a1, num_a2, a1=None, a2=None):

        self.player1 = self.Player(prior_1, T1, a1, U1, num_a1)
        self.player2 = self.Player(prior_2, T2, a2, U2, num_a2)

        # Construct self-type conditional belief of other player types, p( | t_n)
        self.player1.belief = belief_construct(cond_player=self.player1, targ_player=self.player2)
        self.player2.belief = belief_construct(cond_player=self.player2, targ_player=self.player1)

    class Player:
        def __init__(self, prior, T, a, U, num_a, A=None):
            self.T = T
            self.prior = prior
            self.U = U
            self.num_a = num_a
            if a is None:
                self.a = [f'{indx}' for indx in range(num_a)]
            else:
                self.a = a

            self.belief = {}
            self.interim_exp = {}

            for count in range(len(T)):
                A = two_type_strgy_name_gen(A=A, action_name=a, num_a=num_a)
            self.A = A

    def interim_exp_comp(self, cond_player, targ_player):
        # Compute interim expected utility under all pure strategies (2 player)

        for indx_t_c, t_c in enumerate(cond_player.T):      # Over all types of cond_player
            cond_player.interim_exp[t_c] = {}
            for indx_a_c, a_c in enumerate(cond_player.a):      # Over actions in cond_player action space
                # e.g: a_cond = ['U', 'D'], action space, not strategy space
                # Fix one pure strategy, pi_n(t_n)
                for strgy_t in targ_player.A:   # Over all pure strategies of targ_player
                    # e.g. A_targ: = ['LL', 'LR', 'RL', 'RR'], strategy space, not action space
                    holder = []
                    for indx_t_targ, a_t in enumerate(strgy_t):     # Over 'L', 'R' in 'LR' from A2, for example
                        # Compute p(t_{-n}|t_n) * u_n(t_n, t_{-n}, a_n, pi_{-n}(t_{-n}) for one type t_{-n}
                        if cond_player is self.player1:
                            holder.append(
                                cond_player.U[indx_t_c][indx_a_c, targ_player.a.index(a_t)] * self.player1.belief[t_c][
                                    targ_player.T[indx_t_targ]])
                        elif cond_player == self.player2:
                            holder.append(
                                cond_player.U[indx_t_c][targ_player.a.index(a_t), indx_a_c] * self.player2.belief[t_c][
                                    targ_player.T[indx_t_targ]])
                        else:
                            raise ValueError('interim_exp_comp error')
                    # Compute the interim expected utility for player n given its type t_n, and pure strategy
                    # pi_n(t_n) = a_n, and pi_{-n}
                    cond_player.interim_exp[t_c][(a_c, strgy_t)] = sum(holder)
        return None

    def ex_ante_exp_comp(self, action_names=None):
        '''
            Assumes that only player 2's type varies
            (this means that player 1 has one action per row in U1,
             while 2 has nA2**2 (one choice per type))
            Both players have one utility matrix for each realization
            of player 2's type.

            INPUTS:
                U1: list of 2 payoff matrices for player 1 (row player)
                U2: list of 2 payoff matrices for player 2 (column player)
                p: (scalar) Probability that player 2 is the first type
                action_names: [optional] 2-list of names of actions (nA1 and nA2 long)
                            e.g. [['U','D'], ['LL','RL', 'LR', 'RR']]
                T1: list of player 1 types
                T2: list of player 2 types
            OUTPUTS:
                payoff1, payoff2: wide-form payoff matrices suitable for finding the NE
                A1, A2: names of actions
        '''
        # 2-player ex Ante Bayesian game formulation

        # Check the prior distribution are valid
        for player in [self.player1, self.player2]:
            for item in player.prior:
                assert np.isscalar(item)
                assert item >= 0.0
                assert item <= 1.0
            # Check the utility matrix matches the action space size
            for u in player.U:
                assert u.shape == (len(self.player1.a), len(self.player2.a))

        prior_list = [self.player1.prior, self.player2.prior]

        num_a1, num_a2 = len(self.player1.a), len(self.player2.a)  # Number of actions for player 1 and 2
        num_T1, num_T2 = len(self.player1.T), len(self.player2.T)  # Number of types for player 1 and 2

        # Initialize wide payoff matrix
        payoff1 = np.empty((num_a1 * num_T1, num_a2 * num_T2))
        payoff2 = np.empty((num_a1 * num_T1, num_a2 * num_T2))

        # Compute interim expected utility for player 1
        self.interim_exp_comp(cond_player=self.player1, targ_player=self.player2)
        # Compute interim expected utility for player 2
        self.interim_exp_comp(cond_player=self.player2, targ_player=self.player1)

        for i_row, strgy_1 in enumerate(self.player1.A):
            for i_col, strgy_2 in enumerate(self.player2.A):
                holder1 = []
                holder2 = []
                # Ex-ante expected utility for player 2
                for indx_t2, t2 in enumerate(T2):
                    holder2.append(self.player2.prior[indx_t2]
                                         * self.player2.interim_exp[t2][(strgy_2[indx_t2], strgy_1)])
                for indx_t1, t1 in enumerate(T1):
                    holder1.append(self.player1.prior[indx_t1]
                                         * self.player1.interim_exp[t1][(strgy_1[indx_t1], strgy_2)])

                payoff1[i_row, i_col] = sum(holder1)
                payoff2[i_row, i_col] = sum(holder2)

        return payoff1, payoff2


if __name__ == "__main__":
    prior_1 = np.array([1])
    T1 = ['SH_1']
    u11 = np.array([[3, 0], [2, 1]])
    U1 = [u11]
    a1 = ['U', 'D']
    num_a1 = 2

    p = 0.2
    prior_2 = np.array([p, 1-p])
    T2 = ['PD_2', 'SH_2']
    u21 = np.array([[3, 4], [1, 2]])
    u22 = np.array([[3, 2], [0, 1]])
    U2 = [u21, u22]
    a2 = ['L', 'R']
    num_a2 = 2

    game_info = BNE_instance(prior_1=prior_1, prior_2=prior_2, T1=T1, T2=T2, num_a1=num_a1, num_a2=num_a2, a1=a1, a2=a2,
                             U1=U1, U2=U2)
    game_info.interim_exp_comp(game_info.player2, game_info.player1)
    game_info.interim_exp_comp(game_info.player1, game_info.player2)
    payoff1, payoff2 = game_info.ex_ante_exp_comp(action_names=[game_info.player1.a, game_info.player2.a])
    # print(f'{game_info.player2.A}')

    # Construct the ex-ante utility matrix for both players
    print(bimatrix.print_payoffs([payoff1, payoff2],
                                 [game_info.player1.A, game_info.player2.A],
                                 round_decimals=3).to_string())

    # Eliminated strictly dominated actions for both players, if exists
    A_, payoff_ = bimatrix.IESDS(U=[payoff1, payoff2],A=[game_info.player1.A, game_info.player2.A], DOPRINT=False)
    print(bimatrix.print_payoffs(payoff_, A_, round_decimals=3).to_string())

    # Use nashpy to solve the ex-ante equilibrium (usually odd # expected)
    eqs = list(nashpy.Game(payoff_[0], payoff_[1]).support_enumeration())
    print(f'Found {len(eqs)} equilibria')
    for i, eq in enumerate(eqs):
        print(f'{i}: strgy1={eq[0]}, strgy2={eq[1]}')

    print('No code error spotted')

