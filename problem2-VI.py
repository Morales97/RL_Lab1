import numpy as np
import random as rnd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

class City:
    def __init__(self, allow_passing = False):
        '''Initialize rules'''
        self.allow_p = allow_passing
        self.positions = [(i,j) for i in range(3) for j in range(6)]
        self.n_pos = len(self.positions)
        self.n_states = self.n_pos**2
        self.banks = [(0,0),(2,0),(2,5),(0,5)]
        self.station = (1,2)
        self.actions = [(1,0),(-1,0),(0,1),(0,-1),(0,0)] # D,U,R,L,none
        self.n_actions = len(self.actions)
        self.p_moves = [(1,0),(-1,0),(0,1),(0,-1)] # D,U,R,L
        self.init_state = self.pos_to_state((self.banks[0],
                                             self.station))
        self.trans_p = self.__tp() # trans_p[next_s,curr_s,action]
        self.r = self.__gen_rewards() # r[state, action]
        self.policy = None # policy[state] gives optimal action index
        self.value = None # value[state]
        self.disc = None # currently used discount for value and policy
        
        for state in range(self.n_states): # just to make sure
            assert (state == self.pos_to_state(self.state_to_pos(state)))
    
    def pos_to_state(self, position_pair):#robber_position, police_position):
        '''Maps position pairs to their state number'''
        return (self.n_pos * self.positions.index(position_pair[0]) +
                            self.positions.index(position_pair[1]))
        
    def state_to_pos(self, state):
        '''Maps state numbers to their position pair'''
        return (self.positions[state//self.n_pos],
                self.positions[state % self.n_pos])

    def __move(self, position, action):
        '''Return new position if possible, otherwise False'''
        new_p = tuple(x + y for x, y in zip(position, action))
        return new_p if new_p in self.positions else False

    def __gen_rewards(self):
        '''Generate self.r (rewards) as r[state_number, action number]'''
        reward = np.zeros([self.n_states,self.n_actions])
        for s in range(self.n_states):
            state_rob = self.state_to_pos(s)[0]
            state_pol = self.state_to_pos(s)[1]
            if state_rob == state_pol:
                reward[s,:] = -50   # If the polices catches the robber, -50 SEK
            elif state_rob in self.banks:
                reward[s,:] = 10    # If the robber is in the bank without the police, 10 SEK
        return reward

    def __tp(self):
        '''Generate transition prob tensor'''
        prob = np.zeros([self.n_states,self.n_states,self.n_actions])
        for s,sp in ((i,j) for i in range(self.n_pos) for j in range(self.n_pos)):
            if s == sp: # every action gives prob 1 of going to initial state
                prob[self.init_state, self.n_pos*s+sp, :] = 1
            else:
                robber_pos, po_pos = self.positions[s], self.positions[sp]
                diff = np.subtract(robber_pos, po_pos) # cop-to-robber vector
                valid_new_po_pos = []
                for m in self.p_moves:
                    if abs(diff[0]*m[0]>=0) and abs(diff[1]*m[1]>=0):
                        # if police move is in the generalized direction of the robber
                        npp = self.__move(po_pos, m)
                        if npp:
                            valid_new_po_pos.append(npp)
                n_vnpp = len(valid_new_po_pos)
      
                for a in range(self.n_actions):
                    nrp = self.__move(robber_pos, self.actions[a])
                    if nrp:
                        if not self.allow_p and nrp == po_pos:
                            # If we dont want to allow them to pass through each others
                            # this statement checks.
                            # it makes the police stay where robber is
                            prob[self.pos_to_state((nrp,nrp)),
                                 self.n_pos*s+sp, a] = 1
                        else:
                            for npp in valid_new_po_pos:
                                prob[self.pos_to_state((nrp,npp)),
                                     self.n_pos*s+sp, a] = 1/n_vnpp
                    else: # action moves robber against wall=> robber doesn't move
                        for npp in valid_new_po_pos:
                            prob[self.pos_to_state((robber_pos,npp)),
                                    self.n_pos*s+sp, a] = 1/n_vnpp
        return prob

    def show(self):
        '''Draw field and weights in terminal'''
        print('City:\n'+'█'+' ██ '*6 +'█')
        for i in range(3):
            row = '█'
            for j in range(6):
                if (i,j) in self.banks:
                    row +='Bank'
                elif (i,j) == self.station:
                    row+=' PS '
                else: row+='    '
            print(row+'█')
        print('█'+' ██ '*6 +'█')
        W = np.zeros((3,6))
        for p in self.positions:
            W[p] = self.r[self.pos_to_state((p,self.station)),-1]
        print(f'Rewards for standing still when police at station:\n{W}')

    def robbing_vi(self, discount, epsilon, talk=False):
        '''Set self.policy and self.value according to VI'''
        if not discount < 1 or not discount > 0:
            print('invalid discount')
            return
        V = np.zeros(self.n_states)
        Q = np.zeros((self.n_states, self.n_actions))
        thresh = epsilon*(1 - discount)/discount
        delta = np.inf
        n = 0
        while delta > thresh:
            n += 1
            if talk and not n%10:
                print(f'n={n}, delta={delta}, threshold={thresh}', end='\r')
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Q[s, a] = self.r[s, a] + discount*np.dot(self.trans_p[:,s,a],V)
            bellman_V = np.max(Q, 1)
            delta = np.linalg.norm(bellman_V - V)
            V  = np.copy(bellman_V)
        self.disc = discount
        self.policy = np.argmax(Q,1)
        self.value = V
        if talk: print(f'Policy generated for discount={discount}; iterations: {n}')

    def sim_policy(self, T, seed, do_plot=False, do_anim=False):
        '''simulate the currently held policy for T turns'''
        rnd.seed(seed)
        if self.policy is None:
            print('Generate a policy first, with robbing_vi(discount, epsilon)')
            return
        path = [self.state_to_pos(self.init_state)[0]]
        po_path = [self.state_to_pos(self.init_state)[1]]
        money = [0]
        caught = 0
        state = self.pos_to_state((path[-1],po_path[-1]))
        for t in range(T):
            state_rob = self.state_to_pos(state)[0]
            state_pol = self.state_to_pos(state)[1]
            if state_rob == state_pol:
                caught+=1
                money.append(money[-1] - 50)
            elif state_rob in self.banks:
                money.append(money[-1] + 10)
            else:
                money.append(money[-1])
            opta = self.policy[state] # follow policy
            state = rnd.choices(range(self.n_states),
                                weights = self.trans_p[:,state,opta])[0]
            pos_pair = self.state_to_pos(state)
            path.append(pos_pair[0])
            po_path.append(pos_pair[1])
        
        if do_plot or do_anim:
            tmplist = [elem for elem in zip(*path)]
            pathx = tmplist[1]
            pathy = tmplist[0]
            tmplist = [elem for elem in zip(*po_path)]
            po_pathx = tmplist[1]
            po_pathy = tmplist[0]
            assert (pathx != po_pathx) # lists are weird
        
        if do_plot:
            plt.figure(0)
            plt.scatter(*zip(*[(b[1],b[0]) for b in self.banks]),s=1000,c = "palegreen",marker="s")
            plt.scatter(*self.station[::-1],s=1000,c = "k",marker="s")
            plt.scatter(po_pathx,po_pathy,c=range(T+1),cmap="autumn",marker='s',s=100)
            plt.scatter(pathx,pathy,c=range(T+1),cmap="winter")
            plt.xticks(range(6))
            plt.yticks(range(3))
            plt.gca().invert_yaxis()
            plt.title(f'Paths in City with discount = {self.disc}\n'+ 
                      f'seed = {seed}; t_end = {T}; '+
                      f'max(money) = {int(max(money))}; caught {caught} times')
            plt.show()
            
            plt.figure(1)
            plt.plot(range(T+1),money)
            plt.xticks(range(0,T+1,5))
            plt.xlim([0,T+1])
            plt.title(f'Money vs time discount = {self.disc} for seed = {seed}')
            plt.show()
        if do_anim:
            self.__animate_sim(T,pathx,pathy,po_pathx,po_pathy,seed,money)

        return money[-1]
    
    def __animate_sim(self,T,pathx,pathy,po_pathx,po_pathy,seed,money):
        f = plt.figure(0)
        plt.scatter(*zip(*[(b[1],b[0]) for b in self.banks]),s=1000,c = "palegreen",marker="s")
        plt.scatter(*self.station[::-1],s=1000,c = "k",marker="s")
        plt.xticks(range(6))
        plt.yticks(range(3))
        plt.gca().invert_yaxis()
        x = [pathx[0],po_pathx[0]]
        y = [pathy[0],po_pathy[0]]
        scat = plt.scatter(x,y, c=['b' ,'r'],s=[100,200],zorder=100)

        def animationUpdate(t):
            '''anim update function'''
            plt.plot(po_pathx[:t+1],po_pathy[:t+1],c='r',linewidth=3)
            plt.plot(pathx[:t+1],pathy[:t+1],c='b')
            x = [pathx[t],po_pathx[t]]
            y = [pathy[t],po_pathy[t]]
            if t < len(money) - 1 and money[t+1] < money[t]:
                    print(f'caught at x={[pathx[t-1],po_pathx[t-1]]}, '+
                          f'y={[pathy[t-1],po_pathy[t-1]]}')
                    scat.set_sizes([100,1000])
            else:
                scat.set_sizes([100,200])
            scat.set_offsets(np.c_[x,y])
            plt.title(f'λ = {self.disc}; seed = {seed}; t = {t}; money = {int(money[t])}')
            
            if t == len(pathx) - 1:
                caught = 0
                for i in range(len(money)-1):
                    if money[i+1] < money[i]: caught += 1
                plt.title(f'Paths in City with discount = {self.disc}\n'+ 
                      f'seed = {seed}; t_end = {t}; '+
                      f'money_end = {int(money[t])}; caught {caught} times')
                plt.savefig(f'{self.disc}_s{seed}_t{t}.png')
            return scat,
        anim = FuncAnimation(f, animationUpdate, frames=T+1, interval=100, blit=False)
        writergif = animation.PillowWriter(fps=4)
        anim.save(f'p2_{self.disc}_{seed}.gif', writer=writergif)
        plt.clf()
        print(f'Saved video as p2_{self.disc}_{seed}.gif') 


def interact_with_discount(city, discount, eps, seed):
    '''Meant to be used in a jupyter notebook'''
    city.robbing_vi(discount,eps)
    city.sim_policy(T=100,seed=seed,do_plot=True,do_anim=False)
    vf = city.value[city.init_state]
    print(f'V_0 = {vf}')


def val_func_vs_discount(city,eps):
    '''Plots value at initial state vs discount'''
    discounts = np.linspace(1e-5,1,100,endpoint=False)
    vfs=[]
    for d in discounts:
        city.robbing_vi(d,eps)
        vfs.append(city.value[city.init_state])
    plt.plot(discounts,vfs)
    plt.title('Value function at (Bank 1, PS) vs discount λ')
    plt.xlabel('λ')
    plt.ylabel('V(initial state)')
    plt.show()
    print("Plot done")
    return


def main():
    city = City()
    city.show()
    print('Getting V vs discount')
    option = input('Select: \n\t(A) to simulate and animate one game, \n\t(B) to simulate 1000 games and return average money made, \n\t(C) to sweep through lambdas separated by 0.1 and return average money made \n\t(D) to plot value funciton against lambda\n')
    if option == 'C':
        T = int(input('Enter time horizon: '))
        for l in range(1,10):
            l = l/10
            print('Calculating policy for lambda = ' + str(l))
            city.robbing_vi(l, 0.1)
            money = 0
            for i in range(1000):
                money += city.sim_policy(T, i, False, False)
            money = money/1000
            print('Average money made at lambda = ' + str(l) + ': ' + str(money))
    elif option == 'D':
        val_func_vs_discount(city,1)
    else:
        lambda_ = float(input('Enter discount factor: '))
        T = int(input('Enter time horizon: '))
        city.robbing_vi(lambda_, 0.1)
        if option == 'A':
            city.sim_policy(T, 0, True, True)
        elif option == 'B':
            money = 0
            for i in range(1000):
                money += city.sim_policy(T, i, False, False)
            money = money/1000
            print('Average money made: ' + str(money))
    print('Done')
    #pdb.set_trace()
    


if __name__=="__main__": main()