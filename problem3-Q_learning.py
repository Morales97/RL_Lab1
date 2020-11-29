import numpy as np
import random as rnd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from winsound import Beep
import pdb


class Town:
    def __init__(self, allow_passing = False):
        '''Initialize rules'''
        self.allow_p = allow_passing
        self.positions = [(i,j) for i in range(4) for j in range(4)]
        self.n_pos = len(self.positions)
        self.n_states = self.n_pos**2
        self.A = (0,0)
        self.B = (1,1)
        self.station = (3,3)
        self.init_state = self.pos_to_state((self.A,
                                             self.station))
        self.actions = [(1,0),(-1,0),(0,1),(0,-1),(0,0)] # D,U,R,L,none
        self.n_actions = len(self.actions)
        self.p_moves = [(1,0),(-1,0),(0,1),(0,-1)] # D,U,R,L
        self.r = self.__gen_rewards() # r[state, action]
        self.trans_p = self.__tp() # trans_p[next_s,curr_s,action]

        for state in range(self.n_states): # just to make sure
            assert (state == self.pos_to_state(self.state_to_pos(state)))
    
    def pos_to_state(self, position_pair):#robber_position, police_position):
        '''Map position pairs to their state number'''
        return (self.n_pos * self.positions.index(position_pair[0]) +
                            self.positions.index(position_pair[1]))
        
    def state_to_pos(self, state):
        '''Map state numbers to their position pair'''
        return (self.positions[state//self.n_pos],
                self.positions[state % self.n_pos])

    def __move(self, position, action):
        '''Return new position if possible, otherwise False'''
        new_p = tuple(x + y for x, y in zip(position, action))
        return new_p if new_p in self.positions else False

    def __gen_rewards(self):
        '''Generate self.r (rewards) as r[state_number, action number]'''
        reward = np.zeros([self.n_states,self.n_actions])
        for s,sp in ((i,j) for i in range(self.n_pos) for j in range(self.n_pos)):
            if s == sp:
                # Every action from touching the police gives reward -10
                reward[self.n_pos*s + sp, :] = -10
            elif self.positions[s] == self.B:
                # Every turn you stay at a bank without the police, get 1
                reward[self.n_pos*s + sp, -1] = 1
        return reward

    def __tp(self):
        '''Generate transition prob tensor'''
        prob = np.zeros([self.n_states,self.n_states,self.n_actions])
        for s,sp in ((i,j) for i in range(self.n_pos) for j in range(self.n_pos)):
            robber_pos, po_pos = self.positions[s], self.positions[sp]
            valid_new_po_pos = []
            for m in self.p_moves:
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
        print('Town initialy:\n'+'█'+' █ '*4 +'█')
        for i in range(4):
            row = '█'
            for j in range(4):
                if (i,j) == self.A: row += ' A '
                elif (i,j) == self.B: row +=' B '
                elif (i,j) == self.station: row+='P_S'
                else: row+='   '
            print(row+'█')
        print('█'+' █ '*4 +'█')
        W = np.zeros((4,4))
        for p in self.positions:
            W[p] = self.r[self.pos_to_state((p,self.station)),-1]
        print(f'Rewards for standing still when police at starting position:\n{W}')

    def show_state(self, state):
        '''Draw field and weights in terminal'''
        print(f'Town in state {state}:\n'+'█'+' █ '*4 +'█')
        robber_pos, po_pos = self.state_to_pos(state)
        for i in range(4):
            row = '█'
            for j in range(4):
                if (i,j) == robber_pos: row += ' R '
                elif (i,j) == self.B: row +=' B '
                elif (i,j) == po_pos: row+=' P '
                else: row+='   '
            print(row+'█')
        print('█'+' █ '*4 +'█')

    def step(self, state, action):
        '''Accept a state idx and action idx, Return new state and reward'''
        assert (state < self.n_states and state >= 0), 'State idx o.o.b.'
        assert (action < self.n_actions and action >= 0), 'Action idx o.o.b.'
        reward = self.r[state, action]
        new_state = rnd.choices(range(self.n_states),
                                weights = self.trans_p[:,state,action])[0]
        return new_state, reward
    
    def sim_policy(self, Q, T, seed, do_plot=False, do_anim=False):
        '''Simulate game for robber following given Q-function'''
        rnd.seed(seed)
        state_seq = [self.init_state]
        money = [0]
        caught = 0
        for t in range(T):
            action = np.argmax(Q[state_seq[-1],:])
            new_state, reward = self.step(state_seq[-1], action)
            if reward < 0: caught += 1
            state_seq.append(new_state)
            money.append(money[-1]+reward)
        print(f'Money at t={T}: {money[-1]}')

        if do_plot or do_anim: # Unpack states
            position_pair_seq = [self.state_to_pos(state) for state in state_seq]
            pathx = [pp[0][0] for pp in position_pair_seq]
            pathy = [pp[0][1] for pp in position_pair_seq]
            po_pathx = [pp[1][0] for pp in position_pair_seq]
            po_pathy = [pp[1][1] for pp in position_pair_seq]
        if do_plot:
            plt.figure(0)
            plt.scatter(*self.B[::-1],s=1000,c = "palegreen",marker="s")
            plt.scatter(*self.station[::-1],s=1000,c = "k",marker="s")
            plt.scatter(po_pathx,po_pathy,c=range(T+1),cmap="autumn",marker='s',s=100)
            plt.scatter(pathx,pathy,c=range(T+1),cmap="winter")
            plt.xticks(range(4))
            plt.yticks(range(4))
            plt.gca().invert_yaxis()
            plt.title(f'Paths in City\n'+ 
                      f'seed = {seed}; t_end = {T}; '+
                      f'max(money) = {int(max(money))}; caught {caught} times')
            plt.show()
            
            plt.figure(1)
            plt.plot(range(T+1),money)
            plt.xticks(range(0,T+1,5))
            plt.xlim([0,T+1])
            plt.title(f'Money vs time for seed = {seed}')
            plt.show()
        if do_anim:
            self.__animate_sim(T,pathx,pathy,po_pathx,po_pathy,seed,money)

    def __animate_sim(self,T,pathx,pathy,po_pathx,po_pathy,seed,money):
        f = plt.figure(0)
        plt.scatter(*self.B[::-1],s=1000,c = "palegreen",marker="s")
        plt.scatter(*self.station[::-1],s=1000,c = "k",marker="s")
        plt.xticks(range(4))
        plt.yticks(range(4))
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
                    #print(f'caught at x={[pathx[t-1],po_pathx[t-1]]}, '+
                    #      f'y={[pathy[t-1],po_pathy[t-1]]}')
                    scat.set_sizes([100,1000])
            else:
                scat.set_sizes([100,200])
            scat.set_offsets(np.c_[x,y])
            plt.title(f'seed = {seed}; t = {t}; money = {int(money[t])}')
            if t == len(pathx) - 1:
                caught = 0
                for i in range(len(money)-1):
                    if money[i+1] < money[i]: caught += 1
                plt.title(f'Paths in Town\n'+ 
                      f'seed = {seed}; t_end = {t}; '+
                      f'money_end = {int(money[t])}; caught {caught} times')
                # plt.savefig(f'{seed}a{t}.png')
            return scat,
        anim = FuncAnimation(f, animationUpdate, frames=T+1, interval=100, blit=False)
        writergif = animation.PillowWriter(fps=4)
        anim.save(f'p3_{T}_{seed}.gif', writer=writergif)
        plt.clf()
        print(f'Saved video as p3_{T}_{seed}.gif')

    def randomize_town(self):
            '''Pick random and update associated data
            Not part of Problem 3 PLEASE IGNORE
            '''
            self.A = rnd.choice(self.positions)
            self.B = rnd.choice(self.positions)
            remaining_pos = [pos for pos in self.positions if pos != self.A]
            # Police must not start at the same position as robbers
            self.station = rnd.choice(remaining_pos)
            self.init_state = self.pos_to_state((self.A,
                                                self.station))
            self.r = self.__gen_rewards()
            self.trans_p = self.__tp() # trans_p[next_s,curr_s,action]


class Learner:
    def __init__(self):
        '''class to hold Q and "convergence values"'''
        self.disc = None
        self.Q = None
        self.saved_its = None
        self.saved_values = None

    def Q_learn(self, env, discount = 0.8, iterations = int(1e7), do_beep = False):
        '''Make the Q matrix by QL and save some values at the initial state'''
        self.disc = discount
        self.Q = np.zeros((env.n_states,env.n_actions))
        n_updates = np.zeros((env.n_states,env.n_actions))
        itsdivby1000 = iterations//1000
        state = env.init_state
        self.saved_its = []
        self.saved_values = []
        if do_beep: Beep(1000,200)
        for i in range(iterations):
            if not i%itsdivby1000:
                print("Progress: " + "█"*(10*i//iterations)+
                      '-'*(10-10*i//iterations) + f'iteration:{i}', end="\r")
                self.saved_its.append(i)
                self.saved_values.append(max(self.Q[env.init_state,:]))
            action = rnd.randrange(env.n_actions)
            new_state, reward = env.step(state, action)

            # Update
            n_updates[state,action] += 1
            alpha = 1/np.power(n_updates[state,action],2/3)
            self.Q[state,action] = ((1-alpha)*self.Q[state,action] +
                alpha * (reward + discount*max(self.Q[new_state,:])))
            state = new_state
        print(f'Q learning done, iterations = {iterations}')
        if do_beep: Beep(2500,200)
        filename = f'QL_d{int(discount*10)}_i1e{int(np.log10(iterations))}.npy'
        np.save(filename, self.Q)
        print(f'Saved Q as {filename}, retrieve with self.Q or np.load')

    def SARSA_learn(self, env, epsilon = 0.1,
                    discount = 0.8, iterations = int(1e7), do_beep = False):
        '''Make the Q matrix by SARSA(ε-greedy) and save some values at the
        initial state'''
        self.disc = discount
        self.Q = np.zeros((env.n_states,env.n_actions))
        n_updates = np.zeros((env.n_states,env.n_actions))
        itsdivby1000 = iterations//1000
        state = env.init_state
        self.saved_its = []
        self.saved_values = []
        if do_beep: Beep(1000,200)
        for i in range(iterations):
            if not i%itsdivby1000:
                print("Progress: " + "█"*(10*i//iterations)+
                      '-'*(10-10*i//iterations) + f'iteration:{i}', end="\r")
                self.saved_its.append(i)
                self.saved_values.append(max(self.Q[env.init_state,:]))
            if rnd.random() < epsilon:
                action = rnd.randrange(env.n_actions)
            else:           
                action = np.argmax(self.Q[state, :])
            new_state, reward = env.step(state, action)
            if rnd.random() < epsilon:
                next_action = rnd.randrange(env.n_actions)
            else:           
                next_action = np.argmax(self.Q[new_state, :])

            # Update
            n_updates[state,action] += 1
            alpha = 1/np.power(n_updates[state,action],2/3)
            self.Q[state,action] = ((1-alpha)*self.Q[state,action] +
                alpha * (reward + discount*self.Q[new_state, next_action]))
            state = new_state
        print(f'SARSA learning done, iterations = {iterations}')
        if do_beep: Beep(2500,200)
        filename = (f'SARSA_e{int(epsilon*10)}_d{int(discount*10)}'+
                    f'_i1e{int(np.log10(iterations))}.npy')
        np.save(filename, self.Q)
        print(f'Saved Q as {filename},\n retrieve with self.Q or np.load({filename})')

    def plot_convergence(self, logscale = False):
        '''Plot saved values vs saved its'''
        if self.saved_its is None: print('learn first')
        else:
            plt.plot(self.saved_its, self.saved_values)
            plt.plot([self.saved_its[0],self.saved_its[-1]],
                     [self.saved_values[-1]]*2)
            if logscale: plt.xscale('log')
            plt.title('Value at initial state vs iteration number')
            plt.xlabel('Iteration number')
            plt.ylabel('V(init_state)')


def test_down_right_wait():
    '''Show what happens if robber does D,R,W in a fresh town'''
    town = Town()
    town.show()
    s = town.init_state
    town.show_state(s)
    for a in [0,2,4]:
        s,r = town.step(s,a)
        print(f'state={s}, action={a}, gave reward={r}')
        town.show_state(s)


def test_down_left_left():
    '''Show what happens if robber does D,L,L in a fresh town'''
    town = Town()
    town.show()
    s = town.init_state
    town.show_state(s)
    for a in [0,3,3]:
        s,r = town.step(s,a)
        print(f'state={s}, action={a}, gave reward={r}')
        town.show_state(s)


def conv_vs_e(epsilons):
    '''Plots and returns "convergence values" for SARSA with different ε'''
    conv_matrix=[]
    town = Town()
    for eps in epsilons:
        learner = Learner()
        learner.SARSA_learn(town, eps, iterations=int(1e7))
        plt.plot(learner.saved_its, learner.saved_values,label=f'ε={eps}')
        conv_matrix.append(learner.saved_values)
    plt.legend()
    plt.show()
    return learner.saved_its, conv_matrix


def main():
    town = Town()
    learner = Learner()
    learner.Q_learn(town)
    pdb.set_trace()


if __name__=='__main__': main()
