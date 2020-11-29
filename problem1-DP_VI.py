# EL2805 - Reinforcement Learning - KTH
# Authors: 
#   - Valentin Minoz, 981114-4097
#   - Daniel Morales, 971203-T393
# November 2020
# ----------------------------------
# ------------- Lab 1 --------------
# ----------------------------------
# ------------ PROBLEM 1 -----------
# ----------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time
from IPython import display
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1

    def __init__(self, maze, minotaur_stay=False, cross_minotaur=False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                       = np.transpose(maze);
        self.minotaur_stay              = minotaur_stay;        # Bool: Allow minotaur to take action STAY
        self.cross_minotaur             = cross_minotaur;       # Bool: Allow player to move into the minotaurs cell
        self.actions_p, self.actions_m  = self.__actions();
        self.states, self.map           = self.__states();
        self.n_actions                  = len(self.actions_p);
        self.n_states                   = len(self.states);
        self.transition_probabilities   = self.__transitions();
        self.rewards                    = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
        self.seed                       = 0                     # To save gifs with different names

    def __actions(self):
        actions_p = dict();
        actions_p[self.MOVE_LEFT]  = (0,-1);
        actions_p[self.MOVE_RIGHT] = (0, 1);
        actions_p[self.MOVE_UP]    = (-1,0);
        actions_p[self.MOVE_DOWN]  = (1,0);

        actions_m = actions_p.copy()
        actions_p[self.STAY]       = (0, 0);
        if self.minotaur_stay:
            actions_m[self.STAY]   = (0, 0);

        return actions_p, actions_m;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            states[s] = (i,j,k,l);
                            map[(i,j,k,l)] = s;
                            s += 1;
        return states, map

    def __move_player(self, state, action):
        "Returns next position of the player given state an action"
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions_p[action][0];
        col = self.states[state][1] + self.actions_p[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Cannot walk towards minotaur
        # If next action is impossible (hit wall or outside boundaries), STAY
        if hitting_maze_walls or ((row,col) == (self.states[state][2], self.states[state][3]) and not self.cross_minotaur) :
            return (self.states[state][0], self.states[state][1]);
        else:
            return (row, col);

    def __move_minotaur(self, state):
        " Returns list of next possible positions of the minotaur given current state "
        possible_pos = list()
        for action in self.actions_m:
            row = self.states[state][2] + self.actions_m[action][0];
            col = self.states[state][3] + self.actions_m[action][1];
            if row in range(self.maze.shape[0]) and col in range(self.maze.shape[1]):
                possible_pos.append((row,col))
        return possible_pos

    def __move(self, state, action):
        """ Return next state given current state and action. 
            Note that next state is RANDOM """
        next_pos_player = self.__move_player(state, action);
        next_pos_minotaur = rnd.choice(self.__move_minotaur(state));
        return self.map[(next_pos_player[0], next_pos_player[1], next_pos_minotaur[0], next_pos_minotaur[1])];

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. 
        # The next position of the player given an action is deterministic
        # The next position of the minotaur is random
        for s in range(self.n_states):
            player_state    = (self.states[s][0], self.states[s][1])
            minotaur_state  = (self.states[s][2], self.states[s][3])
            # If minotaur eats player, the game is over. 
            # If the player reaches the exit, the game is over.
            # We will consider the last state as final
            if self.maze[player_state] == 2 or player_state == minotaur_state:
                transition_probabilities[s,s,:] = 1
            else:
                for a in self.actions_p:
                    next_pos_player = self.__move_player(s,a);
                    list_pos_minotaur = self.__move_minotaur(s);
                    p = len(list_pos_minotaur);
                    for next_pos_minotaur in list_pos_minotaur:
                        next_s = self.map[(next_pos_player[0], next_pos_player[1], next_pos_minotaur[0], next_pos_minotaur[1])]
                        transition_probabilities[next_s, s, a] = 1/p;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):
        "Return rewards(s,a) matrix"

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            # If minotaur eats player, reward is 0 (this is necessary to specify explicitly)
            state = self.states[s]
            if (state[0], state[1]) == (state[2], state[3]):
                rewards[s,:] = 0
            else:
                for a in self.actions_p:
                    next_s = self.__move(s,a);
                    current_pos_player  = (self.states[s][0], self.states[s][1])
                    next_pos_player     = (self.states[next_s][0], self.states[next_s][1])
                    # Reward for reaching the exit. If player already exited the maze, do not give reward, only when first reaching the maze
                    if self.maze[next_pos_player] == 2 and next_pos_player != current_pos_player:
                        rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD;
        return rewards;

    def simulate(self, start, policy, method):
        rnd.seed(self.seed)
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions_p)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

    def init_state(self):
        return self.map[(0,0,5,6)]

    def tensor_check(self):
        " Sanity check for transition probability tensor "
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if sum(self.transition_probabilities[:,s,a]) != 1.0:
                    print('error')

    def simulate(self, start, policy, method, life_mean):
        " Simulate a game. Return path, whether or not the player exited the maze, and duration t "
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        exited_maze = False
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
                # Check if exited the maze
                if self.maze[self.states[next_s][0], self.states[next_s][1]] == 2:
                    exited_maze = True;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            p = 1 / life_mean;
            T = np.random.geometric(p)
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while not exited_maze and t < T:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                # Check if exited the maze
                if self.maze[self.states[next_s][0], self.states[next_s][1]] == 2:
                    exited_maze = True;
        return path, exited_maze, t

    def simulate_many(self, start, policy, method, life_mean, N):
        " Run N simulations. Return average win rate "
        exit_count = 0;
        average_t_exit = 0;
        average_t_death = 0;
        average_t = 0;
        longest_t = 0;
        longest_path = list();
        for n in range(N):
            path, exited_maze, t = self.simulate(start, policy, method, life_mean);
            if exited_maze:
                exit_count += 1;
                average_t_exit += t;
                if t > longest_t:
                    longest_t = t;
                    longest_path = path;
            else:
                average_t_death += t;
        average_t = average_t_death + average_t_exit;
        return exit_count/N, average_t_exit/exit_count, average_t_death/(N-exit_count), average_t/N, longest_t, longest_path

    def animate_sim(self, path, exited_maze, T):
        " Animate a game and save as a GIF"
        pathx = [path[i][0] for i in range(len(path))]
        pathy = [path[i][1] for i in range(len(path))]
        mino_pathx = [path[i][2] for i in range(len(path))]
        mino_pathy = [path[i][3] for i in range(len(path))]

        wall = [(6,4)]
        for i in range(4):
            wall.append((i,2))
        for i in range(3):
            wall.append((i+1,5))
            wall.append((2,5+i))
        for j in range(1,7):
            wall.append((5,j))

        A = (0,0)
        B = (6,5)

        f = plt.figure(0)
        plt.scatter(*zip(*[(s[1],s[0]) for s in wall]),1000,c = "r",marker="x")
        plt.scatter(*A[::-1],c=0,cmap='winter',s=300,marker='o')
        plt.scatter(*B[::-1],c='palegreen',s=300,marker='D')
        plt.gca().invert_yaxis()
        
        scat = plt.scatter(x=[pathx[0],mino_pathx[0]],
                    y=[pathy[0],mino_pathy[0]],
                    c=['b' ,'r'],s=[100,200],zorder=100)
        def animationUpdate(t):
            '''anim update function'''
            plt.plot(mino_pathx[:t+1],mino_pathy[:t+1],c='r',linewidth=3)
            plt.plot(pathx[:t+1],pathy[:t+1],c='b')
            x = [pathx[t],mino_pathx[t]]
            y = [pathy[t],mino_pathy[t]]
            scat.set_offsets(np.c_[x,y])
            #plt.title(f'seed = {seed}; t = {t}; Win = {not dead}')
            # plt.savefig(f'{seed}a{t}.png')
            return scat,
        anim = FuncAnimation(f, animationUpdate, frames=T+1, interval=100, blit=False)

        writergif = animation.PillowWriter(fps=5) 
        anim.save(f'p1_{self.seed}.gif', writer=writergif)
        plt.clf()
        print(f'Saved video as p1_{self.seed}.gif')
        self.seed += 1;


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    print("delta: " + str(np.linalg.norm(V - BV)))
    print("#iterations: " + str(n))
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Paths
    path_p  = [(path[i][1],path[i][0]) for i in range(len(path))]
    path_m  = [(path[i][3],path[i][2]) for i in range(len(path))]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path_p)):
        grid.get_celld()[(path_p[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path_p[i])].get_text().set_text('Player')
        grid.get_celld()[(path_m[i])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path_m[i])].get_text().set_text('Minotaur')
        if i > 0:
            if path_p[i] == (5,6):
                grid.get_celld()[(path_p[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path_p[i])].get_text().set_text('Player is out')
            if path_p[i-1] != path_m[i] and path_p[i-1] != path_p[i]:
                grid.get_celld()[(path_p[i-1])].set_facecolor(col_map[maze[path_p[i-1]]])
                grid.get_celld()[(path_p[i-1])].get_text().set_text('')
            if path_m[i-1] != path_p[i] and path_m[i-1] != path_m[i]:
                grid.get_celld()[(path_m[i-1])].set_facecolor(col_map[maze[path_m[i-1]]])
                grid.get_celld()[(path_m[i-1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)

def __main__():
    # Init maze
    maze = np.array([
                [0,0,1,0,0,0,0,0],
                [0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,1,1],
                [0,0,1,0,0,1,0,0],
                [0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,0],
                [0,0,0,0,1,2,0,0]
                ])
    # Init environment. Select minotaur can stay for more interesting results
    env = Maze(maze, minotaur_stay=True)    

    method = input('b for (b) Dynamic Programming \nc for (c) Value Iteration\nInput: ')

    if method == 'b':
        T = 20
        print(f'Getting policy for T = {T}')
        V, policy = dynamic_programming(env,T)
        print(f'Policy ready. P(win) = {V[env.init_state(),:]}')
        run = True
        while run:
            run = input('-- Enter anything to simulate a game, or just press enter to exit--')
            if run:
                path, exited, t = env.simulate((0,0,5,6), policy, 'DynProg', T)
                env.animate_sim(path, exited, t)
    if method == 'c':
        gamma = 1 - 1/30
        epsilon = 0.001
        print(f'Getting policy for gamma = {gamma} and epsilon = {epsilon}')
        V, policy = value_iteration(env, gamma, epsilon)
        run = True
        while run:
            run = input('-- Enter anything to simulate a game, or just press enter to exit--')
            if run:
                path, exited, t = env.simulate((0,0,5,6), policy, 'ValIter', 30)
                env.animate_sim(path, exited, t)

if __name__ == '__main__': __main__()

