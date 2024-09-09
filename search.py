# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import random
import util
from util import Stack
from util import Queue
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# DANIELA 
def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    actual = problem.getStartState()
    visitados = {actual:["",""]}
    pila, final = Stack(), []
    for x in problem.getSuccessors(actual): pila.push([x,actual])
    while not pila.isEmpty() and not problem.isGoalState(actual):
        next = pila.pop()
        if not (next[0][0] in visitados):
            visitados[next[0][0]] = [next[0][1],next[1]]
            actual = next[0][0]
            if not problem.isGoalState(actual):
                for x in problem.getSuccessors(actual):pila.push([x,actual])
    padre = visitados[actual][1]
    final.append(visitados[actual][0])
    while visitados[padre][0] != "":
        final.append(visitados[padre][0])
        padre = visitados[padre][1]
    final.reverse()
    return final
    ### util.raiseNotDefined()

#JOAN
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    Cola = Queue()   
    pathToCurrent=Queue()                       
    
    visitados = []                            
    tempPath=[]                            
    path=[]                                 
    
    Cola.push(problem.getStartState())
    currState = Cola.pop()
    
    while not problem.isGoalState(currState):
        if currState not in visitados:
            visitados.append(currState)    
            successors = problem.getSuccessors(currState)
            for child,direction,cost in successors:
                Cola.push(child)
                tempPath = path + [direction]
                pathToCurrent.push(tempPath)
        currState = Cola.pop()
        path = pathToCurrent.pop()
        
    return path
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()
    start_state = problem.getStartState()
    frontier.push(([], 0, start_state), 0)
    visited = set()
    while not frontier.isEmpty():
        path, current_cost, current_state = frontier.pop()
        if current_state in visited:
            continue
        visited.add(current_state)
        if problem.isGoalState(current_state):
            return path
        for successor, action, step_cost in problem.getSuccessors(current_state):
            if successor not in visited:
                new_cost = current_cost + step_cost
                new_path = path + [action]
                frontier.push((new_path, new_cost, successor), new_cost)
    return []
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# DANIELA 
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Prioridad = costo + heuristica
    frontier = util.PriorityQueue()
    visited = {}
    startState = problem.getStartState()
    frontier.push((startState, [], 0), heuristic(startState, problem))
    while not frontier.isEmpty():
        currState, path, overallCost = frontier.pop()  # Pop low prioridad
        if currState in visited and visited[currState] <= overallCost: # skip si ya visitado
            continue
        visited[currState] = overallCost # visitado
        if problem.isGoalState(currState): # si se alcanza el objetivo, retorna
            return path
        for successor, action, nextCost in problem.getSuccessors(currState): # Sucesores
            newCost = overallCost + nextCost
            if successor not in visited or newCost < visited[successor]:
                new_path = path + [action]
                priority = newCost + heuristic(successor, problem)
                frontier.push((successor, new_path, newCost), priority)
    return []
    ## util.raiseNotDefined()


#GPT ASSITST PROMPT: I have the following class that i use to solve paths in a pacman game. 
# [shows the code above] I want to include a Genetic Search Path, could you implement it? 
def geneticSearch(problem: SearchProblem, population_size=100, generations=1000, mutation_rate=0.1):
    """
    Genetic algorithm to find a path in the search problem.
    
    Args:
        problem (SearchProblem): The problem to solve.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to run.
        mutation_rate (float): Probability of mutation.

    Returns:
        List of actions representing the path from start to goal.
    """
    def generate_individual():
        """Generate a random individual (sequence of actions)."""
        actions = []
        current_state = problem.getStartState()
        for _ in range(20):  # Limiting to a max to prevent infinite loops
            if problem.isGoalState(current_state):
                break
            successors = problem.getSuccessors(current_state)
            if not successors:
                break
            action = random.choice(successors)[1]
            actions.append(action)
            current_state = successors[0][0]
        return actions

    def fitness(individual):
        """Calculate the fitness of an individual."""
        current_state = problem.getStartState()
        cost = 0
        for action in individual:
            successors = problem.getSuccessors(current_state)
            for successor, succ_action, step_cost in successors:
                if succ_action == action:
                    current_state = successor
                    cost += step_cost
                    break
        if problem.isGoalState(current_state):
            return 1 / (1 + cost)  # Higher fitness for shorter paths
        return 1 / (1 + cost + 1000)  # Penalize paths that don't reach the goal

    def crossover(parent1, parent2):
        """Crossover between two parents."""
        crossover_point = random.randint(0, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(individual):
        """Mutate an individual."""
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(individual) - 1)
            individual[mutation_point] = random.choice(
                [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST])
        return individual

    # Initialize population
    population = [generate_individual() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        population = sorted(population, key=lambda ind: -fitness(ind))
        if fitness(population[0]) == 1:
            break  # Found an optimal solution

        # Selection: Select the top individuals to breed
        breeding_pool = population[:population_size // 2]
        # Crossover and Mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(breeding_pool, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    # Return the best individual
    best_individual = max(population, key=lambda ind: fitness(ind))
    return best_individual

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
gen = geneticSearch
