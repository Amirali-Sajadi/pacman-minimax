# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    # """
    # def __init__(self, evalFn = 'amberBetterevalFunc', depth = '2'):
    #     self.index = 0 # Pacman is always agent index 0
    #     self.evaluationFunction = util.lookup(evalFn, globals())
    #     self.depth = int(depth)
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        """
            miniMax: receives state, agent(0,1,2...) and current depth
            miniMax: returns a list: [cost,action]
            Example with depth: 3
            That means pacman played 3 times and all ghosts played 3 times
        """

        # util.raiseNotDefined()
        # MAX plays first
        actions = gameState.getLegalActions(0)
        maxResult = float('-inf')
        actions.remove(Directions.STOP)
        for a in actions:
            successor = gameState.generateSuccessor(0, a)
            # Start with depth = 0 and the agent index = 1 (first ghost)
            currentResult = self.minValue(successor, 0, 1)
            if currentResult > maxResult:
                maxResult = currentResult
                maxAction = a
        return maxAction

    def minValue(self, gameState, currDepth, currAgent):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(currAgent)
        successors = []
        for a in actions:
            successors.append(gameState.generateSuccessor(currAgent, a))
        agents = gameState.getNumAgents()
        if currAgent < agents - 1:
            # There are still some ghosts to choose their moves, so increase agent index and call minValue again
            return min([self.minValue(s, currDepth, currAgent + 1) for s in successors])
        else:
            # Depth is increased when it is MAX's turn
            return min([self.maxValue(s, currDepth + 1) for s in successors])

    def maxValue(self, gameState, currDepth):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        successors = []
        for a in actions:
            successors.append(gameState.generateSuccessor(0, a))

        # Agent with index == 1 (the first ghost) plays next
        return max([self.minValue(s, currDepth, 1) for s in successors])



# class MinimaxAgent(MultiAgentSearchAgent):
#
#     def init(self, depth = '1'):
#         self.index = 0 # Pacman is always agent index 0
#         self.evaluationFunction = betterEvaluationFunction
#         self.depth = int(depth)
#         self.BEST_ACTION = None
#
#
#     def minimax(self, depth, state, player):
#         if depth == 0 or state.isWin() or state.isLose():
#             return self.evaluationFunction(state)
#
#         if player == 0:
#             v = float('-inf')
#             legal_actions = state.getLegalActions(0)
#             # print(legal_actions)
#
#         for action in legal_actions:
#             next_state = state.generateSuccessor(0, action)
#             new_v = self.minimax(depth, next_state, player + 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """
            AB: receives a state an agent(0,1,2...) and current depth
            AB: return a list:[cost,action]
            Example with depth: 3
            That means pacman played 3 times and all ghosts 3 times
            AB: Cuts some nodes. That means we can use higher depth in the same
            time of minimax algorithm in lower depth
        """

        def AB(gameState,agent,depth,a,b):
            result = []

            # Terminate state #
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            # Reached max depth #
            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth #
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # Calculate nextAgent #

            # Last ghost: nextAgent = pacman #
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            # Availiable ghosts. Pick next ghost #
            else:
                nextAgent = agent + 1

            # For every successor find minmax value #
            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    # Fix result #
                    result.append(nextValue[0])
                    result.append(action)

                    # Fix initial a,b (for the first node) #
                    if agent == self.index:
                        a = max(result[0],a)
                    else:
                        b = min(result[0],b)
                else:
                    # Check if minMax value is better than the previous one #
                    # Chech if we can overpass some nodes                   #

                    # There is no need to search next nodes                 #
                    # AB Prunning is true                                   #
                    if result[0] > b and agent == self.index:
                        return result

                    if result[0] < a and agent != self.index:
                        return result

                    previousValue = result[0] # Keep previous value
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    # Max agent: Pacman #
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # a may change #
                            a = max(result[0],a)

                    # Min agent: Ghost #
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # b may change #
                            b = min(result[0],b)
            return result

        # Call AB with initial depth = 0 and -inf and inf(a,b) values      #
        # Get an action                                                    #
        # Pacman plays first -> self.index                                 #
        return AB(gameState,self.index,0,-float("inf"),float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        """
        expectiMax: receives state, agent(0,1,2...) and current depth
        expectiMax: returns a list: [cost,action]
        Example with depth: 3
        That means pacman played 3 times and all ghosts played 3 times
        Now ghosts move randomly. The difference with minimax is that
        ghosts may not play the best move. We can win more easily.
        This is close to our world, as no one playes optimal every time
        We can imagine chance nodes like min nodes. But their value is equal
        with the sum: (moveValue * probability) for every move
        """

        def expectiMax(gameState,agent,depth):
            result = []

            # Terminate state #
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            # Reached max depth #
            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth(last ghost) #
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # Calculate nextAgent #

            # Last ghost: nextAgent = pacman #
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            # Availiable ghosts. Pick next ghost #
            else:
                nextAgent = agent + 1

            # For every successor find minimax value #
            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)
                    # Fix chance node                               #
                    # Probability: 1 / p -> 1 / total legal actions #
                    # Ghost pick an action based in 1 / p. As all   #
                    # actions have the same probability             #
                    if(agent != self.index):
                        result.append((1.0 / len(gameState.getLegalActions(agent))) * nextValue[0])
                        result.append(action)
                    else:
                        # Fix result with minimax value and action #
                        result.append(nextValue[0])
                        result.append(action)
                else:

                    # Check if miniMax value is better than the previous one #
                    previousValue = result[0] # Keep previous value. Minimax
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    # Max agent: Pacman #
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action

                    # Min agent: Ghost                                         #
                    # Now we don't select a better action but we continue to   #
                    # calculate our sum to find the total value of chance node #
                    else:
                        result[0] = result[0] + (1.0 / len(gameState.getLegalActions(agent))) * nextValue[0]
                        result[1] = action
            return result

        # Call expectiMax with initial depth = 0 and get an action  #
        # Pacman plays first -> agent == 0 or self.index            #
        # We can will more likely than minimax. Ghosts may not play #
        # optimal in some cases                                     #

        return expectiMax(gameState,self.index,0)[1]



def nearest_food_distance(state):
    state.getFood()
    walls = state.getWalls()
    row, col = 0, 0
    for i in walls:
        for j in walls[0]:
            col += 1
            row += 1

    def is_in_bounds(i, j):
        if 0 < i < row and 0 < j < col:
            return True
        else:
            return False

    pac_position = state.getPacmanPosition()
    visited = set()
    queue = util.Queue()
    queue.push([pac_position, 0])
    while not queue.isEmpty():
        temp_position = queue.pop()
        x, y = temp_position[0]

        if state.hasFood(x, y):
            return temp_position[1]

        if temp_position[0] in visited:
            continue

        visited.add(temp_position[0])

        x, y = temp_position[0]
        if not walls[x - 1][y] and is_in_bounds(x - 1, y):
            queue.push([(x - 1, y), temp_position[1] + 1])
        if not walls[x + 1][y] and is_in_bounds(x + 1, y):
            queue.push([(x + 1, y), temp_position[1] + 1])
        if not walls[x][y - 1] and is_in_bounds(x, y - 1):
            queue.push([(x, y - 1), temp_position[1] + 1])
        if not walls[x][y + 1] and is_in_bounds(x, y + 1):
            queue.push([(x, y + 1), temp_position[1] + 1])

        return float('inf')





# def betterEvaluationFunction1(currentGameState):
#
#     score = 0
#     pac_pos = currentGameState.getPacmanPosition()
#     food_remain = currentGameState.getNumFood()
#     ghost_states = currentGameState.getGhostStates()
#     ghost_distance = 0
#
#
#     if currentGameState.isWin():
#         return currentGameState.getScore() + 10000
#     if currentGameState.isLose():
#         return -10000
#
#     score += currentGameState.getScore() / 2
#
#     score -= 100 * food_remain
#
#     score += 10/nearest_food_distance(currentGameState)
#
#     for ghost in ghost_states:
#         d = manhattanDistance(ghost.getPosition(), pac_pos)
#         ghost_distance += d
#         if d < 3:
#             score -= d * 10
#
#     return score + currentGameState.getScore()
#



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # If successor state is a win state return very high score.
    # if currentGameState.isWin():
    #     return 999999

    """ Manhattan distance to the available foods from the successor state """
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = []
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    """ Manhattan distance to each ghost in the game from successor state"""
    ghostPos = currentGameState.getGhostStates()

    ghostDistance = []
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos, pos.getPosition()))

    # """ Manhattan distance to each ghost in the game from current state"""
    # ghostPosCurrent = []
    # for ghost in currentGameState.getGhostStates():
    #     ghostPosCurrent.append(ghost.getPosition())

    # ghostDistanceCurrent = []
    # for pos in ghostPosCurrent:
    #     ghostDistanceCurrent.append(manhattanDistance(newPos, pos))

    score = 0
    # Get Number of food available in successor state
    numberOfFoodLeft = len(foodList)
    # Get Number of food available in current state
    numberOfFoodLeftCurrent = len(currentGameState.getFood().asList())
    # Get Number of Power Pellets available in successor state
    numberofPowerPellets = len(currentGameState.getCapsules())
    # Get state of ghosts in successor state
    sumScaredTimes = sum(newScaredTimes)

    # Relative Score
    score += currentGameState.getScore()
    # if action == Directions.STOP:
    #     # Penalty for stop
    #     score -= 10000

    # Add Score if pacman eats power pellet in next state.
    # if newPos in currentGameState.getCapsules():
    #     score += 150 * numberofPowerPellets
    # # Add score if there are lesser number of food available in successor state.
    # if numberOfFoodLeft < numberOfFoodLeftCurrent:
    #     score += 2000

    # For each food left subtract 10 score.
    # score -= 10 * numberOfFoodLeft

    # If ghosts are scared lesser distance to ghosts is better.
    if sumScaredTimes > 0:
        if 3 > min(ghostDistance):
            score += 20
        else:
            score -= 10
    # If ghosts are not scared greater distance to ghosts is better.
    else:
        if 5 < min(ghostDistance):
            score -= 10
        # else:
        #     score += 20
    if len(foodDistance):
        return score + (1 / (min(foodDistance)**2))
    return score

# Abbreviation
# better = betterEvaluationFunction
