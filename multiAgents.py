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

    def getAction(self, gameState):

        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return 999999

        foodList = newFood.asList()
        from util import manhattanDistance
        foodDistance = [0]
        for pos in foodList:
            foodDistance.append(manhattanDistance(newPos, pos))

        ghostPos = []
        for ghost in newGhostStates:
            ghostPos.append(ghost.getPosition())

        ghostDistance = []
        for pos in ghostPos:
            ghostDistance.append(manhattanDistance(newPos, pos))

        ghostPosCurrent = []
        for ghost in currentGameState.getGhostStates():
            ghostPosCurrent.append(ghost.getPosition())

        ghostDistanceCurrent = []
        for pos in ghostPosCurrent:
            ghostDistanceCurrent.append(manhattanDistance(newPos, pos))

        score = 0

        numberOfFoodLeft = len(foodList)

        numberOfFoodLeftCurrent = len(currentGameState.getFood().asList())

        numberofPowerPellets = len(successorGameState.getCapsules())

        sumScaredTimes = sum(newScaredTimes)

        score += successorGameState.getScore() - currentGameState.getScore()

        if action == Directions.STOP:
            score -= 10

        if newPos in currentGameState.getCapsules():
            score += 15 * numberofPowerPellets

        if numberOfFoodLeft < numberOfFoodLeftCurrent:
            score += 20

        score -= 10 * numberOfFoodLeft

        if sumScaredTimes > 0:
            if min(ghostDistanceCurrent) < min(ghostDistance):
                score += 20
            else:
                score -= 10

        else:
            if min(ghostDistanceCurrent) < min(ghostDistance):
                score -= 10
            else:
                score += 20

        return score


def scoreEvaluationFunction(currentGameState):

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        # util.raiseNotDefined()
        actions = gameState.getLegalActions(0)
        maxResult = float('-inf')
        actions.remove(Directions.STOP)
        for a in actions:
            successor = gameState.generateSuccessor(0, a)
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
            return min([self.minValue(s, currDepth, currAgent + 1) for s in successors])

        else:
            return min([self.maxValue(s, currDepth + 1) for s in successors])

    def maxValue(self, gameState, currDepth):

        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)

        successors = []

        for a in actions:
            successors.append(gameState.generateSuccessor(0, a))

        return max([self.minValue(s, currDepth, 1) for s in successors])


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def AB(gameState,agent,depth,a,b):

            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            if agent == gameState.getNumAgents() - 1:
                depth += 1

            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):

                if not result:
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)
                    result.append(nextValue[0])
                    result.append(action)

                    if agent == self.index:
                        a = max(result[0],a)
                    else:
                        b = min(result[0],b)

                else:
                    if result[0] > b and agent == self.index:
                        return result

                    if result[0] < a and agent != self.index:
                        return result

                    previousValue = result[0]
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            a = max(result[0],a)

                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            b = min(result[0],b)
            return result

        return AB(gameState,self.index,0,-float("inf"),float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        def expectiMax(gameState,agent,depth):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            if agent == gameState.getNumAgents() - 1:
                depth += 1

            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    if(agent != self.index):
                        result.append((1.0 / len(gameState.getLegalActions(agent))) * nextValue[0])
                        result.append(action)

                    else:
                        result.append(nextValue[0])
                        result.append(action)
                else:
                    previousValue = result[0]
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action

                    else:
                        result[0] = result[0] + (1.0 / len(gameState.getLegalActions(agent))) * nextValue[0]
                        result[1] = action
            return result

        return expectiMax(gameState,self.index,0)[1]


def betterEvaluationFunction(currentGameState):

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = []

    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    ghostPos = currentGameState.getGhostStates()
    ghostDistance = []

    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos, pos.getPosition()))

    score = 0
    numberOfFoodLeft = len(foodList)
    numberOfFoodLeftCurrent = len(currentGameState.getFood().asList())
    numberofPowerPellets = len(currentGameState.getCapsules())
    sumScaredTimes = sum(newScaredTimes)

    score += currentGameState.getScore()

    if sumScaredTimes > 0:
        if 3 > min(ghostDistance):
            score += 20
        else:
            score -= 10

    else:
        if 5 < min(ghostDistance):
            score -= 10

    if len(foodDistance):
        return score + (1 / (min(foodDistance)**2))

    return score