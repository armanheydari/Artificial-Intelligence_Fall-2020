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
import random
import util

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        nearestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if nearestGhost < 1.5:
            return -10000000
        if len(newFood.asList()) > 0:
            nearestFood = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        else:
            nearestFood = 10000000
        return successorGameState.getScore() + 10 / nearestFood


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        # util.raiseNotDefined()
        if self.index == 0:
            return self.maxValue(gameState, 1)[0]
        else:
            return self.minValue(gameState, self.depth, 1)[0]

    def maxValue(self, gameState, depth):
        nextState = None
        nextAction = None
        if depth > self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return nextAction, nextState, self.evaluationFunction(gameState)
        else:
            maximumValue = -100000000
            for legalAction in gameState.getLegalActions(0):
                state = gameState.generateSuccessor(0, legalAction)
                temp = self.minValue(state, depth + 1, 1)[2]
                if maximumValue < temp:
                    nextAction = legalAction
                    nextState = state
                    maximumValue = temp
            return nextAction, nextState, maximumValue

    def minValue(self, gameState, depth, ghostIndex):
        nextState = None
        nextAction = None
        if depth > self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return nextAction, nextState, self.evaluationFunction(gameState)
        else:
            minimumValue = 100000000
            for legalAction in gameState.getLegalActions(ghostIndex):
                state = gameState.generateSuccessor(ghostIndex, legalAction)
                if ghostIndex < gameState.getNumAgents() - 1:
                    temp = self.minValue(state, depth + 1, ghostIndex + 1)[2]
                else:
                    temp = self.maxValue(state, depth + 1)[2]
                if minimumValue > temp:
                    nextAction = legalAction
                    nextState = state
                    minimumValue = temp
            return nextAction, nextState, minimumValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.index == 0:
            return self.maxValue(gameState, 1, -100000000, 100000000)[0]
        else:
            return self.minValue(gameState, self.depth, 1, -100000000, 100000000)[0]

    def maxValue(self, gameState, depth, alpha, beta):
        nextState = None
        nextAction = None
        if depth > self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return nextAction, nextState, self.evaluationFunction(gameState)
        else:
            maximumValue = -100000000
            for legalAction in gameState.getLegalActions(0):
                state = gameState.generateSuccessor(0, legalAction)
                temp = self.minValue(state, depth + 1, 1, alpha, beta)[2]
                if maximumValue < temp:
                    nextAction = legalAction
                    nextState = state
                    maximumValue = temp
                alpha = max(alpha, maximumValue)
                if beta < alpha:
                    break
            return nextAction, nextState, maximumValue

    def minValue(self, gameState, depth, ghostIndex, alpha, beta):
        nextState = None
        nextAction = None
        if depth > self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return nextAction, nextState, self.evaluationFunction(gameState)
        else:
            minimumValue = 100000000
            for legalAction in gameState.getLegalActions(ghostIndex):
                state = gameState.generateSuccessor(ghostIndex, legalAction)
                if ghostIndex < gameState.getNumAgents() - 1:
                    temp = self.minValue(state, depth + 1, ghostIndex + 1, alpha, beta)[2]
                else:
                    temp = self.maxValue(state, depth + 1, alpha, beta)[2]
                if minimumValue > temp:
                    nextAction = legalAction
                    nextState = state
                    minimumValue = temp
                beta = min(beta, minimumValue)
                if beta < alpha:
                    break
            return nextAction, nextState, minimumValue


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
        return self.maxValue(gameState, 1)[0]

    def maxValue(self, gameState, depth):
        nextState = None
        nextAction = None
        if depth > self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return nextAction, nextState, self.evaluationFunction(gameState)
        else:
            maximumValue = -100000000
            for legalAction in gameState.getLegalActions(0):
                state = gameState.generateSuccessor(0, legalAction)
                temp = self.expectedValue(state, depth + 1, 1)
                if maximumValue < temp:
                    nextAction = legalAction
                    nextState = state
                    maximumValue = temp
            return nextAction, nextState, maximumValue

    def expectedValue(self, gameState, depth, ghostIndex):
        if depth > self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        else:
            expectingValue = 0
            for legalAction in gameState.getLegalActions(ghostIndex):
                state = gameState.generateSuccessor(ghostIndex, legalAction)
                if ghostIndex < gameState.getNumAgents() - 1:
                    temp = self.expectedValue(state, depth + 1, ghostIndex + 1)
                else:
                    temp = self.maxValue(state, depth + 1)[2]
                expectingValue += temp
            return expectingValue / len(gameState.getLegalActions(ghostIndex))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    nearestFood = 0
    if len(foodList) > 0:
        nearestFood = min([manhattanDistance(newPos, food) for food in foodList])
    return currentGameState.getScore() - nearestFood


# Abbreviation
better = betterEvaluationFunction
