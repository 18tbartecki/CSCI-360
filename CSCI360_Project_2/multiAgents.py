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
        # Collect legal moves and child states
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
        This question is not included in project for CSCI360
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return childGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """
    def max_value(self, gameState, depth):
        value = float('-inf')
        minimaxAction = ""
        # Check if terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState), ""

        # Go through actions and successors to find accurate values
        for action in gameState.getLegalActions(0):
            successor = gameState.getNextState(0, action)
            temp = value
            value = max(value, self.min_value(successor, depth, 1))
            if value > temp:
                minimaxAction = action
        return value, minimaxAction

    def min_value(self, gameState, depth, ghost):
        value = float('inf')
        # Check if terminal state then return only value, don't need action
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(ghost):
            successor = gameState.getNextState(ghost, action)
            # If only one ghost or final ghost
            if ghost == gameState.getNumGhost():
                value = min(value, self.max_value(successor, depth + 1)[0])
            # For multiple ghosts use recursion to find temporary values
            else:
                temp_value = self.min_value(successor, depth, ghost+1)
                if value > temp_value:
                    value = temp_value
        return value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumGhost():
        Returns the total number of ghosts in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        depth = 1
        value, action = self.max_value(gameState, depth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """
    def max_value(self, gameState, depth, alpha, beta):
        value = float('-inf')
        minimaxAction = ""
        # Check if terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState), ""

        # Go through actions and successors to find accurate values
        for action in gameState.getLegalActions(0):
            successor = gameState.getNextState(0, action)
            temp = value
            value = max(value, self.min_value(successor, depth, 1, alpha, beta))
            if value > temp:
                minimaxAction = action
            if value > beta:
                return value, minimaxAction
            alpha = max(alpha, value)
        return value, minimaxAction

    def min_value(self, gameState, depth, ghost, alpha, beta):
        value = float('inf')
        # Check if terminal state then return only value, don't need action
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(ghost):
            successor = gameState.getNextState(ghost, action)
            # If only one ghost or final ghost
            if ghost == gameState.getNumGhost():
                value = min(value, self.max_value(successor, depth + 1, alpha, beta)[0])
            # For multiple ghosts use recursion to find temporary values
            else:
                temp_value = self.min_value(successor, depth, ghost+1, alpha, beta)
                if value > temp_value:
                    value = temp_value
            if value < alpha:
                return value
            beta = min(beta, value)
        return value


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        depth = 1
        value, action = self.max_value(gameState, depth, float('-inf'), float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """
    def max_value(self, gameState, depth):
        value = float('-inf')
        expectimaxAction = ""
        # Check if terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState), ""

        # Go through actions and successors to find accurate values
        for action in gameState.getLegalActions(0):
            successor = gameState.getNextState(0, action)
            temp = value
            # Call value function to determine score
            value = max(value, self.value(successor, depth, 0))
            if value > temp:
                expectimaxAction = action
        return value, expectimaxAction

    def exp_value(self, gameState, depth, ghost):
        value = 0

        # Check if terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        actions = len(gameState.getLegalActions(ghost))
        # Go through actions and successors to find accurate values
        for action in gameState.getLegalActions(ghost):
            successor = gameState.getNextState(ghost, action)
            p = 1/actions
            # Multiply score by probability of successor being chosen
            value += p * self.value(successor, depth, ghost)
        return value

    def value(self, successor, depth, ghost):
        actions = len(successor.getLegalActions(ghost))
        if actions == 0:
            return self.evaluationFunction(successor)
        # If final level of ghosts then choose max value next otherwise do another exp
        if ghost == successor.getNumGhost():
            return self.max_value(successor, depth + 1)[0]
        else:
            return self.exp_value(successor, depth, ghost + 1)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        depth = 1
        value, action = self.max_value(gameState, depth)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    score = currentGameState.getScore()

    newGhostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    pacmanPos = currentGameState.getPacmanPosition()

    capsules = currentGameState.getCapsules()
    closestCapsule = 100000
    for capsule in capsules:
        capsuleDistance = util.manhattanDistance(capsule, pacmanPos)
        if capsuleDistance < closestCapsule:
            closestCapsule = capsuleDistance

    food = currentGameState.getFood()
    closestFood = 100000
    for x in range(food.width):
        for y in range(food.height):
            if food[x][y]:
                foodDistance = util.manhattanDistance((x, y), pacmanPos)
                if foodDistance < closestFood:
                    closestFood = foodDistance

    ghosts = currentGameState.getGhostPositions()
    closestGhost = 100000
    for i, ghost in enumerate(ghosts):
        ghostDistance = util.manhattanDistance(ghost, pacmanPos)
        if scaredTimes[i] == 0:
            if ghostDistance < closestGhost:
                closestGhost = ghostDistance

    if closestCapsule != 100000:
        score += closestCapsule*10
    else:
        score += 500
    if closestFood != 100000:
        score += closestFood*10
    else:
        score += 500
    if closestGhost != 100000:
        score -= closestGhost*55
    else:
        score -= 500

    return score



# Abbreviation
better = betterEvaluationFunction
