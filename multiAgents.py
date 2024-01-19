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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 1

        # Use the reciprocal
        foodScore = 1.0 / closestFoodDistance

        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        minimumGhostDistance = min(ghostDistances)

        # If a ghost is right next to you, never consider going that way
        if minimumGhostDistance <= 1:
            return -float('inf')

        closestGhostDistance = min(ghostDistances) if ghostDistances else 1

        # Use the reciprocal
        ghostScore = 1.0 / closestGhostDistance

        return (successorGameState.getScore() + foodScore) - ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            #If agent is pacman
            if agentIndex == 0: 
                bestValue = float('-inf')
                legalActions = state.getLegalActions(agentIndex)

                # Loop through all valid actions for the pacman
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    #Recurse for each agent
                    value = minimax(successorState, depth, agentIndex + 1)
                    bestValue = max(bestValue, value)

                return bestValue
            #If agent is a ghost
            else:
                bestValue = float('inf')
                legalActions = state.getLegalActions(agentIndex)

                # Loop through all valid actions for the ghost
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)

                    #If the agent is the last ghost of this depth
                    if agentIndex == state.getNumAgents() - 1:
                        #Recurse for agents at next depth
                        value = minimax(successorState, depth - 1, 0)
                    # Else there are more ghosts to check
                    else:
                        #Recurse for each next agent
                        value = minimax(successorState, depth, agentIndex + 1)

                    bestValue = min(bestValue, value)
                return bestValue
        
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = minimax(successorState, self.depth, 1)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimaxSearch(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # If agent is pacman
            if agentIndex == 0:
                bestValue = float('-inf')
                legalActions = state.getLegalActions(agentIndex)
                # Loop through all valid actions for the pacman
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    value = expectimaxSearch(successorState, depth, agentIndex + 1)
                    bestValue = max(bestValue, value)

                return bestValue
            # Else agent is a ghost
            else:
                legalActions = state.getLegalActions(agentIndex)
                numActions = len(legalActions)
                totalValue = 0

                # Loop through all valid actions for the ghost
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)

                    #If agent is the final ghost
                    if agentIndex == state.getNumAgents() - 1:
                        #Recurse on next depth
                        value = expectimaxSearch(successorState, depth - 1, 0) 
                    else:
                        #Recurse on next agent
                        value = expectimaxSearch(successorState, depth, agentIndex + 1)

                    # Get the average of the values
                    totalValue += value / numActions
                return totalValue

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = expectimaxSearch(successorState, self.depth, 1)

            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I made this function, so that it takes into account all the different factors for calculating the best direction.
                I use the distance to the closest food as a positive factor, distance to the closest ghost as a negative factor,
                the amount of food left as a positive factor and the amount of time that the ghosts are scared for as a positive factor.
                I took the reciprocal of the closest food distance because when a food is super close (low distance) the reciprocal
                makes the value more positive since the evalution function should be higher for good moves. I also took the reciprocal for 
                the amount of food left because low amounts of food should be better (higher eval score). The closest ghost distance
                is a negative factor because we usually prefer to be further away from the ghost if possible. So, the reciprocal was taken
                so that having smaller distances to ghost will have a larger value and then in the final calculation the value is subtracted.
                Lastly, I prioritized pacman eating the scared ghost because this ensures that the pacman is prioritizing getting a high score.
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Find the distance to the closest food
    closestFoodDistance = min((manhattanDistance(pacmanPosition, foodPos) + 1) for foodPos in foodPositions) if foodPositions else 1
    
    # Get reciprocal
    foodScore = 1.0 / closestFoodDistance

    # Find the distance to the closest ghost
    closestGhostDistance = min((manhattanDistance(pacmanPosition, ghostState.getPosition()) + 1) for ghostState in ghostStates) if ghostStates else 1
    
    # Get reciprocal
    ghostScore = 1.0 / closestGhostDistance

    # Calculate number of foods left to eat
    remainingFood = len(foodPositions)

    # Get reciprocal (ensure's no division by 0 issue)
    foodPelletScore = 1.0 / (remainingFood + 1)

    # Calculates the time that the ghosts are scared
    scaredGhostScore = sum(scaredTime for scaredTime in scaredTimes)

    # Evaluate the state evalution with each factor having a weight
    return currentGameState.getScore() + (10 * foodScore) - (5 * ghostScore) + (100 * foodPelletScore) + (100 * scaredGhostScore)

# Abbreviation
better = betterEvaluationFunction
