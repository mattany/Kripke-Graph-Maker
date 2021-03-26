import copy
from collections import deque
from typing import List, NamedTuple, Tuple, Set
import numpy as np
import networkx as nx
from graphviz import Source

TURN = 0

VARIABLES_AMT = 5


class State(NamedTuple):
    turn: bool = False
    c0: bool = False
    c1: bool = False
    pc0: int = -1
    pc1: int = -1

    def __repr__(self):
        return f"turn:{int(self.turn)}, c0:{int(self.c0)}, c1:{int(self.c1)}\n, pc0:{self.pc0}, pc1:{self.pc1}\n\n"


class Transition(NamedTuple):
    origin: State
    destination: State


def same(item):
    return item


def increment(item):
    return item + 1


def zero(item):
    return 0


def one(item):
    return 1


class Rule:
    def __init__(self, requirements, actions):
        assert len(requirements) == VARIABLES_AMT
        assert len(actions) == VARIABLES_AMT
        self.requirements = requirements
        self.actions = actions

    @classmethod
    def fieldSatisfiesRequirement(cls, field, requirement):
        return requirement is None or field == requirement

    def stateSatisfiesRequirements(self, state):
        for i, requirement in enumerate(self.requirements):
            field = state[i]
            if not self.fieldSatisfiesRequirement(field, requirement):
                return False
        return True

    def getTransition(self, state: State):
        """
        :param state:
        :return: Transition from state to next state according to rule if state satisfies rule requirements,
        otherwise false.
        """
        if self.stateSatisfiesRequirements(state):
            variables = list()
            for i, action in enumerate(self.actions):
                variables.append(action(state[i]))
            return Transition(state, State(*variables))
        return False


rules = {
    # Start
    Rule([None, None, None, -1, None], [same, same, same, increment, same]),
    Rule([None, None, None, None, -1], [same, same, same, same, increment]),

    # while true do
    Rule([None, None, None, 0, None], [same, same, same, increment, same]),
    Rule([None, None, None, None, 0], [same, same, same, same, increment]),

    # ci = 0
    Rule([None, None, None, 1, None], [same, zero, same, increment, same]),
    Rule([None, None, None, None, 1], [same, same, zero, same, increment]),

    # while c!i=0 do
    # c!i == 0
    Rule([None, None, 0, 2, None], [same, same, same, increment, same]),
    Rule([None, 0, None, None, 2], [same, same, same, same, increment]),
    # c!i == 1
    Rule([None, None, 1, 2, None], [same, zero, same, lambda x: 7, same]),
    Rule([None, 1, None, None, 2], [same, same, zero, same, lambda x: 7]),

    # if turn=!i
    # turn == !i
    Rule([1, None, None, 3, None], [same, same, same, increment, same]),
    Rule([0, None, None, None, 3], [same, same, same, same, increment]),
    # turn == i
    Rule([0, None, None, 3, None], [same, same, same, lambda x: 2, same]),
    Rule([1, None, None, None, 3], [same, same, same, same, lambda x: 2]),

    # ci=1
    Rule([None, None, None, 4, None], [same, one, same, increment, same]),
    Rule([None, None, None, None, 4], [same, same, one, same, increment]),

    # wait until turn=i
    # turn == i
    Rule([0, None, None, 5, None], [same, same, same, increment, same]),
    Rule([1, None, None, None, 5], [same, same, same, same, increment]),

    # turn == !i
    Rule([1, None, None, 5, None], [same, same, same, same, same]),
    Rule([0, None, None, None, 5], [same, same, same, same, same]),

    # ci = 0
    Rule([None, None, None, 6, None], [same, zero, same, increment, same]),
    Rule([None, None, None, None, 6], [same, same, zero, same, increment]),

    # ci = 1
    Rule([None, None, None, 7, None], [same, one, same, increment, same]),
    Rule([None, None, None, None, 7], [same, same, one, same, increment]),

    # turn = !i
    Rule([None, None, None, 8, None], [one, same, same, zero, same]),
    Rule([None, None, None, None, 8], [zero, same, same, same, zero]),
}

initialStates = {
    State(False, False, False, -1, -1),
    # State(False, False, True, -1, -1),
    # State(False, True, False, -1, -1),
    # State(False, True, True, -1, -1),
}


class KripkeStructure(object):
    def __init__(self, initialStates: Set[State], rules: Set[Rule]):
        self.state_amount = len(initialStates)

        self.states = initialStates
        self.rules = rules
        # assert (self.states[i].id == i for i in range(self.state_amount))
        # self.transitions = sorted(transitions,
        #                           key=lambda t: (min(t.origin.p0, t.origin.p1), max(t.origin.p0, t.origin.p1)))
        self.transitions = self.generateTransitions()

    def draw_transitions(self):
        G = nx.DiGraph()
        edges = {}
        # for t in self.transitions:
        #     edges[(t.origin, t.destination)] = t.probability
        states = [str(s) for s in self.states]
        G.add_nodes_from(states)
        for t in self.transitions:
            G.add_edge(str(t.origin), str(t.destination))
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos)
        nx.drawing.nx_pydot.write_dot(G, '../Chain.dot')
        Source.from_file('../Chain.dot').view()

    def generateTransitions(self):
        visited = set()
        q = deque(self.states)
        transitions = set()
        i = 0
        while q:
            curState = q.popleft()
            for rule in self.rules:
                transition = rule.getTransition(curState)
                if transition:
                    transitions.add(transition)
                    if transition.destination not in visited:
                        q.append(transition.destination)
            visited.add(curState)
            if i % 1000:
                print(len(visited))
            i += 1
        self.states = visited
        self.state_amount = len(self.states)
        return transitions

        # for state in states:
        #     inital_state_probability = 0
        #     for event in events:
        #         next_state = states[-1]
        #         new_state_name = ""
        #         appended = state.name + event.sign
        #         state_names = [_.name for _ in states]
        #         for i in range(len(appended)):
        #             if appended[i:] in state_names:
        #                 new_state_name = appended[i:]
        #                 break
        #         if new_state_name == "":
        #             inital_state_probability += event.probability
        #         else:
        #             next_state = next(_ for _ in states if _.name == new_state_name)
        #             if state.name == "":
        #                 state = states[-1]
        #             transitions.append(Transition(state, next_state, event.probability))
        #     next_state = states[-1]
        #     if inital_state_probability > 0:
        #         transitions.append(Transition(state, next_state, inital_state_probability))
        # return transitions

    # @classmethod
    # def construct_states(cls, starting_states, terminals):
    #     state_dictionary = {t[:i]: 0 for t in terminals for i in range(1, len(t) + 1)}
    #     state_dictionary["_"] = 0
    #     if not starting_states:
    #         starting_states = ["_"]
    #     length = len(starting_states)
    #     for state_name in starting_states:
    #         state_dictionary[state_name] = 1 / length
    #     states = list()
    #     for i, (k, v) in enumerate(state_dictionary.items()):
    #         if i < len(state_dictionary.items()) - 1:
    #             states.append(State(k, i + 1, v))
    #     empty_state_probability = next(v for k, v in state_dictionary.items() if k == "_")
    #     states.append(State("_", 0, empty_state_probability))
    #     return states


if __name__ == "__main__":
    kripke = KripkeStructure(initialStates, rules)
    kripke.draw_transitions()