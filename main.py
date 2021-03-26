import copy
from typing import List, NamedTuple, Tuple
import numpy as np
import networkx as nx
from graphviz import Source


class State(NamedTuple):
    turn: bool
    c0: bool
    c1: bool
    pc0: int
    pc1: int

    def __eq__(self, other):
        return self.turn == other.turn and \
               self.c0 == other.c0 and \
               self.c1 == other.c1 and \
               self.pc0 == other.pc0 and \
               self.pc1 == other.pc1


class Transition(NamedTuple):
    origin: State
    destination: State
    probability: float


class Rule:
    def __init__(self, turnAction, c0Action, c1Action, pc0Action, pc1Action):
        self.turnAction = turnAction
        self.c0Action = c0Action
        self.c1Action = c1Action
        self.pc0Action = pc0Action
        self.pc1Action = pc1Action

    def getTransition(self, state: State):
        newNode = State(0, 0, 0, 0, 0)
        newNode.turn = self.turnAction(state.turn)
        newNode.c0 = self.c0Action(state.c0)
        newNode.c1 = self.c1Action(state.c1)
        newNode.pc0 = self.pc0Action(state.pc0)
        newNode.pc1 = self.pc1Action(state.pc1)
        return Transition(state, newNode, )

    

class KripkeStructure(object):

    def __init__(self, states: List[State], rules: List[State]):
        self.state_amount = len(states)
        for i, state in enumerate(states):
            for j in range(i + 1, self.state_amount):
                assert (state != states[j])
        self.states = sorted(states, key=lambda s: s.id)
        # assert (self.states[i].id == i for i in range(self.state_amount))
        self.transitions = sorted(transitions, key=lambda t: t.origin.id)
        self.transition_matrix = None
        self.state_vector = None
        self.steps = 0
        self.construct_matrix()
        self.construct_vector()

    def construct_matrix(self):
        self.transition_matrix = np.zeros([self.state_amount, self.state_amount], dtype=float)
        for t in self.transitions:
            assert 0 <= t.origin.id < self.state_amount and 0 <= t.destination.id < self.state_amount
            self.transition_matrix[t.origin.id, t.destination.id] = t.probability
        assert (sum(t) == 1 for t in self.transition_matrix)

    def construct_vector(self):
        self.state_vector = np.array([s.initial_probability for s in self.states])

    def reset(self):
        self.construct_matrix()
        self.construct_vector()
        self.steps = 0

    def progress_k_steps(self, k):
        for i in range(k):
            self.state_vector = np.dot(self.state_vector, self.transition_matrix)
        self.steps += k

    def get_state_vector(self):
        return self.state_vector

    def get_states(self):
        return self.states

    def get_transitions(self):
        return self.transitions

    def get_expected_number_of_steps_to_goal(self, source_state: str, goal_state: str) -> float:
        src, dest = None, None
        for s in self.states:
            if s.name == source_state:
                src = s
            if s.name == goal_state:
                dest = s
        assert src and dest

        linear_equation = np.copy(self.transition_matrix)
        print(linear_equation)
        for i, row in enumerate(linear_equation):
            if i != dest.id:
                linear_equation[i, i] -= 1
            else:
                linear_equation[i] = np.zeros((1, self.state_amount), dtype=float)
                linear_equation[i, i] = 1

        print(linear_equation)
        inverse = np.linalg.inv(linear_equation)
        b_vector = np.full((self.state_amount), -1)
        b_vector[dest.id] = 0
        solution_vector = np.dot(inverse, b_vector)
        return solution_vector[src.id]

    def __repr__(self):
        out_str = f"After {self.steps} steps.\n"
        for i, prob in enumerate(self.state_vector):
            out_str += f"State:{self.states[i].name}, Probability: {prob}\n"
        return out_str

    def show_transition_matrix(self):
        print(self.transition_matrix)

    def draw_transitions(self):
        G = nx.DiGraph()
        edges = {}
        for t in self.transitions:
            edges[(t.origin.name, t.destination.name)] = t.probability
        states = [s.name for s in self.states]
        G.add_nodes_from(states)
        for k, v in edges.items():
            origin, destination = k[0], k[1]
            G.add_edge(origin, destination, weight=v, label=f"{v:.2f}")
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos)
        nx.drawing.nx_pydot.write_dot(G, '../Chain.dot')
        Source.from_file('../Chain.dot').view()

    @classmethod
    def get_chain_from_events(cls, terminals: [str], events: List[Event], starting_states: List[State] = None):
        assert abs(sum(e.probability for e in events) - 1) < 0.00001
        states = cls.construct_states(starting_states, terminals)
        transitions = cls.construct_transitions(events, states)
        return cls(states, transitions)

    @classmethod
    def construct_transitions(cls, events, states):
        transitions = list()
        for state in states:
            inital_state_probability = 0
            for event in events:
                next_state = states[-1]
                new_state_name = ""
                appended = state.name + event.sign
                state_names = [_.name for _ in states]
                for i in range(len(appended)):
                    if appended[i:] in state_names:
                        new_state_name = appended[i:]
                        break
                if new_state_name == "":
                    inital_state_probability += event.probability
                else:
                    next_state = next(_ for _ in states if _.name == new_state_name)
                    if state.name == "":
                        state = states[-1]
                    transitions.append(Transition(state, next_state, event.probability))
            next_state = states[-1]
            if inital_state_probability > 0:
                transitions.append(Transition(state, next_state, inital_state_probability))
        return transitions

    @classmethod
    def construct_states(cls, starting_states, terminals):
        state_dictionary = {t[:i]: 0 for t in terminals for i in range(1, len(t) + 1)}
        state_dictionary["_"] = 0
        if not starting_states:
            starting_states = ["_"]
        length = len(starting_states)
        for state_name in starting_states:
            state_dictionary[state_name] = 1 / length
        states = list()
        for i, (k, v) in enumerate(state_dictionary.items()):
            if i < len(state_dictionary.items()) - 1:
                states.append(State(k, i + 1, v))
        empty_state_probability = next(v for k, v in state_dictionary.items() if k == "_")
        states.append(State("_", 0, empty_state_probability))
        return states
