import numpy as np
import copy

# store history of innovation numbers of genes
innovation_number = 0
innovation_numbers = {}

# define constants
DISABLE_PROBABILITY = 0.5

EXCESS_COEFFICIENT = 1
DISJOINT_COEFFICIENT = 1
WEIGHT_DIFFERENCE_COEFFICIENT = 0.4
COMPATIBILITY_THRESHOLD = 0.3
ELITISM = 2

PROB_MUTATE_WEIGHTS = 0.8
PROB_ADD_CONNECTION = 0.05
PROB_ADD_NODE = 0.03

NUM_INPUTS = 5
NUM_OUTPUTS = 2


# get gene from innovation number
def get_gene(innov):
    for gene in innovation_numbers:
        if innovation_numbers[gene] == innov:
            return gene


class Gene:
    def __init__(self, inp, out):
        global innovation_number
        self.inp = inp
        self.out = out
        self.weight = np.random.randn()
        self.enabled = True
        # give next free if new input/output
        if (inp, out) in innovation_numbers:
            self.innov = innovation_numbers[(inp, out)]
        else:
            self.innov = innovation_number + 1
            innovation_numbers[(inp, out)] = self.innov
        innovation_number += 1

    # to get state of genome
    def __repr__(self):
        return f"Weight: {round(self.weight, 4)}\tEnabled: {self.enabled}\tInnovation Number: {self.innov}"


class Genome:
    def __init__(self, i, o, default_init=True):
        self.i = i
        self.o = o
        self.genes = {}
        self.node_id = 0
        self.nodes = {}
        self.fitness = 0
        for _ in range(i):
            self.add_node("i")
        self.add_node("b")
        self.bias = np.random.randn()
        for _ in range(o):
            self.add_node("o")
        # starts fully connected from inputs to outputs
        if default_init:
            for n_id, n in self.nodes.items():
                for n1_id, n1 in self.nodes.items():
                    if n == "i" and n1 == "o":
                        self.genes[(n_id, n1_id)] = Gene(n_id, n1_id)

    def __str__(self):
        return f"Nodes:\n{self.nodes}\nGenes:\n" + "\n".join([str(n_ids) + ": " + str(gene) for n_ids, gene in self.genes.items()])

    def add_node(self, t):
        self.node_id += 1
        self.nodes[self.node_id] = t

    # start at each output node and get the values of nodes before it recursively
    def feed_forward(self, input, node=None):
        if node is None:
            outputs = []
            for n_id, n in self.nodes.items():
                if n == "o":
                    s = 0
                    for _, gene in self.genes.items():
                        if gene.enabled and gene.out == n_id:
                            s += gene.weight * self.feed_forward(input, node=gene.inp)
                    outputs.append(s)
            return np.argmax(outputs)
        if self.nodes[node] == "i":
            return input[node - 1]
        if self.nodes[node] == "b":
            return self.bias
        for n_id, n in self.nodes.items():
            if n_id == node:
                s = 0
                for _, gene in self.genes.items():
                    if gene.enabled and gene.out == n_id:
                        s += gene.weight * self.feed_forward(input, node=gene.inp)
                return s

    # is a certain input/output pair corresponding to a gene present in the genome
    def has_gene(self, gene_tuple):
        return gene_tuple in self.genes

    # assumes self is the fitter parent - crosses over two parents to give a child genome
    def crossover(self, genome):
        child = Genome(self.i, self.o, default_init=False)
        max_innov = max(max([g.innov for g in self.genes.values()]), max([g.innov for g in genome.genes.values()]))
        for i in range(1, max_innov + 1):
            gene_i = get_gene(i)
            if self.has_gene(gene_i) and genome.has_gene(gene_i):
                par1, par2 = self.genes[gene_i], genome.genes[gene_i]
                to_add = copy.deepcopy(np.random.choice(par1, par2))
                if not par1.enabled or not par2.enabled:
                    to_add.enabled = False if np.random.random() < DISABLE_PROBABILITY else True
                child.genes[gene_i] = to_add
            elif self.has_gene(gene_i):
                child.genes[gene_i] = self.genes[gene_i]
        return child

    # after crossover gives a child from two parents, a sequence of mutations will be applied randomly
    def mutate(self):
        # either the weight is changed by a small gaussian or completely reassigned
        if np.random.random() < PROB_MUTATE_WEIGHTS:
            random_gene = np.random.choice(self.genes.keys())
            self.genes[random_gene] += np.random.normal(loc=0, scale=0.1)
        else:
            random_gene = np.random.choice(self.genes.keys())
            self.genes[random_gene] = np.random.randn()
        # connect two previously disconnected nodes. since this is feedforward neat we strictly need to connect in order of the forward pass
        if np.random.random() < PROB_ADD_CONNECTION:
            while True:
                from_node = np.random.choice(self.nodes.keys())
                while self.nodes[from_node] == "o":
                    from_node = np.random.choice(self.nodes.keys())
                to_node = np.random.choice(self.nodes.keys())
                while to_node == from_node or not ((self.nodes[from_node] == "i" and self.nodes[to_node] == "o") or (self.nodes[from_node] == "i" and self.nodes[to_node] == "h") or (self.nodes[from_node] == "h" and self.nodes[to_node] == "o")):
                    to_node = np.random.choice(self.nodes.keys())
                if (from_node, to_node) not in self.genes:
                    break
            self.genes[(from_node, to_node)] = Gene(from_node, to_node)
        # add a node, separating a connection into two connections, one to the intermediary node and one to the original final node
        if np.random.random() < PROB_ADD_NODE:
            connection = np.random.choice(self.genes.keys())
            from_node, to_node = connection
            self.genes[connection].enabled = False
            new_node = max(self.nodes.keys()) + 1
            self.nodes[new_node] = "h"
            self.genes[(from_node, new_node)] = Gene(from_node, new_node)
            self.genes[(new_node, to_node)] = Gene(new_node, to_node)

    # needed for checking speciation - counts the number of disjoint and excess genes between two genomes
    def count_disjoint_excess_weight_diff(self, genome):
        max_innov = max(max([g.innov for g in self.genes.values()]), max([g.innov for g in genome.genes.values()]))
        disjoint, excess = 0, 0
        presence_type = None
        for i in range(max_innov, 0, -1):
            gene_i = get_gene(i)
            if (self.has_gene(gene_i) and not genome.has_gene(gene_i) and (presence_type is None or presence_type == 1)) or (not self.has_gene(gene_i) and genome.has_gene(gene_i) and (presence_type is None or presence_type == 2)):
                excess += 1
                if presence_type is None:
                    if self.has_gene(gene_i) and not genome.has_gene(gene_i):
                        presence_type = 1
                    elif not self.has_gene(gene_i) and genome.has_gene(gene_i):
                        presence_type = 2
            else:
                break
        for j in range(1, max_innov + 1 - excess):
            gene_i = get_gene(j)
            if (self.has_gene(gene_i) and not genome.has_gene(gene_i)) or (not self.has_gene(gene_i) and genome.has_gene(gene_i)):
                disjoint += 1
        diffs = []
        for k in range(1, max_innov + 1):
            gene_i = get_gene(k)
            if self.has_gene(gene_i) and genome.has_gene(gene_i):
                diffs.append(abs(self.genes[gene_i].weight - genome.genes[gene_i].weight))
        return disjoint, excess, sum(diffs) / len(diffs)

    def get_normalizing_factor(self, genome):
        N = max(len(self.genes), len(genome.genes))
        if N < 20:
            return 1
        return N

    # get how compatible two genomes are as per the formula
    def compatibility(self, genome):
        disjoint, excess, weight_diff = self.count_disjoint_excess_weight_diff(genome)
        N = self.get_normalizing_factor(genome)
        return EXCESS_COEFFICIENT * excess / N + DISJOINT_COEFFICIENT * disjoint / N + WEIGHT_DIFFERENCE_COEFFICIENT * weight_diff


class Species:
    def __init__(self):
        self.members = []

    # use the first genome in the species as the comparison to decide whether a new genome is to be added
    def add_genome(self, to_add):
        if not self.members:
            self.members.append(to_add)
            return True
        if self.members[0].compatibility(to_add) < COMPATIBILITY_THRESHOLD:
            self.members.append(to_add)
            return True
        return False

    # select parents with a likelihood proportional to their fitness
    def select_parent(self):
        rand = np.random.random()
        total_fitness = sum(m.fitness for m in self.members)
        fitness_sums = [0]
        running_sum = 0
        for m in self.members:
            running_sum += m.fitness / total_fitness
            fitness_sums.append(running_sum)
        for i in range(len(fitness_sums) - 1):
            if fitness_sums[i] < rand < fitness_sums[i + 1]:
                return self.members[i]

    # cull the bottom half in terms of fitness, then replace those lost with the fitter individuals' children
    def reproduce(self):
        self.members.sort(key=lambda l: l.fitness)
        prev_length = len(self.members)
        self.members = self.members[len(self.members) // 2:]
        new_length = len(self.members)
        for _ in range(new_length - prev_length):
            first = self.select_parent()
            second = self.select_parent()
            while second == first:
                second = self.select_parent()
            first, second = max([first, second], key=lambda l: l.fitness), min([first, second], key=lambda l: l.fitness)
            child = first.crossover(second)
            child.mutate()
            self.members.append(child)


class Population:
    def __init__(self):
        self.species = []

    # add a genome to an existing species - if none are compatible, a new species is created
    def add_genome(self, genome):
        for species in self.species:
            if species.add_genome(genome):
                break
        else:
            new_species = Species()
            new_species.add_genome(genome)
            self.species.append(new_species)

    # call reproduce on each species
    def update_population(self):
        for species in self.species:
            species.reproduce()
