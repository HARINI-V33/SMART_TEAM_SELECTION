import random

# Define the parameters
population_size = 100
mutation_rate = 0.01
num_generations = 100

# Define the target solution (for demonstration purposes)
target_solution = "Hello, Genetic Algorithm!"

# Generate initial population
def generate_individual(length):
    return ''.join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!?") for _ in range(length))

def generate_population(population_size, length):
    return [generate_individual(length) for _ in range(population_size)]

# Define fitness function
def calculate_fitness(individual, target):
    return sum(1 for i, j in zip(individual, target) if i == j)

# Selection: Tournament Selection
def tournament_selection(population, fitness_func):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    tournament_fitness = [fitness_func(individual, target_solution) for individual in tournament]
    return tournament[tournament_fitness.index(max(tournament_fitness))]

# Crossover: Single Point Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutate(individual, mutation_rate):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!?")
    return ''.join(mutated_individual)

# Genetic Algorithm
def genetic_algorithm(population_size, target, mutation_rate, num_generations):
    population = generate_population(population_size, len(target))
    for generation in range(num_generations):
        # Calculate fitness for each individual
        fitness_scores = [calculate_fitness(individual, target) for individual in population]
        
        # Check if target is reached
        if max(fitness_scores) == len(target):
            break
        
        # Select parents and create next generation
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, calculate_fitness)
            parent2 = tournament_selection(population, calculate_fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    
    best_individual = max(population, key=lambda x: calculate_fitness(x, target))
    return best_individual, generation

# Run the genetic algorithm
best_solution, num_generations = genetic_algorithm(population_size, target_solution, mutation_rate, num_generations)

print("Best solution found:", best_solution)
print("Number of generations:", num_generations)
