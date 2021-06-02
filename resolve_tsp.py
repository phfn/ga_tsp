from datetime import time
from random import randint

from timeit import default_timer as timer
import numba
import pandas as pd
import numpy as np
import argparse
from numba import njit, prange
from tqdm import tqdm
from timebudget import timebudget
timebudget.set_quiet()  # don't show measurements as they happen
timebudget.report_at_exit()  # Generate report when the program exits
np.set_printoptions(threshold=np.inf)
from plot import plot

def read_map(path: str) -> np.ndarray:
    with open(path) as file:
        df = pd.read_csv(file, sep="\s+", header=None, dtype=np.int64)
    df = df - 1

    print(df)
    return df.to_numpy()

# @njit
def map_to_coords(map):
    number_of_cities = np.max(map)+1
    coords = np.zeros(shape=(number_of_cities, 2), dtype=np.int64)

    for i, row in enumerate(map):
        for j, element in enumerate(row):
            if element > 0:
                coords[element] = (i, j)

    return coords

# @njit
def generate_distance(map: np.ndarray):
    number_of_cities = np.max(map)+1
    distance = np.zeros(shape=(number_of_cities, number_of_cities))
    coords = map_to_coords(map)

    for i in range(number_of_cities):
        for j in range(number_of_cities):
            d = np.linalg.norm(coords[i] - coords[j])
            distance[i][j] = d
            distance[j][i] = d

    return distance


    
    

def init_gen(l: int, n: int):
    gene = np.zeros(shape=(n,l), dtype=np.int64)
    rg = np.random.default_rng()
    arr = np.arange(l, dtype=np.int64)

    for i in range(n):
        rg.shuffle(arr)
        gene[i] = arr

    return gene


@njit
def gen_fitness(genes, dist):
    fitness = np.zeros(shape=(len(genes),))

    for i in range(len(genes)):
        cg = genes[i] #current gene

        for j in range(len(cg)):
            j_1 = j + 1

            if j_1>=len(cg):
                j_1 = 0
            fitness[i] += dist[ cg[j] ][ cg[j_1] ]
        # fitness[i] = cd

    return fitness


@njit
def genes_sort(genes: np.ndarray, dist) -> np.ndarray:
    fit = gen_fitness(genes, dist).argsort()
    sorted = np.zeros_like(genes)

    for index, sorted_index in enumerate(fit):
        sorted[index] = genes[sorted_index]

    return sorted


@timebudget
@njit
def replicate(genes: np.ndarray, dist) -> np.ndarray:
    genes_sorted = genes_sort(genes, dist)
    genes_new = np.zeros_like(genes)
    a = genes_sorted[0]
    b = genes_sorted[1]


    for i in range(len(genes)):
        if i < len(genes)//2:
            genes_new[i] = a.copy() if i%2==0 else b.copy()
        else:
            genes_new[i] = genes[np.random.randint(len(genes)-1)].copy()

    return genes_new


@timebudget
@njit
def greedy_crossover(gene1: np.ndarray, gene2: np.ndarray, dist) -> np.ndarray:
    new_gene = np.zeros_like(gene1)-1  # init with -1
    new_gene[0] = gene1[0]

    for i in range(1, len(new_gene)):
        prev_city = new_gene[i-1]

        # find next city in gene1 and gene2
        index_1 = np.where(gene1 == prev_city)[0][0] + 1
        index_2 = np.where(gene2 == prev_city)[0][0] + 1

        # move to the start if overflowing
        index_1 = index_1 if index_1 < len(gene1) else 0
        index_2 = index_2 if index_2 < len(gene1) else 0

        candidate1 = gene1[index_1]
        candidate2 = gene2[index_2]

        # distance is infinive if city is allready in new_gene
        # this way a gene allready taken is allways bad
        dist1 = dist[prev_city][index_1] if candidate1 not in new_gene else np.Infinity
        dist2 = dist[prev_city][index_2] if candidate2 not in new_gene else np.Infinity

        next_city = candidate1 if dist1 < dist2 else candidate2

        # take random if both infinive (see up 7 lines)
        while next_city == np.Infinity or next_city in new_gene:
            next_city = np.random.choice(gene1)
            if next_city not in new_gene:
                break

        new_gene[i] = next_city

    return new_gene


@timebudget
# @njit
def recombine(gene, dist, pc, n_protect):
    gene_sorted = genes_sort(gene, dist)
    n_cross = int(pc * len(gene))
    new_gene = np.zeros_like(gene) - 1  # inti with -1
    # print(len(gene), n_protect, n_cross, n_protect+n_cross)
    assert n_cross + n_protect <= len(gene)

    for i in range(n_protect):
        new_gene[i] = gene_sorted[i]

    for i in prange(n_protect, n_protect+n_cross):
        # crossover_gereeedy
        a = np.random.randint(0, len(gene))
        b = np.random.randint(0, len(gene))
        new_gene[i] = greedy_crossover(gene[a], gene[b], dist)

    for i in range(n_protect+n_cross, len(gene)):
        # random
        ri = np.random.randint(0, len(gene))
        new_gene[i] = gene[ri]

    return new_gene


@timebudget
@njit
def mutate(genes, pm):
    for _ in range(int(pm * len(genes) * len(genes[0]))):
        a = np.random.randint(0, len(genes))
        b = np.random.randint(0, len(genes[a]))
        c = np.random.randint(0, len(genes[a]))
        genes[a][b], genes[a][c] = genes[a][c], genes[a][b]

    return genes





def sumup_genes(genes, dist):
    fitness = gen_fitness(genes, dist)
    # for i in range(len(genes)):
    #     print(f"{i}: {fitness[i]} {genes[i]}")
    print(f"avg fitness: {sum(gen_fitness(genes, dist))/len(genes)}")
    print(f"best fitness: {min(gen_fitness(genes, dist))}")
    print_gene(genes_sort(genes, dist)[0], dist)
    print("-------------")


def print_gene(gene, dist):
    print(str(gene +1).replace("  ", " 0")[1:-1])


# @njit
def run(pc, pm, map, n_gene, max_generations, print_best = False):
    dist = generate_distance(map)
    genes = init_gen(len(dist), n_gene)
    n_generations = 0

    for i in range(max_generations):
        n_generations = i+1

        genes = mutate(genes, pm)

        if min(gen_fitness(genes, dist))<=args.best_fitness:
            break
        if parse().replicate:
            genes = replicate(genes, dist)

        if min(gen_fitness(genes, dist))<=args.best_fitness:
            break
        genes = recombine(genes, dist, pc, args.protect)

        if min(gen_fitness(genes, dist))<=args.best_fitness:
            break

    # if print_best:
        # sumup_genes(genes, dist)

    return n_generations


def find_pc_pm():
    args = parse()
    map = read_map(args.input)
    n_gene = args.n_gene
    max_generations = args.max_generations
    max_fitness = args.best_fitness

    pcr = np.arange(args.pc1, args.pc2, args.pcs)
    pmr = np.arange(args.pm1, args.pm2, args.pms)
    avgs = []

    with tqdm(total=len(pcr)*len(pmr)*args.runs) as pbar:
        for pc in pcr:
            for pm in pmr:
                avg = 0
                for _ in range(args.runs):
                    avg += run(pc,pm,map,n_gene,max_generations, max_fitness)
                    pbar.update(1)

                avg = avg/args.runs
                avgs.append((pc, pm, avg))
                # print((pc, pm, avg))

    return avgs









def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help = "input map", type=str, default="./05-map-10x10-36border.txt")
    parser.add_argument("--n_gene", "-n", help = "Anzahl der Gene", type=int, default=200)
    parser.add_argument("--max_generations", "-g", help = "Maximalanzahl der Generationen, danach Abbruch", type=int, default=1000)
    parser.add_argument("--best_fitness", "-b", help = "Bei welcher Fitness soll abgebrochen werden", type=float, default=0)
    parser.add_argument("--pc", "-r", help = "Rekombinationsrate", type=float, default=0.9)
    parser.add_argument("--pm", "-m", help = "Mutationsrate", type=float, default=0.0015)
    parser.add_argument("--runs", "-a", help = "Zahl der Läufe, über die gerundet wird", type=int, default=50)
    parser.add_argument("--search", "-s", help = "search for best pm pc in a period", default=False, action="store_true")
    parser.add_argument("--pc1",help = "min Rekombinationsrate", type=float, default=0)
    parser.add_argument("--pc2",help = "max Rekombinationsrate", type=float, default=0.9)
    parser.add_argument("--pcs",help = "steps Rekombinationsrate", type=float, default=0.05)
    parser.add_argument("--pm1",help = "min Mutationsrate", type=float, default=0.0)
    parser.add_argument("--pm2",help = "max Mutationsrate", type=float, default=0.2)
    parser.add_argument("--pms",help = "steps Mutationsrate", type=float, default=0.005)
    parser.add_argument("--filename", "-f", help = "Name to save the data as csv and a plot of the data", type=str, default="ga")
    parser.add_argument("--protect", "-p", help = "protect the best n gene from mutation and crossover", type=int, default=0)
    parser.add_argument("--replicate", "-x", help = "replicate or do not Replicate", type=bool, default=True)



    return parser.parse_args()



if __name__ == "__main__":
    args = parse()

    if args.search:
        start = timer()
        avgs = find_pc_pm()
        end = timer()
        df = pd.DataFrame(avgs)
        name = args.filename
        df.to_csv(f"{args.filename}.csv")
        plot(df, f"{args.filename}.png", f"pc1: {args.pc1} pc2: {args.pc2} pcs: {args.pcs} pm1: {args.pm1} pm2: {args.pm2} pms: {args.pms} \n {(' s=' + str(args.s)) if args.replication_scheme == 2 else ''} recomb: crossover protect:{'best' if args.protect else 'None'} time: {round(end-start)}")
    else:
        r = run(args.pc, args.pm, read_map(args.input), args.n_gene, args.max_generations, print_best=True)
        print(r)

    # test_crossover()
    # find_pc_pm(map, 50, 500, 36)
    # print(map)
    # coords = map_to_coords(map)
    # # print(coords)
    # dist = generate_distance(map)
    # # print(np.around(dist, decimals=2))
    # # print(dist)
    # # print(dist[16][1])
    # gene = init_gen(36, 100)
    # # print(gene)
    # fitness = gen_fitness(gene, dist)
    # # print(fitness)
    # # print(gen_fitness([[i for i in range(36)]], dist))
    # # print_gene(gene, dist)
    # gene = replicate(gene, dist)
    # # print_gene(gene, dist)
    # # print(gene[0])
    # # print(gene[-1])
    # # crossed = greedy_crossover(gene[0],gene[-1], dist)
    # # print(crossed)

    # # print(crossed.shape)
    # print(gen_fitness([gene[0],gene[1],crossed],dist))

    # for i in range(500):
    #     gene = crossover(gene, dist, 0.8)
    #     print_gene(gene, dist)
