from resolve_tsp import read_map, gen_fitness, generate_distance, greedy_crossover, genes_sort, init_gen
import numpy as np

map = np.array(
        [
            [ 0,  1,  2, 3],
            [11, -1, -1, 4],
            [10, -1, -1, 5],
            [ 9,  8,  7, 6],
            ]
        )

def test_generate_distance():
    np.set_printoptions(precision=2)
    dist = generate_distance(map)
    for i in range(len(dist)):
        assert dist[i][i] == 0
    assert dist[0][6] == np.sqrt((3**2)*2)


def test_fitness():
    from resolve_tsp import gen_fitness
    dist = generate_distance(map)
    genes = np.array([
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [11,10,9,8,7,6,5,4,3,2,1,0],
        [0,2,1,3,4,5,6,7,8,9,10,11]
        ])
    fitness = gen_fitness(genes, dist)
    assert fitness[0] == 12
    assert fitness[1] == 12
    assert fitness[2] == 14
    map_36 = read_map("./05-map-10x10-36border.txt")
    dist = generate_distance(map_36)
    genes = np.array([
        np.arange(36),
        np.flipud(np.arange(36))
        ])
    fitness = gen_fitness(genes, dist)
    assert fitness[0] == 36
    assert fitness[1] == 36
    


def test_crossover():
    map = np.array(
            [
                [1,0,2,0],
                [4,0,0,0],
                [0,0,0,0],
                [0,0,0,3]
                ]
            ) -1 
    dist = generate_distance(map)
    genes = init_gen(len(dist), 2)
    print(dist)
    print(genes[0])
    print(genes[1])
    crossed = greedy_crossover(genes[0], genes[1], dist)
    print(crossed)
    # print_gene(genes_sort(gene, dist), dist)
    # for i in range(500):
    #     gene = recombine(gene, dist, 0.8)
    #     print_gene(gene, dist)
