import matplotlib.pyplot as plt
import pandas as pd
import argparse

def plot(df:pd.DataFrame, name:str = "plot.png", text = ""):
    "pc, pm, gens"

    df.sort_values(by=df.iloc[:,2].name, inplace=True)
    smallest = df.iloc[0]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Rekombinationsrate')
    ax.set_ylabel('Mutationsrate')
    ax.set_zlabel("Generationen")
    ax.view_init(30, 340)
    plt.yticks(rotation=270+65, fontsize=8)
    plt.xticks(fontsize=8)
    trisurf = ax.plot_trisurf(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], cmap="plasma")
    plt.suptitle(f"best: pc={round(smallest.iloc[0],)}, pm={round(smallest.iloc[1], 6)} => {round(smallest.iloc[2], 6)} Generationen\n{text}", wrap=True)
    cbar=fig.colorbar(mappable=trisurf, shrink=0.8, pad=0.2)
    plt.savefig(name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot a csv file generated by new.py. must be in orger [pc, pm, gens]")
    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument("output", help="output file", type=str)
    args = parser.parse_args()

    with open(args.input) as file:
        df = pd.read_csv(file, index_col=0)
        plot(df, args.output)
