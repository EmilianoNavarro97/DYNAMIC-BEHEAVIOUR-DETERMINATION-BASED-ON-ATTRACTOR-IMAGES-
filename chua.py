from json import load
from functions import chua_integrator
from tqdm import tqdm


def main():
    with open('data2.json') as f:
        data = load(f)

    # Iterating over the json data
    for k in data.keys():
        diagram = data[k]['diagram']
        n_attractors = data[k]['n_attractors']
        regular_axis = data[k]['eje_regular']
        chaotic_axis = data[k]['eje_caotico']

        print(f'{k}:')
        chua_integrator(diagram, n_attractors, regular_axis, 'regular', base_path='ExportChua/regular',
                        save_diagram=False, save_points=False)
        chua_integrator(diagram, n_attractors, chaotic_axis, 'caotico', base_path='ExportChua/caotico',
                        save_diagram=False, save_points=False)


if __name__ == '__main__':
    main()
