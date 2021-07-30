from json import load
from functions import chua_integrator


def main():
    with open('data.json') as f:
        data = load(f)

    # Iterating over the json data
    for k in data.keys():
        diagram = data[k]['diagram']
        n_attractors = data[k]['n_attractors']
        regular_axis = data[k]['eje_regular']
        chaotic_axis = data[k]['eje_caotico']

        print(f'{k}:')
        chua_integrator(diagram, n_attractors, regular_axis, 'regular', base_path='ExportChua/regular')
        chua_integrator(diagram, n_attractors, chaotic_axis, 'caotico', base_path='ExportChua/caotico')


if __name__ == '__main__':
    main()
