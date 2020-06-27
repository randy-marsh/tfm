"""
Didactic Plot showing what is a fog event
"""
import argparse
import numpy
import src.utils.commons
import src.models.base_model
import plotly.plotly
import plotly.graph_objs
import plotly.offline
import typing
import os


def parse_args() -> typing.Dict:
    """
    Default arg parser
    reads output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, help="output path",
                        default="../../reports/figures/fog_event.html")
    args = parser.parse_args()
    return vars(args)


def main(output_path: str):

    x = [hour for hour in range(1, 21)]
    fog_values = [None for hour in range(1, 5)] + [2000, 1998, 1995, 1992, 1995, 2000] + [None for hour in range(11, 21)]
    data = [plotly.graph_objs.Scatter(x=x,
                                      y=[2000 if (i <= 5) or (i >= 10) else None
                                         for i in range(1, 21)],
                                      name='No Fog Event')]
    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Fog Event',
        xaxis=dict(
            title='Hours'
        ),
        yaxis=dict(
            title='Runway Visual Range [m]'
        ),
    )
    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    fig.add_trace(plotly.graph_objs.Scatter(x=x,
                                            y=fog_values,
                                            name='Fog event'))
    plotly.offline.plot(fig, filename=output_path)

if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)