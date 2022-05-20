import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument("--mode", default='client')
    argparser.add_argument("--port", default=50722)
    args = argparser.parse_args(args=[])
    return args
