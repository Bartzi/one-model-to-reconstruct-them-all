import argparse

from utils.command_line_args import add_default_args_for_projecting


def test_default_args():
    parser = argparse.ArgumentParser()
    parser = add_default_args_for_projecting(parser)

    assert isinstance(parser, argparse.ArgumentParser)
