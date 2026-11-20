"""momentumlmb command line. Heavy imports stay inside each subcommand so
startup is instant."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend")
    parser.add_subparsers(dest="command", required=True)
    parser.parse_args()


if __name__ == "__main__":
    main()
