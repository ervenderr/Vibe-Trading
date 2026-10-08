import argparse

from src.desk.server import serve

parser = argparse.ArgumentParser(prog="python -m src.desk", description="Paper Desk: read-only autopilot dashboard")
parser.add_argument("--port", type=int, default=8765)
serve(parser.parse_args().port)
