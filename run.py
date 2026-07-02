#!/usr/bin/env python3
"""
run.py – Launcher for surveillance + agent with shared Queue
"""
import multiprocessing
import argparse
from office_agent import run_agent
from main import run_surveillance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=int, required=True)
    parser.add_argument("--rtsp", type=str, required=True)
    args = parser.parse_args()

    # Create shared queue for inter-process communication
    manager = multiprocessing.Manager()
    shared_queue = manager.Queue(maxsize=200)

    # Start the intelligence agent process (async LLM)
    agent_process = multiprocessing.Process(target=run_agent, args=(shared_queue,))
    agent_process.daemon = True
    agent_process.start()

    # Start surveillance (this will run in the main process)
    run_surveillance(args.camera_id, args.rtsp, shared_queue)