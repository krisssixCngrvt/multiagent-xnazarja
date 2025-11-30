#!/bin/bash

# Quick run script for testing
# Uses fewer episodes for faster execution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run.sh" --quick "$@"
