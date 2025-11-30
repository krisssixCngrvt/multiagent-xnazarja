#!/bin/bash

# Multi-Agent Foraging RL Coach - Run Script
# Optimized for achieving the highest possible score

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Multi-Agent Foraging RL Coach${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo -e "${RED}Error: Java is not installed${NC}"
    echo "Please install Java 11 or higher"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
echo -e "Java version: ${YELLOW}$(java -version 2>&1 | head -n 1)${NC}"

# Create output directory
OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"

# Compile Java files
echo ""
echo -e "${YELLOW}Compiling Java sources...${NC}"

SOURCES=$(find "$SCRIPT_DIR/src" -name "*.java")
if [ -z "$SOURCES" ]; then
    echo -e "${RED}Error: No Java source files found${NC}"
    exit 1
fi

javac -d "$OUT_DIR" $SOURCES 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"

# Run the application
echo ""
echo -e "${YELLOW}Starting Multi-Agent RL Training...${NC}"
echo ""

# Default configuration for best results
# You can modify these parameters for different scenarios
java -cp "$OUT_DIR" rl.Main \
    --agents 3 \
    --width 10 \
    --height 10 \
    --food 15 \
    --obstacles 5 \
    --max-steps 200 \
    --train-episodes 2000 \
    --eval-episodes 10 \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Execution completed successfully!${NC}"
else
    echo -e "${RED}Execution failed with exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
