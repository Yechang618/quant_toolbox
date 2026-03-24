#!/bin/bash
#
# run_factor_gen.sh
# Shell script wrapper for running the feature generation pipeline
#
# Usage:
#   ./run_factor_gen.sh [OPTIONS]
#
# Options:
#   -s, --symbol SYMBOL     Process specific symbol (can be used multiple times)
#   -d, --date DATE         Process specific date in YYYYMMDD format (can be used multiple times)
#   -m, --mode MODE         Process specific trade mode: 0, 2, or all (default: all)
#   --dry-run               Run without saving output files
#   --log-level LEVEL       Set logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
#   --market-root PATH      Override market data root path
#   --output-dir PATH       Override output directory path
#   -h, --help              Show this help message
#
# Examples:
#   ./run_factor_gen.sh -s 1MBABYDOGE -d 20251227
#   ./run_factor_gen.sh -s BTC -s ETH --dry-run --log-level DEBUG
#   ./run_factor_gen.sh --market-root /data/market --output-dir /data/output

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
VENV_PATH="${PROJECT_ROOT}/.venv2"
PYTHON_CMD="python"
LOG_LEVEL="INFO"
DRY_RUN=""
MARKET_ROOT=""
OUTPUT_DIR=""
SYMBOLS=()
DATES=()
MODES=()

# Colors for output (if terminal supports it)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

# Show help message
show_help() {
    grep -A 100 "^# Usage:" "$0" | grep -v "^# \$" | sed 's/^# \?//'
    exit 0
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -s|--symbol)
                SYMBOLS+=("$2")
                shift 2
                ;;
            -d|--date)
                DATES+=("$2")
                shift 2
                ;;
            -m|--mode)
                MODES+=("$2")
                shift 2
                ;;
            --dry-run)
                DRY_RUN="--dry-run"
                shift
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --market-root)
                MARKET_ROOT="--market-root $2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="--output-dir $2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Activate virtual environment if it exists
    if [[ -d "$VENV_PATH" ]]; then
        log_info "Activating virtual environment: $VENV_PATH"
        source "$VENV_PATH/bin/activate"
        PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
        log_info "Using system python3"
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        log_info "Using system python"
        PYTHON_CMD="python"
    else
        log_error "No Python interpreter found"
        exit 1
    fi
    
    # Verify Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    log_info "Using $PYTHON_VERSION"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    log_info "Working directory: $(pwd)"
}

# Build command arguments
build_command() {
    local cmd="$PYTHON_CMD script/run_factor_gen.py"
    
    # Add symbol arguments
    for symbol in "${SYMBOLS[@]}"; do
        cmd="$cmd --symbol $symbol"
    done
    
    # Add date arguments
    for date in "${DATES[@]}"; do
        cmd="$cmd --date $date"
    done
    
    # Add mode filter if specified
    if [[ ${#MODES[@]} -gt 0 ]]; then
        local mode_filter=$(IFS=,; echo "${MODES[*]}")
        log_warning "Mode filtering via shell script is limited. Use trade_mode column filtering in post-processing."
    fi
    
    # Add optional arguments
    [[ -n "$DRY_RUN" ]] && cmd="$cmd $DRY_RUN"
    [[ "$LOG_LEVEL" != "INFO" ]] && cmd="$cmd --log-level $LOG_LEVEL"
    [[ -n "$MARKET_ROOT" ]] && cmd="$cmd $MARKET_ROOT"
    [[ -n "$OUTPUT_DIR" ]] && cmd="$cmd $OUTPUT_DIR"
    
    echo "$cmd"
}

# Run the pipeline
run_pipeline() {
    local cmd
    cmd=$(build_command)
    
    log_info "Executing command:"
    echo "  $cmd"
    echo ""
    
    # Create logs directory if needed
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Run with output logging
    local log_file="$PROJECT_ROOT/logs/factor_gen_$(date '+%Y%m%d_%H%M%S').log"
    
    if [[ -n "$DRY_RUN" ]]; then
        log_warning "DRY RUN MODE - No files will be saved"
    fi
    
    log_info "Logging to: $log_file"
    
    # Execute and capture output
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        log_success "Pipeline completed successfully"
        log_info "Output log: $log_file"
        
        # Show output summary if not dry run
        if [[ -z "$DRY_RUN" ]]; then
            log_info "Checking output files..."
            for mode in 0 2; do
                local output_dir="$PROJECT_ROOT/dataset/preprocessed/mode$mode"
                if [[ -d "$output_dir" ]]; then
                    local count=$(find "$output_dir" -name "*.csv" -type f 2>/dev/null | wc -l)
                    if [[ $count -gt 0 ]]; then
                        log_info "  mode$mode: $count file(s) generated"
                    fi
                fi
            done
        fi
        return 0
    else
        log_error "Pipeline failed with exit code $?"
        log_error "Check log file for details: $log_file"
        return 1
    fi
}

# Main execution
main() {
    echo "========================================"
    echo "  Feature Generation Pipeline Runner"
    echo "========================================"
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    # Setup environment
    setup_environment
    
    # Show configuration
    log_info "Configuration:"
    [[ ${#SYMBOLS[@]} -gt 0 ]] && log_info "  Symbols: ${SYMBOLS[*]}" || log_info "  Symbols: all"
    [[ ${#DATES[@]} -gt 0 ]] && log_info "  Dates: ${DATES[*]}" || log_info "  Dates: all"
    log_info "  Log level: $LOG_LEVEL"
    [[ -n "$DRY_RUN" ]] && log_info "  Mode: DRY RUN"
    echo ""
    
    # Run pipeline
    if run_pipeline; then
        exit 0
    else
        exit 1
    fi
}

# Trap keyboard interrupt
trap 'echo ""; log_warning "Interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"