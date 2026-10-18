# Shared settings for the repair scripts (sourced).
#   COVENANT_ROOT  checker root (default: this repository)
#   MD             model driver build dir (default: $COVENANT_ROOT/model_driver/build)
#   WORK           work dir: src/ (sources and builds), runs/ (traces and checks), cases/ (summaries)
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPAIRS=$(cd "$SCRIPTS/.." && pwd)
COVENANT_ROOT=$(cd "${COVENANT_ROOT:-$REPAIRS/../..}" && pwd)
MD=$(readlink -m "${MD:-$COVENANT_ROOT/model_driver/build}")
: "${WORK:?set WORK to a work directory}"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
T=$WORK/src
PATCHES=$REPAIRS/patches
export COVENANT_ROOT MD WORK
