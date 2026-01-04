#!/bin/bash
#
# Argbash template for download_results.sh
#
# Generate the script with:
#   argbash download_results.m4 -o download_results.sh
#
# Reference: https://argbash.readthedocs.io/en/stable/guide.html

# ARG_POSITIONAL_SINGLE([destination], [Local destination directory for downloaded results])
# ARG_OPTIONAL_SINGLE([host], [H], [SSH host in user@hostname format (required)], [])
# ARG_OPTIONAL_SINGLE([remote-dir], [r], [Remote results directory on cluster], [\$SCRATCH/quariadne-benchmark])
# ARG_OPTIONAL_BOOLEAN([dry-run], [n], [Show what would be transferred without executing], [off])
# ARG_OPTIONAL_BOOLEAN([logs-only], [l], [Download only SLURM logs, skip results], [off])
# ARG_OPTIONAL_BOOLEAN([results-only], [R], [Download only results, skip SLURM logs], [off])
# ARG_HELP([Download benchmark results from Eiger cluster using rsync])
# ARGBASH_GO

# [ <-- needed because of Argbash potential parsing issues

echo "Parsed arguments:"
echo "  destination:  ${_arg_destination}"
echo "  host:         ${_arg_host}"
echo "  remote-dir:   ${_arg_remote_dir}"
echo "  dry-run:      ${_arg_dry_run}"
echo "  logs-only:    ${_arg_logs_only}"
echo "  results-only: ${_arg_results_only}"

# ] <-- needed because of Argbash potential parsing issues
