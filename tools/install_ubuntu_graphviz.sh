#!/bin/sh

# Install a functional version of `graphviz` including the plugins we use.  Requires root.

set -e

apt-get update
apt-get install -y graphviz libgvplugin-neato-layout8
