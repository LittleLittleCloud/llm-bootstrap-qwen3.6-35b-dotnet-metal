#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
xcrun clang++ -std=c++17 -O3 -fobjc-arc -dynamiclib -framework Foundation -framework Metal bridge.mm -o build/libbootstrap_metal.dylib
