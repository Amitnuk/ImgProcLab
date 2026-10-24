#!/usr/bin/env zsh


# PROJECT DIRECTORY
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESOURCE_DIR="${ROOT_DIR}/resources"
BIN_ROOT_DIR="${ROOT_DIR}/bin"
BIN_DIR="${BIN_ROOT_DIR}/Linux/64/Release"
EXECUTABLE="${ROOT_DIR}/bin/Linux/64/Release/ImgProcLabDemo"


rm -rf "${BIN_ROOT_DIR}" 
rm -rf "${BUILD_DIR}"/*

if [[ ! -e ${BUILD_DIR} ]]; then 
    echo "Creating the ${BUILD_DIR} directory"
    mkdir ${BUILD_DIR}
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DPROFILE_CUDA=OFF
cmake --build "${BUILD_DIR}"

 
# Benchmake configuration 
ALGORITHM="gaussian"
VERSIONS=(v1 v2 v3)
KERNEL_SIZES=(3 5 7 9 11 13 15 17 19 21 23 25 27 29 31)
SIGMA=2.0

WARMAPS=10
ITERATIONS=100 

IMAGES=("${RESOURCE_DIR}/parrot.png"
        "${RESOURCE_DIR}/einstein.jpg"
        "${RESOURCE_DIR}/Car.jpeg"
        "${RESOURCE_DIR}/Sunset.jpg")



# ! logical not 
# -x execuyable file
# >&2 redirect thz output ro stderr instead of stdout
if [[ ! -x "$EXECUTABLE" ]]; then 
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 1
fi


for image in "${IMAGES[@]}"; do
    if [[ ! -e "$image" ]]; then 
        echo "Image not found: $image" >&2
        exit 1
    fi
    echo "Image: $image"

    for version in "${VERSIONS[@]}"; do
        for kernel_size in "${KERNEL_SIZES[@]}"; do
            echo "---------------------------------"
            echo "Image $(basename "$image")"
            echo "Version "$version""
            echo "Kernel size "$kernel_size""
            echo "Sigma "$SIGMA""
            echo "---------------------------------"
            
            "$EXECUTABLE" \
                --image "$image" \
                --algo "$ALGORITHM" \
                --version "$version" \
                --kernel-size "$kernel_size" \
                --sigma "$SIGMA" \
                --benchmark \
                --iterations "$ITERATIONS" \
                --warmups "$WARMAPS"
        done
    done
done
