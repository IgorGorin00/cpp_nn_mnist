#!/usr/bin/bash

declare -A LINKS=(
    ["train_images"]="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    ["train_labels"]="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    ["test_images"]="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    ["test_labels"]="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
);

echo "Downloading MNIST dataset files...";

for LINK in "${!LINKS[@]}"; do
    FILE="${LINK}";
    URL="${LINKS[$LINK]}";
    wget -N "$URL" -O "./$FILE.gz";
    gunzip "./$FILE.gz";
    mv "./$FILE" "./${FILE}.ubyte";
done

echo "MNIST dataset files downloaded and extracted successfully."
