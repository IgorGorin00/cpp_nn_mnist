#!/usr/bin/bash

declare -A LINKS=(
    ["train_images"]="https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz"
    ["train_labels"]="https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz"
    ["test_images"]="https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz"
    ["test_labels"]="https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz"
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
