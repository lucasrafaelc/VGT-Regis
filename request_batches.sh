#!/bin/bash

# Set variables
ServerUser="user_vgt"
ServerIP="geodigital.inf.ufrgs.br"
ServerScriptPath="batch_manager.sh"
NumBatches="$1"  # Get the first argument passed to the script
BatchPathsFile="BatchFilePaths.txt"

# Check if the number of batches is provided
if [ -z "$NumBatches" ]; then
    echo "Please provide the number of batches to request."
    exit 1
fi

# Step 1: Retrieve the batch files from the server, but don't move them yet
echo "Requesting $NumBatches batches from the server..."
BatchFiles=$(ssh "$ServerUser@$ServerIP" "bash $ServerScriptPath $NumBatches")

# Save the list of batch files into an array
IFS=$'\n' read -rd '' -a BatchFilesArray <<< "$BatchFiles"

# Step 2: Save the paths to the batch files instead of downloading them
echo "Saving the paths of the batch files..."
for BatchFile in "${BatchFilesArray[@]}"; do
    BatchFile=$(echo "$BatchFile" | xargs)  # Trim whitespace
    if [ -n "$BatchFile" ]; then
        BatchFilePath="processed_batches/$BatchFile"
        echo "Saving path: $BatchFilePath"
        
        # Save the path to a file
        echo "$BatchFilePath" >> "$BatchPathsFile"
    fi
done

echo "Paths saved to $BatchPathsFile."
