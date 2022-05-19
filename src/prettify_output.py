# Reads the Output file from the fastai and removes all the weird
# ascii characters and replaces them with the correct ones.

def prettify(file_handle):
    # Read the file
    lines = file_handle.readlines()
    # Remove all full block U+2588 characters
    lines = [line.replace('\u2588', '') for line in lines]
    # Remove all empty lines (lines with only whitespace)
    lines = [line for line in lines if line.strip()]
    
    # Return a string with all the prettified content
    return ''.join(lines)

if __name__ == '__main__':
    import os
    files = [f for f in os.listdir('./models/resnet_mixup') if f.startswith('Output') and f.endswith('.out')]
    if len(files) == 0:
        print("No file found")
        exit(1)
    elif len(files) > 1:
        print("More than one file found")
        exit(1)
    
    print(prettify(files[0]))