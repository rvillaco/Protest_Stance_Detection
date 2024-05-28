import subprocess

## run this to generate (imbalanced) heterogenous results.
## params can be changed in config.py.

if __name__ == '__main__':
    subprocess.run(['python3', 'homo_train_balanced.py'])
    subprocess.run(['python3', 'homo_inference_balanced.py'])
    subprocess.run(['python3', 'r_print_results.py'])