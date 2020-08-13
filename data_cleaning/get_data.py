import subprocess

with open("links.txt", 'r') as f:
    for line in f:
        print(line.strip())
        subprocess.run(f'wget -P data/ {line.strip()}', shell=True)
