"""
Computer Systems Architecture - Lab 3

Threadpool assignment:
    Use a pool of threads to search for a DNA sequence in a list of DNA samples
"""
from concurrent.futures import ThreadPoolExecutor as ThreadPool
import random
from os import urandom

random.seed(urandom(64))

print("DNA search")

NR_DNA_SAMPLES = 1000
DNA_SAMPLE_LENGTH = 10000
SEARCH_SEQUENCE_LENGTH = 10

MAX_WORKERS = 30

dna_samples = ["".join([random.choice("ACGT") for _ in range(DNA_SAMPLE_LENGTH)]) for _ in range(NR_DNA_SAMPLES)]

SEARCH_SEQUENCE  = "".join([random.choice("ACGT") for _ in range(SEARCH_SEQUENCE_LENGTH)])

def search_dna_sequence(sequence : str, sample : str) -> bool:
    return sequence in sample


def thread_job(sample_index : int) -> tuple[bool, str]:
    if search_dna_sequence(SEARCH_SEQUENCE, dna_samples[sample_index]):
        return True, f"DNA sequence found in sample {sample_index}"
    return False, f"DNA sequence not found in sample {sample_index}"


if __name__ == "__main__":

    thread_pool = ThreadPool(max_workers=MAX_WORKERS)

    futures = []

    with thread_pool:
        for i in range(NR_DNA_SAMPLES):
            futures.append(thread_pool.submit(thread_job, i))

    result = [future.result() for future in futures if future.result()[0] == True]
    print(result[0][1] if result else "DNA sequence not found in any sample")