from datasets import load_dataset

print("Loading English Amazon reviews...")

dataset = load_dataset("goosmanlei/amazon_reviews_multi", "en")

print("\nDataset splits:")
print(dataset)

print("\nFirst training example:")
print(dataset["train"][0])

print("\nAvailable columns:")
print(dataset["train"].column_names)