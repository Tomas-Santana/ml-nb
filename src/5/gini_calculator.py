import questionary

def gini_impurity(sample_sizes: list[int]) -> float:
    total_samples = sum(sample_sizes)
    if total_samples == 0:
        return 0.0
    return 1 - sum((size / total_samples) ** 2 for size in sample_sizes)

def validate_integer(value):
  if value.lower() == 'end':
      return True
  try:
      int(value)
      return True
  except ValueError:
      return False

if __name__ == "__main__":
    sample_sizes = []
    
    while True:
        size = questionary.text("Enter the sample size for a class (or 'end' to finish):", validate=validate_integer).ask()
        if size.lower() == 'end':
            break
        sample_sizes.append(int(size))
    
    gini = gini_impurity(sample_sizes)
    print(f"Gini Impurity: {gini:.4f}")
    print("Sample sizes:", sample_sizes)
    