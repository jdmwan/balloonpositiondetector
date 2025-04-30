from collections import Counter
from torchvision.datasets import ImageFolder
train_dataset = ImageFolder("BalloonDataset/train")
class_counts = Counter(train_dataset.targets)
print(class_counts)
print(train_dataset.class_to_idx)
