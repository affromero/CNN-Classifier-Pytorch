import torch
from torchvision import datasets, transforms
from PIL import Image
#==========================================================================#
def get_data(config):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																	std=[0.229, 0.224, 0.225])
	transform_train = transforms.Compose([transforms.Resize((config.image_size, config.image_size), interpolation=Image.ANTIALIAS), \
										  transforms.RandomHorizontalFlip(),
										  transforms.ToTensor(), \
										  normalize])
	transform_test = transforms.Compose([transforms.Resize((config.image_size, config.image_size), interpolation=Image.ANTIALIAS), \
										  transforms.ToTensor(), \
										  normalize])

	data_train = datasets.ImageFolder('data/train_128', transform=transform_train)
	train_loader = torch.utils.data.DataLoader(data_train, batch_size=config.batch_size, \
																												 shuffle=True, \
																												 num_workers=config.num_workers)

	data_val = datasets.ImageFolder('data/val_128', transform=transform_test)
	val_loader = torch.utils.data.DataLoader(data_val, batch_size=config.batch_size, \
																										 shuffle=False, \
																										 num_workers=config.num_workers)

	data_test = datasets.ImageFolder('data/test_128', transform=transform_test)
	test_loader = torch.utils.data.DataLoader(data_test, batch_size=config.batch_size, \
																											 shuffle=False, \
																											 num_workers=config.num_workers)
	return train_loader, val_loader, test_loader