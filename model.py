import torch
import torch.nn as nn
import importlib
#==========================================================================#
def remove_layer_SqueezeNet(model, num_classes):
	modules = model.modules()
	for m in modules:
		if isinstance(m, nn.Conv2d) and m.weight.data.size()[0]==1000:
			w1 = m.weight.data[:num_classes]
			b1 = m.bias.data[:num_classes]
	mod = list(model.classifier)
	mod.pop(1)
	mod.insert(1, nn.Conv2d(512, num_classes, kernel_size=1, stride=1))
	new_classifier = torch.nn.Sequential(*mod)
	model.classifier = new_classifier
	modules = model.modules()
	flag = False
	for m in modules:
		if isinstance(m, nn.Conv2d) and m.weight.data.size()[0]==num_classes:
			m.weight.data = w1
			m.bias.data = b1	
			flag = True
	assert flag

#==========================================================================#
def remove_last_layer_FC(model, num_classes):
	modules = model.modules()
	for m in modules:
		if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1000:
			w1 = m.weight.data[:num_classes]
			b1 = m.bias.data[:num_classes]
	try:
		if type(model.classifier)==nn.Sequential:
			mod = list(model.classifier) #Alexnet, VGG
		else:
			mod = [model.classifier] #DenseNet

	except: 
		mod = [model.fc] #ResNet

	weight = mod[-1].weight.size(1)
	mod.pop()
	mod.append(torch.nn.Linear(weight,num_classes))
	new_classifier = torch.nn.Sequential(*mod)
	model.classifier = new_classifier
	modules = model.modules()
	flag = False
	for m in modules:
		if isinstance(m, nn.Linear) and m.weight.data.size()[0]==num_classes:
			m.weight.data = w1
			m.bias.data = b1	
			flag = True
	assert flag

#==========================================================================#
def print_network(model, name):
		num_params=0
		for p in model.parameters():
				num_params+=p.numel()
		# print(name)
		# print(model)
		print("The number of parameters {}: {}".format(name, num_params))

#==========================================================================#
def get_model(name, num_classes):
	remove_layer = remove_layer_SqueezeNet if name=='squeezenet' else remove_last_layer_FC
	general_name = filter(str.isalpha, name.split('_')[0])
	if name=='squeezenet':
		from models.squeezenet import squeezenet1_1 as net
	else:
		exec "from torchvision.models.{} import {} as net".format(general_name, name)

	model = net(pretrained=True) 
	print_network(model, '{} - ImageNet (1000 outputs)'.format(name.upper()))	
	remove_layer(model, num_classes)
	print_network(model, '{} - Custom ({} outputs)'.format(name.upper(), num_classes))	

	return model