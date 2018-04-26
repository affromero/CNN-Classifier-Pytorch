#!/usr/local/bin/ipython
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
import os, glob, argparse, tqdm
from data_loader import get_data
from model import get_model
#==========================================================================#
def create_folder(folder):
	if not os.path.isdir(folder): os.makedirs(folder)

#==========================================================================#
def load_pretrained_model(model, name):
	try:
		dir_ = sorted(glob.glob(os.path.join('snapshot', name, '*.pth')))[-1]
	except:
		return 0
	model.load_state_dict(torch.load(dir_))
	print('Loaded trained model: {}!'.format(dir_))
	return int(os.path.basename(dir_).split('.')[0])

#==========================================================================#
def save_model(model, name, epoch):
	dir_ = os.path.join('snapshot', name, '%s.pth'%(str(epoch).zfill(4)))
	create_folder(os.path.dirname(dir_))
	torch.save(model.state_dict(), dir_)
	print('!!Saving model: {}!'.format(dir_))
				
#==========================================================================#				
def update_lr(lr, optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

#==========================================================================#				
def to_cuda(data):
	if torch.cuda.is_available():
		data = data.cuda()
	return data

#==========================================================================#
def solver(name, data_loader, model, epoch, optimizer=None, mode='train'):
		if optimizer is None: model.eval()
		else: model.train()
		loss_cum = []
		Acc = 0
		count_test = 0
		test_out = []
		Loss = nn.CrossEntropyLoss()
		for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="!{} -> [{}] Epoch: {}".format(name.upper(), mode.upper(),epoch)):
				volatile = True if optimizer is None else False
				data = Variable(to_cuda(data), volatile=volatile)
				target = Variable(to_cuda(target), volatile=volatile)

				output = model(data)
				loss = Loss(output,target)	 

				if optimizer is not None:
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				loss_cum.append(loss.data.cpu()[0])
				_, arg_max_out = torch.max(output.data.cpu(), 1)
				if mode=='test':
					for oo in arg_max_out:
						test_out.append('%s,%d\n'%(str(count_test).zfill(4), oo))
						count_test+=1
				Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
		ACC = float(Acc*100)/len(data_loader.dataset)
		LOSS = np.array(loss_cum).mean()
		if mode=='test':
			f=open(os.path.join('snapshot', name, 'test.txt'),'w')
			for line in test_out: f.writelines(line)
			f.close()
		else:
			print("LOSS %s: %0.3f || ACC %s: %0.2f"%(mode.upper(), LOSS, mode.upper(), ACC))
		
		return ACC

#==========================================================================#
def train(config, train_loader, val_loader, model):
	val_before = 0
	for epoch in range(config.start_epoch, config.num_epochs):
		_, solver(config.model, train_loader, model, epoch, optimizer=config.optimizer, mode='train')
		val_acc = solver(config.model, val_loader, model, epoch, mode='val')

		if val_acc>val_before:
			save_model(model, config.model, epoch+1)
			val_before=val_acc
			flag_stop=0
		else:
			flag_stop+=1

		if flag_stop==config.stop_training: 
			return

		# Decay learning rate
		if (epoch+1) > (config.num_epochs - config.num_epochs_decay):
			config.lr -= (config.lr / float(config.num_epochs_decay))
			update_lr(config.lr, config.optimizer)
			print ('Decay learning rate to: {}.'.format(config.lr))

#==========================================================================#
def test(config, val_loader, test_loader, model):
	assert start_epoch>0, "you must first TRAIN"
	solver(config.model, val_loader, model, config.start_epoch, mode='val')
	solver(config.model, test_loader, model, config.start_epoch, mode='test')

#==========================================================================#
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_size', type=int, default=224)
	parser.add_argument('--lr', type=float, default=0.001)

	# Training settings
	parser.add_argument('--batch_size', type=int, default=128)	
	parser.add_argument('--num_epochs', type=int, default=59)
	parser.add_argument('--num_epochs_decay', type=int, default=60)
	parser.add_argument('--stop_training', type=int, default=3, help='Stops after N epochs if acc_val is not increasing')
	parser.add_argument('--num_workers', type=int, default=4)	
	parser.add_argument('--model', type=str, default='densenet201')	
	parser.add_argument('--TEST', action='store_true', default=False)
	config = parser.parse_args()

	train_loader, val_loader, test_loader = get_data(config)
	#Train, Val, Test loaders that are foun in './data'

	num_classes=len(train_loader.dataset.classes)
	#Numbet of classes

	model = get_model(config.model, num_classes) 
	#Returns the model and the batch size that fits in a 4GB GPU
	
	if torch.cuda.is_available(): model.cuda()
	
	#============================Optimizer==================================#
	config.optimizer = torch.optim.Adam(model.parameters(), config.lr, [0.5, 0.999])

	#================= Look if there is a previous snapshot ================#
	config.start_epoch = load_pretrained_model(model, config.model)

	if config.TEST:
		test(config, val_loader, test_loader, model)
	else:
		train(config, train_loader, val_loader, model)
		#Train until VALIDATION convergence, i.e., stops after -confign.stop_training- plateau region

		test(config, val_loader, test_loader, model)
