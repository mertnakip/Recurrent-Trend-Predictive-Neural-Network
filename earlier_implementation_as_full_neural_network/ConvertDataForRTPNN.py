
def convert_data_for_rTPNN(x_array):
	# x_array.shape = (T, 2, I)
	I = x_array.shape[-1]
	x_list = []
	for i in range(I):
		x_list.append(x_array[:, :, i:i+1])
	return x_list
