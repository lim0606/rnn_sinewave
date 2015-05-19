import os

filenames = os.listdir(".")

for filename in filenames:
    if "weight_%d" % 1 in filename:
        new_filename = filename.split("weight_%d" % 1)[0] + "weight_input_dim_%d_hidden_%d" % (1, 10) + filename.split("weight_%d" % 1)[1]
        os.rename(filename, new_filename)
    elif "weight_%d" % 5 in filename: 
        new_filename = filename.split("weight_%d" % 5)[0] + "weight_input_dim_%d_hidden_%d" % (5, 10) + filename.split("weight_%d" % 5)[1]
        os.rename(filename, new_filename)


