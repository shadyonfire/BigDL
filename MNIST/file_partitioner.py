import idx2numpy
import sys
import os
import shutil

size=int(sys.argv[1])
folder_name="./MNIST_"+str(size)
try:
    os.mkdir(folder_name)
except OSError as err:
    print(err)

print("folder created\n---------------------")


print("partitioning images\n--------------------")
nump=idx2numpy.convert_from_file("train-images-idx3-ubyte")
print("Before images:",nump.shape)
nump=nump[:size]
print("After images:",nump.shape)
nump=idx2numpy.convert_to_file(folder_name+"/train-images-idx3-ubyte",nump)
print("partitioning labels\n----------------------")
nump=idx2numpy.convert_from_file("train-labels-idx1-ubyte")
print("Before labels:",nump.shape)
nump=nump[:size]
print("After labels:",nump.shape)
nump=idx2numpy.convert_to_file(folder_name+"/train-labels-idx1-ubyte",nump)
print("Done\n------------------------------------")


print("Copying validation file\n-------------------------")

shutil.copy("./t10k-images-idx3-ubyte",folder_name+"/t10k-images-idx3-ubyte")
shutil.copy("./t10k-labels-idx1-ubyte",folder_name+"/t10k-labels-idx1-ubyte")

print("Done\n----------------")


