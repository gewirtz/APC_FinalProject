import subprocess

def run(cmd):
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stederr = subprocess.PIPE)
    ans, err = p.communicate()
    rc = p.wait()
    return rc, ans, err

mnist_dirs =["data/mnist/training/",  "data/mnist/testing/"];
mnist_train_fnames = [ "train-images.idx3-ubyte", "train-labels.idx1-ubyte"];
mnist_test_fnames = ["t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"];

rc, ans, err = run(["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte"])
