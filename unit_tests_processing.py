import subprocess
import unittest

def run(cmd):
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    ans, err = p.communicate()
    rc = p.wait()
    return rc, ans, err
 
def extractInteger(in_str):
    return int(''.join(x for x in in_str if x.isdigit()))

# I admit, this unit test implementation is pretty awful.  But here we go.
class TestProcess(unittest.TestCase):

    def testExpected(self):
        inp_args = ["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", str(0)]

        rc, ans, err = run(inp_args)
        parsed_output = ans.split("\n")
        
        train_data_size = extractInteger(parsed_output[0])
        train_lbl_size = extractInteger(parsed_output[1])
        test_data_size = extractInteger(parsed_output[2])
        test_lbl_size = extractInteger(parsed_output[3])

        self.assertEqual(train_data_size, train_lbl_size)
        self.assertEqual(test_data_size, test_lbl_size)

        

    def testBadArgs(self):

        inp_args = ["garbage 1", "garbage 2", "garbage 3", "garbage 4", "garbage 5",
                    "garbage 6", "garbage 7", "garbage 8", str(0)]

        try:
            rc, ans, err = run(inp_args);
            print err
        except:
            xdumb = 1 # just do something so Python doesn't complain
        else:
            raise Exception("Garbage input arguments should have been caught!")

    def testBadVals(self):

        inp_args = ["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "1"]

        rc, ans, err = run(inp_args);
        # since the assertion is in a sub-function, we need to handle this a little differently
        if not err:
            raise Exception("Bad pixel should have been caught!")



if __name__ == '__main__':
    unittest.main()
