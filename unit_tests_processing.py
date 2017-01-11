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

    def testMNISTExpected(self):
        inp_args = ["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                    "0", "0", "0"]

        rc, ans, err = run(inp_args)
        parsed_output = ans.split("\n")
        
        train_data_size = extractInteger(parsed_output[0])
        train_lbl_size = extractInteger(parsed_output[1])
        test_data_size = extractInteger(parsed_output[2])
        test_lbl_size = extractInteger(parsed_output[3])

        self.assertEqual(train_data_size, train_lbl_size)
        self.assertEqual(test_data_size, test_lbl_size)


    def testPPMExpected(self):

        #the numers are flags for 
        #"manipulate data flag" 0/1 (can change data to be bad to check if assertions catch it)
        #"processing type flag" 0/1/2 no processing / gaussian / histogram
        #"image type flag" 0/1/2 mnist / ppm/ jpg
        inp_args = ["alpha_driver", 'data/cars/training/', 'data/cars/testing/',
                    'dummy', 'dummy','dummy','dummy',
                    "0","0","1"]

        

    def testBadArgs(self):

        inp_args = ["garbage 1", "garbage 2", "garbage 3", "garbage 4", "garbage 5",
                    "garbage 6", "garbage 7", "garbage 8", "garbage 9"]

        try:
            rc, ans, err = run(inp_args);
        except:
            xdumb = 1 # just do something so Python doesn't complain
        else:
            raise Exception("Garbage input arguments should have been caught!")

    def testMNISTBadVals(self):

        inp_args = ["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", 
                                    "1", "0", "0"]

        rc, ans, err = run(inp_args);
        # since the assertion is in a sub-function, we need to handle this a little differently
        if not err:
            raise Exception("Bad pixel should have been caught!")

        # Now we have finished testing the input data.  Next is the pre-processed data
        
    def testGaussianExpected(self):

        inp_args = ["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                    "0", "1", "0"]

        rc, ans, err = run(inp_args);
        parsed_output = ans.split("\n")

        train_data_size = extractInteger(parsed_output[0])
        test_data_size = extractInteger(parsed_output[2])
        gauss_train_size = extractInteger(parsed_output[4])
        gauss_test_size = extractInteger(parsed_output[5])

        self.assertEqual(train_data_size, gauss_train_size)
        self.assertEqual(test_data_size, gauss_test_size)

    def testHistogramExpected(self):

        inp_args = ["alpha_driver", "data/mnist/training/", "data/mnist/testing/",
                                    "train-labels.idx1-ubyte", "train-images.idx3-ubyte",
                                    "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                    "0", "2", "0"]

        rc, ans, err = run(inp_args);
        parsed_output = ans.split("\n")

        train_data_size = extractInteger(parsed_output[0])
        test_data_size = extractInteger(parsed_output[2])
        hist_train_size = extractInteger(parsed_output[4])
        hist_test_size = extractInteger(parsed_output[5])

        self.assertEqual(train_data_size+1, hist_train_size)
        self.assertEqual(test_data_size+1, hist_test_size)



if __name__ == '__main__':
    unittest.main()
