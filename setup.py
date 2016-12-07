cwd = os.getcwd()
#Compile Processing
os.chdir('processing')
os.system('make')
os.chdir(cwd)
#Compile Models
cwd = os.getcwd()
os.chdir('models')
os.system('make')
os.chdir(cwd)
