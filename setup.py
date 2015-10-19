from setuptools import setup
from setuptools.command.install import install
import os
class CustomInstallCommand(install):
    def run(self):
        mp = os.path.dirname(os.path.abspath(__file__))
        mp = os.path.join(mp, 'zrst')
        with open(os.path.join(mp,'m_path.py'),'w') as f:
            f.write("matlab_path='{}'".format(\
                os.path.join(mp,'matlab')))
        install.run(self)

setup(cmdclass={'install':CustomInstallCommand},name='zrst',version="0.3",description='Library that transcribes wav files into sequences of subword symbols. Requires HTK, SRILM, python, numpy, scipy.',url='https://github.com/C2Tao/zrst/',author='C2Tao',author_email='b97901182@gmail.com',license='MIT',package_dir={'zrst':'zrst'},packages=['zrst'],zip_safe=False)
