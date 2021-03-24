 from setuptools import setup, find_packages

 setup(
     name                = 'haneda',
     version            = '0.1',
     description        = 'haneda',
     author              = 'Daniel',
     author_email        = 'deepvine20@gmail.com',
     install_requires    =  [],
     packages            = find_packages(exclude = []),
     long_description    = open("README.md").read(),
     keywords            = ['ccpy'],
     package_data        = {},
     zip_safe            = False
 )