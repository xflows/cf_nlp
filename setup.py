import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='cf_nlp',
    version='0.1.6',
    packages=['nlp'],
    include_package_data=True,
    license='MIT License',
    description='ClowdFlows natural language processing module',
    long_description=README,
    url='https://github.com/xflows/cf_nlp',
    author='Matej Martinc',
    author_email='matej.martinc@ijs.si',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries'
    ],
    install_requires=[
        'pandas==0.20.1',
        'langid==1.1.6',
		'tweepy==3.5.0',
		'marisa-trie==0.7.4',
		'python-crfsuite==0.9',
		'reldi==1.6',
        'theano',
        'keras==2.2.2',
        'lemmagen==1.2.0',
        'editdistance==0.4',

    ]
)
