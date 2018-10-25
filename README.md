# ClowdFlows NLP Module #


A [ClowdFlows](https://github.com/xflows/clowdflows/) package, which contains widgets for natural language processing. The package can also be used with [ClowdFlows](https://github.com/xflows/clowdflows/) 2.0.

[![Documentation Status](https://readthedocs.org/projects/rdm/badge/?version=latest)](http://clowdflows.readthedocs.io/)

Currently, the project contains components for different corpus operations, basic natural language processing operations such as tokenization, stop word removal, lemmatization, part-of-speech tagging, etc. It also has modules for tweet streaming, term extraction and gender classification.


## Installation, documentation ##

Since three pickled models are too big for github, you have to download the following models manually from external links and add them to the cf_nlp/nlp/models/reldi_tagger subfolder in order to make Reldi tagger and Reldi lemmatizer work:

* http://nlp.ffzg.hr/data/reldi/hr.lexicon.guesser
* http://nlp.ffzg.hr/data/reldi/sr.lexicon.guesser
* http://nlp.ffzg.hr/data/reldi/sl.lexicon.guesser

You can also download a wheel with all the models inside from:

* http://kt.ijs.si/matej_martinc/cf_nlp-0.0.16-py2-none-any.whl

In order to use the NLTK tagger widget, you have to download the following model separately using the 'nltk.download()' command:

* averaged perceptron tagger

For additional instructions, see: https://www.nltk.org/data.html

Please find other installation instructions, examples and API reference on [Read the Docs](http://clowdflows.readthedocs.io/).

## Note ##

Please note that this is a research project and that drastic changes can be (and are) made pretty regularly. Changes are documented in the [CHANGELOG](CHANGELOG.md).

Pull requests and issues are welcome.

## Contributors to the cf_nlp package code ##

Matej Martinc (@matejMartinc), Andraž Pelicon (@PeliconA)

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
