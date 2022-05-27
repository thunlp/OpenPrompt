.. _How_to_write_a_verbalizer:

How to Write a Verbalizer?
=============================

Input Label Words
~~~~~~~~~~~~~~~~~~

For manual verbalizers which we need to specify a set of label words manually, 
we provide multiple input format supports.

.. code-block:: python

    ManualVerbalizer(
        label_words = label_words,
        ...
    )


You can input the label words as a list, for example, 

.. code-block:: python

    ["politics", "technology"]

For a class with multiple label words for a class, input as a nested list.

.. code-block:: python

    [["bad", "terrible"], ["good"]]

You can also input the label words as a dict, this is recommend when your
dataset has explicit name for each class (especially when the number of categories is large).

.. code-block:: python

    {"World": "politics", "Tech": "technology"}

or 

.. code-block:: python

    { 
        "person-scholar": ["scholar", "scientist"],
        "building-library": ["library"], 
        "building-hotel": ["hotel"],
        "location-road/railway/highway/transit": ["road", "railway", "highway", "transit"]
    }

Loading from File
~~~~~~~~~~~~~~~~~~
    
When loading from files, we support various file formats.

.. code-block:: python

    ManualVerbalizer(...).from_file(file_path = file_path)

For the ``dict-like`` input format, we allow `.json` or `.jsonl` file.
We can use a single verbalizer, i.e. (one group of label words), e.g.:

.. code-block:: python
    
    { 
        "person-scholar": ["scholar", "scientist"],
        "building-library": ["library"], 
        "building-hotel": ["hotel"],
        "location-road/railway/highway/transit": ["road", "railway", "highway", "transit"]
    }

or multiple verbalizers (with different label-word sets) , in a dict format, e.g.:

.. code-block:: python

    [
        {
            "person-scholar": ["scholar", "scientist"],
            "building-library": ["library"], 
            "building-hotel": ["hotel"],
            "location-road/railway/highway/transit": ["road", "railway", "highway", "transit"]
        },
        {
            "person-scholar": ["scientist],
            "building-library": ["library"], 
            "building-hotel": ["hotel"],
            "location-road/railway/highway/transit": ["highway"]
        }
    ]

For the ``list-like`` input format, we allow `.txt` file and `.csv` file. The label words for a class is separated by
a comma, and the label words for different classes are written in consecutive lines.
You can also separate multiple verbalizers by entering empty line(s).

.. code-block::

    politics,government,diplomatic,law
    sports,athletics,gymnastics,sportsman
    business,commerce,trade,market,retail,traffic
    technology,engineering,science

    diplomatic,law
    sports,sportsman
    business,commerce,traffic
    technology,engineering

When loading from a file with multiple verbalizers, you can index the verbalizer you want using the `choice` key words, e.g.:

.. code-block:: python

    ManualVerbalizer(...).from_file(file_path = file_path, choice = 0)
