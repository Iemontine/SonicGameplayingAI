from ast import Sub
from pylatex.utils import italic, NoEscape
from pylatex import Document, Section, Subsection, Command, Tabular, Math, TikZ, Axis, Plot, Figure, Matrix, Alignat, NewPage
from pylatex.utils import italic
import numpy as np

image_filename = "Pass1-3_500k_-0.01punishment_2.jpg"

geometry_options = {
    "margin": "1in",
    "includeheadfoot": False
}
doc = Document(geometry_options=geometry_options, lmodern = True)

doc.preamble.append(Command('title', NoEscape(r'Gameplaying AI: Reimplementing Proximal Policy Optimization\\ \large Introduction to Artificial Intelligence Project\\ \large ECS 170 Fall 2024')))
doc.preamble.append(Command('author', NoEscape(r'Darroll Saddi^\(1\), Andrew Yeow^\(1\), Christine Morayata^\(?\), Julia Heiler^\(?\), Ryan Li^\(1\), Steven Yi^\(?\)\\ \small^\(1\)University of California, Davis - Computer Science\\\small^\(2\)University of California, Davis - Cognitive Science')))
doc.preamble.append(Command('date', NoEscape(r'\today')))

doc.append(NoEscape(r'\maketitle'))

# Add a title section
doc.append(NoEscape(r'\begin{abstract}This report documents a retrospective reimplementation of proximal policy optimization (PPO), performed at UC Davis for study purposes. The algorithm was used to succesfully train a model to play Sonic the Hedgehog^\(TM\). We also present this report as an insightful and educational resource for those interested in reinforcement learning and/or recreating our training environment.\end{abstract}'))

# Add a section
with doc.create(Section('Introduction')):
    doc.append(NoEscape(r'Here is an example of a citation\footnote{This is a reference}. Here is another reference\footnote{This is another reference}. This is how I would cite the first reference again^\(1\).'))
with doc.create(Section('The simple stuff')):
    doc.append('Some regular text and some')
    doc.append(italic('italic text. '))
    doc.append('\nAlso some crazy characters: $&#{}')
    with doc.create(Subsection('Math that is incorrect')):
        doc.append(Math(data=['2*3', '=', 9]))

    with doc.create(Subsection('Table of something')):
        with doc.create(Tabular('rc|cl')) as table:
            table.add_hline()
            table.add_row((1, 2, 3, 4))
            table.add_hline(1, 2)
            table.add_empty_row()
            table.add_row((4, 5, 6, 7))

a = np.array([[100, 10, 20]]).T
M = np.matrix([[2, 3, 4],
                [0, 0, 1],
                [0, 0, 2]])

with doc.create(Section('The fancy stuff')):
    with doc.create(Subsection('Correct matrix equations')):
        doc.append(Math(data=[Matrix(M), Matrix(a), '=', Matrix(M * a)]))

    with doc.create(Subsection('Alignat math environment')):
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append(r'\frac{a}{b} &= 0 \\')

    with doc.create(Subsection('Cool graph')):
        with doc.create(Figure(position='h!')) as img:
            img.add_image(image_filename, width='300px')
            img.add_caption('Look at this photograph')

with doc.create(Section('GitHub')):
    doc.append("A repository containing the code used in this project can be found at:\n\nhttps://github.com/Iemontine/ProximalPolicyOptimization.")

doc.append(NewPage())  # Insert a page break

with doc.create(Section('Contributions')):
    with doc.create(Subsection('Darroll Saddi')):
        doc.append('Set up training environment, which includes integrating libraries & frameworks. Performed inference- and intuition-based hyperparameter tuning to produce training results. Designed reward function & multi-pass implementation. Implemented ways to track metrics during training and wrote plotting code. Contributed to documentation on GitHub, write-ups, and presentation. Contributed research and project direction.')
    with doc.create(Subsection('Steven Yi')):
        doc.append('Refactored OpenAI\'s proximal policy optimization implementation to suit our purposes.')
    with doc.create(Subsection('Andrew Yeow')):
        doc.append('Implemented proof of concept with DQL. Helped with PPO refactoring.')
    with doc.create(Subsection('Julia Heiler')):
        doc.append('Contributed to write-ups.')
    with doc.create(Subsection('Ryan Li')):
        doc.append('Contributed to write-ups.')
    with doc.create(Subsection('Christine Morayata')):
        doc.append('Contributed to write-ups.')
try:
    doc.generate_pdf('output', clean_tex=True)
    print("Completed!")
except Exception as e:
    pass