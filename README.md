# Minerva Grading Dashboard
This project is an assignment grading dashboard for course leads, so that they can import the current grading data for a given assignment (across all active sections of the course) and observe differences in grading patterns between sections.

# How to set up
1. [Download] the "Minerva Assignment Data-grabber" chrome extension.
2. [Download](https://github.com/g-nilsson/Grading-Dashboard/archive/refs/heads/main.zip) this project, remember the path to the folder after you've unzipped it.
3. If you haven't already, [install python](https://www.python.org/downloads/)
4. Install the python libraries, plotly & requests-futures by pasting the following commands in your command prompt <br>
```
pip install plotly==5.23.0
```
```
pip install requests-futures
```

# How to use
1. (Optional) Pin the chrome extension
![Pinning Extension](https://github.com/g-nilsson/public_files/blob/main/pin_extension.gif)
3. Navigate to an assignment page in Forum (of the format https://forum.minerva.edu/app/assignments/xxxxxx)
4. Press the button in the chrome extension popup
![Using the chrome extension](https://github.com/g-nilsson/public_files/blob/main/using_extension.gif)

# Functionality
The script will create an html file containing interactive graphs and statistical results for the given assignment across all sections.
The html will describe and summarize data like:
- ANOVA results, whether there's statistical evidence that at least one section is giving higher/lower scores
- Distribution of students' average assignment scores over all sections
- Distributions of students' LO scores over all sections
